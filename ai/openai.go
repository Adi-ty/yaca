package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

const (
	openAIAPIBase = "https://api.openai.com/v1"
	ollamaDefault = "http://localhost:11434/v1"
)

// openAIProvider implements Provider for any OpenAI-compatible chat-completions
// endpoint (OpenAI, Ollama, vLLM, LM Studio, Groq, …). Tool calling uses the
// shared XML protocol (see xmltools.go) rather than native function calling, so
// it behaves identically across hosted and local models — local models in
// particular do not reliably emit native tool_calls.
type openAIProvider struct {
	name       string
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewOpenAI returns a provider for an OpenAI-compatible endpoint. If baseURL is
// empty it defaults to https://api.openai.com/v1; passing a base URL targets
// vLLM / LM Studio / Groq / etc. through the same code path.
func NewOpenAI(apiKey, baseURL string) *openAIProvider {
	if baseURL == "" {
		baseURL = openAIAPIBase
	}
	return &openAIProvider{
		name:       "openai",
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		httpClient: &http.Client{},
	}
}

// NewOllama returns a provider targeting a local (or remote) Ollama instance.
// If baseURL is empty it defaults to http://localhost:11434/v1.
func NewOllama(baseURL string) *openAIProvider {
	if baseURL == "" {
		baseURL = ollamaDefault
	}
	return &openAIProvider{
		name:       "ollama",
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{},
	}
}

func (p *openAIProvider) Name() string { return p.name }

// ── wire types ────────────────────────────────────────────────────────────────

type oaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type oaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaiRequest struct {
	Model         string            `json:"model"`
	Messages      []oaiMessage      `json:"messages"`
	MaxTokens     int               `json:"max_tokens,omitempty"`
	Temperature   *float64          `json:"temperature,omitempty"`
	Stop          []string          `json:"stop,omitempty"`
	Stream        bool              `json:"stream"`
	StreamOptions *oaiStreamOptions `json:"stream_options,omitempty"`
}

// streaming chunk shape — we only consume text content + finish reason + usage.
type oaiChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// ── message conversion ────────────────────────────────────────────────────────

// toMessages converts our message model to OpenAI wire messages. Tools are
// described in the system prompt and tool calls/results are serialised as XML
// text turns — there are no native tool fields on the wire.
func (p *openAIProvider) toMessages(req Request) []oaiMessage {
	var out []oaiMessage
	if sys := BuildToolSystemPrompt(req.System, req.Tools); sys != "" {
		out = append(out, oaiMessage{Role: "system", Content: sys})
	}
	for _, fm := range FlattenMessages(req.Messages) {
		out = append(out, oaiMessage{Role: string(fm.Role), Content: fm.Text})
	}
	return out
}

func toOAIStopReason(s string) StopReason {
	switch s {
	case "tool_calls":
		return StopReasonToolUse
	case "length":
		return StopReasonMaxTokens
	default:
		return StopReasonEndTurn
	}
}

// ── ListModels ────────────────────────────────────────────────────────────────

func (p *openAIProvider) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("%s: build list-models request: %w", p.name, err)
	}
	p.setHeaders(req)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: list models: %w", p.name, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, p.apiError(resp)
	}

	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("%s: decode models response: %w", p.name, err)
	}

	models := make([]Model, len(body.Data))
	for i, m := range body.Data {
		models[i] = Model{ID: m.ID, DisplayName: m.ID}
	}
	return models, nil
}

// ── Stream ────────────────────────────────────────────────────────────────────

func (p *openAIProvider) Stream(ctx context.Context, req Request) (<-chan StreamEvent, error) {
	body, err := p.marshalRequest(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: build stream request: %w", p.name, err)
	}
	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%s: do stream request: %w", p.name, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, p.apiError(resp)
	}

	ch := make(chan StreamEvent, 32)
	go p.consumeSSE(ctx, resp, ch, ToolNameSet(req.Tools))
	return ch, nil
}

// consumeSSE extracts text deltas, usage, and finish reason, routing the text
// through the shared toolStreamer which emits clean EventText + EventToolCall.
func (p *openAIProvider) consumeSSE(ctx context.Context, resp *http.Response, ch chan<- StreamEvent, known map[string]bool) {
	defer resp.Body.Close()
	defer close(ch)

	send := func(e StreamEvent) bool {
		select {
		case ch <- e:
			return true
		case <-ctx.Done():
			return false
		}
	}

	streamer := newToolStreamer(send, known)
	var inputTokens, outputTokens int
	var stopReason StopReason

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk oaiChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Usage != nil {
			inputTokens = chunk.Usage.PromptTokens
			outputTokens = chunk.Usage.CompletionTokens
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				if !streamer.feed(choice.Delta.Content) {
					return
				}
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				stopReason = toOAIStopReason(*choice.FinishReason)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		send(StreamEvent{Type: EventError, Err: fmt.Errorf("%s: SSE read: %w", p.name, err)})
		return
	}
	if !streamer.flush() {
		return
	}
	send(StreamEvent{
		Type:       EventDone,
		StopReason: stopReason,
		Usage:      Usage{InputTokens: inputTokens, OutputTokens: outputTokens},
	})
}

// ── Complete ──────────────────────────────────────────────────────────────────

func (p *openAIProvider) Complete(ctx context.Context, req Request) (Response, error) {
	ch, err := p.Stream(ctx, req)
	if err != nil {
		return Response{}, err
	}

	var blocks []ContentBlock
	var textBuf strings.Builder
	var stopReason StopReason
	var usage Usage

	flushText := func() {
		if textBuf.Len() > 0 {
			blocks = append(blocks, ContentBlock{Type: ContentTypeText, Text: textBuf.String()})
			textBuf.Reset()
		}
	}

	for event := range ch {
		switch event.Type {
		case EventText:
			textBuf.WriteString(event.Delta)
		case EventToolCall:
			flushText()
			blocks = append(blocks, ContentBlock{
				Type:       ContentTypeToolCall,
				ToolCallID: event.ToolCallID,
				ToolName:   event.ToolName,
				ToolInput:  event.ToolInput,
			})
		case EventDone:
			stopReason = event.StopReason
			usage = event.Usage
		case EventError:
			return Response{}, event.Err
		}
	}

	flushText()
	return Response{Content: blocks, StopReason: stopReason, Usage: usage}, nil
}

// ── internal helpers ──────────────────────────────────────────────────────────

func (p *openAIProvider) marshalRequest(req Request) ([]byte, error) {
	ar := oaiRequest{
		Model:       req.Model,
		Messages:    p.toMessages(req),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stop:        StopSequences(req),
		Stream:      true,
	}
	// stream_options.include_usage is an OpenAI-only extension; some strict
	// gateways reject unknown fields, so only send it for OpenAI itself.
	if p.name == "openai" {
		ar.StreamOptions = &oaiStreamOptions{IncludeUsage: true}
	}

	b, err := json.Marshal(ar)
	if err != nil {
		return nil, fmt.Errorf("%s: marshal request: %w", p.name, err)
	}
	return b, nil
}

func (p *openAIProvider) setHeaders(req *http.Request) {
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
}

func (p *openAIProvider) apiError(resp *http.Response) error {
	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return fmt.Errorf("%s: HTTP %d (body unreadable)", p.name, resp.StatusCode)
	}
	return fmt.Errorf("%s: %s: %s", p.name, body.Error.Type, body.Error.Message)
}
