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

// openAIProvider implements Provider for any OpenAI-compatible endpoint.
// Both OpenAI and Ollama use the same wire format; only baseURL and auth differ.
type openAIProvider struct {
	name       string
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewOpenAI returns a provider targeting api.openai.com.
func NewOpenAI(apiKey string) *openAIProvider {
	return &openAIProvider{
		name:       "openai",
		baseURL:    openAIAPIBase,
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
	Role       string        `json:"role"`
	Content    string        `json:"content,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
}

type oaiToolCall struct {
	ID       string      `json:"id"`
	Type     string      `json:"type"`
	Function oaiFunction `json:"function"`
}

type oaiFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type oaiTool struct {
	Type     string     `json:"type"`
	Function oaiToolDef `json:"function"`
}

type oaiToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type oaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaiRequest struct {
	Model         string            `json:"model"`
	Messages      []oaiMessage      `json:"messages"`
	Tools         []oaiTool         `json:"tools,omitempty"`
	MaxTokens     int               `json:"max_tokens,omitempty"`
	Temperature   *float64          `json:"temperature,omitempty"`
	Stream        bool              `json:"stream"`
	StreamOptions *oaiStreamOptions `json:"stream_options,omitempty"`
}

// streaming chunk shape
type oaiChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// ── message conversion ────────────────────────────────────────────────────────

// toOAIMessages flattens our message list into OpenAI's format.
//
// Key differences from our model:
//   - System prompt lives at the front as a system-role message.
//   - Tool results become individual "tool"-role messages (one per block).
//   - Assistant content + tool_calls coexist in a single message.
//   - Thinking blocks are folded into the text content (OpenAI has no native thinking type).
func toOAIMessages(req Request) []oaiMessage {
	var out []oaiMessage

	if req.System != "" {
		out = append(out, oaiMessage{Role: "system", Content: req.System})
	}

	for _, m := range req.Messages {
		switch m.Role {
		case RoleSystem:
			for _, cb := range m.Content {
				if cb.Type == ContentTypeText {
					out = append(out, oaiMessage{Role: "system", Content: cb.Text})
				}
			}

		case RoleUser:
			var texts []string
			for _, cb := range m.Content {
				switch cb.Type {
				case ContentTypeText:
					texts = append(texts, cb.Text)
				case ContentTypeToolResult:
					if len(texts) > 0 {
						out = append(out, oaiMessage{Role: "user", Content: strings.Join(texts, "\n")})
						texts = nil
					}
					out = append(out, oaiMessage{
						Role:       "tool",
						ToolCallID: cb.ToolResultID,
						Content:    cb.ToolResultContent,
					})
				}
			}
			if len(texts) > 0 {
				out = append(out, oaiMessage{Role: "user", Content: strings.Join(texts, "\n")})
			}

		case RoleAssistant:
			msg := oaiMessage{Role: "assistant"}
			var texts []string
			for _, cb := range m.Content {
				switch cb.Type {
				case ContentTypeText, ContentTypeThinking:
					texts = append(texts, cb.Text)
				case ContentTypeToolCall:
					argBytes, _ := json.Marshal(cb.ToolInput)
					msg.ToolCalls = append(msg.ToolCalls, oaiToolCall{
						ID:   cb.ToolCallID,
						Type: "function",
						Function: oaiFunction{
							Name:      cb.ToolName,
							Arguments: string(argBytes),
						},
					})
				}
			}
			if len(texts) > 0 {
				msg.Content = strings.Join(texts, "\n")
			}
			out = append(out, msg)
		}
	}
	return out
}

func toOAITools(tools []ToolSchema) []oaiTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]oaiTool, len(tools))
	for i, t := range tools {
		out[i] = oaiTool{
			Type: "function",
			Function: oaiToolDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}
	return out
}

func toOAIStopReason(s string) StopReason {
	switch s {
	case "stop":
		return StopReasonEndTurn
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
	go p.consumeSSE(ctx, resp, ch)
	return ch, nil
}

// toolAccum holds partial state for one tool call while streaming.
type toolAccum struct {
	id      string
	name    string
	argsBuf strings.Builder
}

func (p *openAIProvider) consumeSSE(ctx context.Context, resp *http.Response, ch chan<- StreamEvent) {
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

	tools := map[int]*toolAccum{}
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
			// Flush accumulated tool calls in index order.
			for idx := 0; ; idx++ {
				ta, ok := tools[idx]
				if !ok {
					break
				}
				var input map[string]any
				if raw := ta.argsBuf.String(); raw != "" {
					if err := json.Unmarshal([]byte(raw), &input); err != nil {
						send(StreamEvent{Type: EventError,
							Err: fmt.Errorf("%s: unmarshal tool arguments: %w", p.name, err)})
						return
					}
				}
				if !send(StreamEvent{
					Type:       EventToolCall,
					ToolCallID: ta.id,
					ToolName:   ta.name,
					ToolInput:  input,
				}) {
					return
				}
			}
			send(StreamEvent{
				Type:       EventDone,
				StopReason: stopReason,
				Usage:      Usage{InputTokens: inputTokens, OutputTokens: outputTokens},
			})
			return
		}

		var chunk oaiChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // tolerate occasional malformed keep-alive chunks
		}

		// Usage arrives in its own final chunk when stream_options.include_usage=true.
		if chunk.Usage != nil {
			inputTokens = chunk.Usage.PromptTokens
			outputTokens = chunk.Usage.CompletionTokens
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				if !send(StreamEvent{Type: EventText, Delta: choice.Delta.Content}) {
					return
				}
			}

			for _, tc := range choice.Delta.ToolCalls {
				ta, ok := tools[tc.Index]
				if !ok {
					ta = &toolAccum{}
					tools[tc.Index] = ta
				}
				if tc.ID != "" {
					ta.id = tc.ID
				}
				if tc.Function.Name != "" {
					ta.name = tc.Function.Name
				}
				ta.argsBuf.WriteString(tc.Function.Arguments)
			}

			if choice.FinishReason != nil && *choice.FinishReason != "" {
				stopReason = toOAIStopReason(*choice.FinishReason)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		send(StreamEvent{Type: EventError, Err: fmt.Errorf("%s: SSE read: %w", p.name, err)})
	}
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
		Messages:    toOAIMessages(req),
		Tools:       toOAITools(req.Tools),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      true,
		StreamOptions: &oaiStreamOptions{IncludeUsage: true},
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
