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
	anthropicAPIBase = "https://api.anthropic.com"
	anthropicVersion = "2023-06-01"
	defaultMaxTokens = 8192
)

// AnthropicProvider implements Provider against Anthropic's Messages API.
//
// Tool calling uses the shared XML protocol (see xmltools.go) rather than
// Anthropic's native tool_use blocks: tools are described in the system prompt
// and the conversation is sent as plain-text turns. This keeps every provider
// on one tool-calling code path.
type AnthropicProvider struct {
	apiKey     string
	httpClient *http.Client
}

// NewAnthropicProvider creates a provider using the given API key.
func NewAnthropicProvider(apiKey string) *AnthropicProvider {
	return &AnthropicProvider{
		apiKey:     apiKey,
		httpClient: &http.Client{},
	}
}

func (p *AnthropicProvider) Name() string { return "anthropic" }

// ── wire types ────────────────────────────────────────────────────────────────

type anthropicBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type anthropicWireMessage struct {
	Role    string           `json:"role"`
	Content []anthropicBlock `json:"content"`
}

type anthropicReq struct {
	Model         string                 `json:"model"`
	MaxTokens     int                    `json:"max_tokens"`
	System        string                 `json:"system,omitempty"`
	Messages      []anthropicWireMessage `json:"messages"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Stream        bool                   `json:"stream"`
}

// ── SSE event shapes ──────────────────────────────────────────────────────────

type sseMessageStart struct {
	Message struct {
		Usage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message"`
}

type sseBlockDelta struct {
	Delta struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		Thinking string `json:"thinking"`
	} `json:"delta"`
}

type sseMessageDelta struct {
	Delta struct {
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// ── conversion helpers ────────────────────────────────────────────────────────

// toWireMessages flattens the structured history into single-text-block messages.
// Anthropic accepts only "user" and "assistant" roles; the system prompt is sent
// separately, so any stray system turn is folded into a user message.
func toWireMessages(msgs []Message) []anthropicWireMessage {
	flat := FlattenMessages(msgs)
	out := make([]anthropicWireMessage, 0, len(flat))
	for _, fm := range flat {
		role := "user"
		if fm.Role == RoleAssistant {
			role = "assistant"
		}
		out = append(out, anthropicWireMessage{
			Role:    role,
			Content: []anthropicBlock{{Type: "text", Text: fm.Text}},
		})
	}
	return out
}

func toStopReason(s string) StopReason {
	switch s {
	case "end_turn":
		return StopReasonEndTurn
	case "tool_use":
		return StopReasonToolUse
	case "max_tokens":
		return StopReasonMaxTokens
	case "stop_sequence":
		return StopReasonStopSeq
	default:
		return StopReason(s)
	}
}

// ── ListModels ────────────────────────────────────────────────────────────────

func (p *AnthropicProvider) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		anthropicAPIBase+"/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build list-models request: %w", err)
	}
	p.setHeaders(req)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: list models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, p.apiError(resp)
	}

	var body struct {
		Data []struct {
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("anthropic: decode models response: %w", err)
	}

	models := make([]Model, len(body.Data))
	for i, m := range body.Data {
		models[i] = Model{ID: m.ID, DisplayName: m.DisplayName}
	}
	return models, nil
}

// ── Stream ────────────────────────────────────────────────────────────────────

func (p *AnthropicProvider) Stream(ctx context.Context, req Request) (<-chan StreamEvent, error) {
	body, err := p.marshalRequest(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		anthropicAPIBase+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: build stream request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: do stream request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, p.apiError(resp)
	}

	ch := make(chan StreamEvent, 32)
	go p.consumeSSE(ctx, resp, ch, ToolNameSet(req.Tools))
	return ch, nil
}

func (p *AnthropicProvider) consumeSSE(ctx context.Context, resp *http.Response, ch chan<- StreamEvent, known map[string]bool) {
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
	scanner.Buffer(make([]byte, 1<<20), 1<<20) // 1 MB — tool inputs can be large

	var eventType string

	for scanner.Scan() {
		line := scanner.Text()

		switch {
		case strings.HasPrefix(line, "event: "):
			eventType = strings.TrimPrefix(line, "event: ")

		case strings.HasPrefix(line, "data: "):
			data := strings.TrimPrefix(line, "data: ")
			if !p.handleSSEData(eventType, data, streamer,
				&inputTokens, &outputTokens, &stopReason, send) {
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		send(StreamEvent{Type: EventError, Err: fmt.Errorf("anthropic: SSE read: %w", err)})
	}
}

// handleSSEData processes one (eventType, data) pair. Text deltas are routed
// through the shared toolStreamer; thinking deltas pass straight through.
// Returns false if the caller should stop reading.
func (p *AnthropicProvider) handleSSEData(
	eventType, data string,
	streamer *toolStreamer,
	inputTokens, outputTokens *int,
	stopReason *StopReason,
	send func(StreamEvent) bool,
) bool {
	switch eventType {

	case "message_start":
		var e sseMessageStart
		if err := json.Unmarshal([]byte(data), &e); err == nil {
			*inputTokens = e.Message.Usage.InputTokens
		}

	case "content_block_delta":
		var e sseBlockDelta
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			return send(StreamEvent{Type: EventError,
				Err: fmt.Errorf("anthropic: parse content_block_delta: %w", err)})
		}
		switch e.Delta.Type {
		case "text_delta":
			return streamer.feed(e.Delta.Text)
		case "thinking_delta":
			return send(StreamEvent{Type: EventThinking, Delta: e.Delta.Thinking})
		}

	case "message_delta":
		var e sseMessageDelta
		if err := json.Unmarshal([]byte(data), &e); err == nil {
			*stopReason = toStopReason(e.Delta.StopReason)
			*outputTokens = e.Usage.OutputTokens
		}

	case "message_stop":
		if !streamer.flush() {
			return false
		}
		return send(StreamEvent{
			Type:       EventDone,
			StopReason: *stopReason,
			Usage: Usage{
				InputTokens:  *inputTokens,
				OutputTokens: *outputTokens,
			},
		})
	}

	return true
}

// ── Complete ──────────────────────────────────────────────────────────────────

// Complete drains Stream and assembles a Response.
func (p *AnthropicProvider) Complete(ctx context.Context, req Request) (Response, error) {
	ch, err := p.Stream(ctx, req)
	if err != nil {
		return Response{}, err
	}

	var blocks []ContentBlock
	var textBuf, thinkBuf strings.Builder
	var stopReason StopReason
	var usage Usage

	flushText := func() {
		if textBuf.Len() > 0 {
			blocks = append(blocks, ContentBlock{Type: ContentTypeText, Text: textBuf.String()})
			textBuf.Reset()
		}
	}
	flushThink := func() {
		if thinkBuf.Len() > 0 {
			blocks = append(blocks, ContentBlock{Type: ContentTypeThinking, Text: thinkBuf.String()})
			thinkBuf.Reset()
		}
	}

	for event := range ch {
		switch event.Type {
		case EventText:
			textBuf.WriteString(event.Delta)
		case EventThinking:
			thinkBuf.WriteString(event.Delta)
		case EventToolCall:
			flushThink()
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

	flushThink()
	flushText()

	return Response{Content: blocks, StopReason: stopReason, Usage: usage}, nil
}

// ── internal helpers ──────────────────────────────────────────────────────────

func (p *AnthropicProvider) marshalRequest(req Request) ([]byte, error) {
	maxTok := req.MaxTokens
	if maxTok == 0 {
		maxTok = defaultMaxTokens
	}
	ar := anthropicReq{
		Model:         req.Model,
		MaxTokens:     maxTok,
		System:        BuildToolSystemPrompt(req.System, req.Tools),
		Messages:      toWireMessages(req.Messages),
		StopSequences: StopSequences(req),
		Stream:        true,
	}
	b, err := json.Marshal(ar)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}
	return b, nil
}

func (p *AnthropicProvider) setHeaders(req *http.Request) {
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "text/event-stream")
}

func (p *AnthropicProvider) apiError(resp *http.Response) error {
	var body struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return fmt.Errorf("anthropic: HTTP %d (body unreadable)", resp.StatusCode)
	}
	return fmt.Errorf("anthropic: %s: %s", body.Error.Type, body.Error.Message)
}
