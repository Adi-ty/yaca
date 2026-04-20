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
	anthropicAPIBase  = "https://api.anthropic.com"
	anthropicVersion  = "2023-06-01"
	defaultMaxTokens  = 8192
)

// AnthropicProvider implements Provider against Anthropic's Messages API.
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
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	Thinking  string         `json:"thinking,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	ToolUseID string         `json:"tool_use_id,omitempty"`
	// tool_result content may be a plain string
	Content string `json:"content,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
}

type anthropicWireMessage struct {
	Role    string           `json:"role"`
	Content []anthropicBlock `json:"content"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicReq struct {
	Model     string                 `json:"model"`
	MaxTokens int                    `json:"max_tokens"`
	System    string                 `json:"system,omitempty"`
	Messages  []anthropicWireMessage `json:"messages"`
	Tools     []anthropicTool        `json:"tools,omitempty"`
	Stream    bool                   `json:"stream"`
}

// ── SSE event shapes ──────────────────────────────────────────────────────────

type sseMessageStart struct {
	Message struct {
		Usage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message"`
}

type sseBlockStart struct {
	Index        int `json:"index"`
	ContentBlock struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"content_block"`
}

type sseBlockDelta struct {
	Index int `json:"index"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
}

type sseBlockStop struct {
	Index int `json:"index"`
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

func toWireMessages(msgs []Message) []anthropicWireMessage {
	out := make([]anthropicWireMessage, len(msgs))
	for i, m := range msgs {
		wm := anthropicWireMessage{Role: string(m.Role)}
		for _, cb := range m.Content {
			wm.Content = append(wm.Content, toWireBlock(cb))
		}
		out[i] = wm
	}
	return out
}

func toWireBlock(cb ContentBlock) anthropicBlock {
	switch cb.Type {
	case ContentTypeText:
		return anthropicBlock{Type: "text", Text: cb.Text}
	case ContentTypeThinking:
		return anthropicBlock{Type: "thinking", Thinking: cb.Text}
	case ContentTypeToolCall:
		return anthropicBlock{
			Type:  "tool_use",
			ID:    cb.ToolCallID,
			Name:  cb.ToolName,
			Input: cb.ToolInput,
		}
	case ContentTypeToolResult:
		return anthropicBlock{
			Type:      "tool_result",
			ToolUseID: cb.ToolResultID,
			Content:   cb.ToolResultContent,
			IsError:   cb.IsError,
		}
	default:
		return anthropicBlock{Type: "text", Text: cb.Text}
	}
}

func toWireTools(tools []ToolSchema) []anthropicTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropicTool, len(tools))
	for i, t := range tools {
		out[i] = anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
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

// blockAccum accumulates state for one content block while streaming.
type blockAccum struct {
	kind      string // "text" | "tool_use" | "thinking"
	toolID    string
	toolName  string
	jsonBuf   strings.Builder
}

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
	go p.consumeSSE(ctx, resp, ch)
	return ch, nil
}

func (p *AnthropicProvider) consumeSSE(ctx context.Context, resp *http.Response, ch chan<- StreamEvent) {
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

	blocks := map[int]*blockAccum{}
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
			if !p.handleSSEData(eventType, data, blocks,
				&inputTokens, &outputTokens, &stopReason, send) {
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		send(StreamEvent{Type: EventError, Err: fmt.Errorf("anthropic: SSE read: %w", err)})
	}
}

// handleSSEData processes one (eventType, data) pair. Returns false if the
// caller should stop reading (context done or terminal error).
func (p *AnthropicProvider) handleSSEData(
	eventType, data string,
	blocks map[int]*blockAccum,
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

	case "content_block_start":
		var e sseBlockStart
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			return send(StreamEvent{Type: EventError,
				Err: fmt.Errorf("anthropic: parse content_block_start: %w", err)})
		}
		ba := &blockAccum{kind: e.ContentBlock.Type}
		if e.ContentBlock.Type == "tool_use" {
			ba.toolID = e.ContentBlock.ID
			ba.toolName = e.ContentBlock.Name
		}
		blocks[e.Index] = ba

	case "content_block_delta":
		var e sseBlockDelta
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			return send(StreamEvent{Type: EventError,
				Err: fmt.Errorf("anthropic: parse content_block_delta: %w", err)})
		}
		ba := blocks[e.Index]
		if ba == nil {
			break
		}
		switch e.Delta.Type {
		case "text_delta":
			if !send(StreamEvent{Type: EventText, Delta: e.Delta.Text}) {
				return false
			}
		case "thinking_delta":
			if !send(StreamEvent{Type: EventThinking, Delta: e.Delta.Thinking}) {
				return false
			}
		case "input_json_delta":
			ba.jsonBuf.WriteString(e.Delta.PartialJSON)
		}

	case "content_block_stop":
		var e sseBlockStop
		if err := json.Unmarshal([]byte(data), &e); err != nil {
			break
		}
		ba := blocks[e.Index]
		if ba == nil || ba.kind != "tool_use" {
			break
		}
		var input map[string]any
		if raw := ba.jsonBuf.String(); raw != "" {
			if err := json.Unmarshal([]byte(raw), &input); err != nil {
				return send(StreamEvent{Type: EventError,
					Err: fmt.Errorf("anthropic: unmarshal tool input JSON: %w", err)})
			}
		}
		if !send(StreamEvent{
			Type:       EventToolCall,
			ToolCallID: ba.toolID,
			ToolName:   ba.toolName,
			ToolInput:  input,
		}) {
			return false
		}

	case "message_delta":
		var e sseMessageDelta
		if err := json.Unmarshal([]byte(data), &e); err == nil {
			*stopReason = toStopReason(e.Delta.StopReason)
			*outputTokens = e.Usage.OutputTokens
		}

	case "message_stop":
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
		Model:     req.Model,
		MaxTokens: maxTok,
		System:    req.System,
		Messages:  toWireMessages(req.Messages),
		Tools:     toWireTools(req.Tools),
		Stream:    true,
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
