package ai

import "context"

// Role identifies the author of a message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// ContentType is the discriminant for ContentBlock.
type ContentType string

const (
	ContentTypeText       ContentType = "text"
	ContentTypeToolCall   ContentType = "tool_call"
	ContentTypeToolResult ContentType = "tool_result"
	ContentTypeThinking   ContentType = "thinking"
	ContentTypeImage      ContentType = "image"
)

// ContentBlock is a tagged union representing one piece of message content.
type ContentBlock struct {
	Type ContentType

	// ContentTypeText / ContentTypeThinking
	Text string

	// ContentTypeToolCall
	ToolCallID string
	ToolName   string
	ToolInput  map[string]any

	// ContentTypeToolResult
	ToolResultID      string // matches ToolCallID of the call
	ToolResultContent string
	IsError           bool

	// ContentTypeImage
	ImageMediaType string // e.g. "image/png"
	ImageData      []byte // raw bytes
}

// Message is a single turn in a conversation.
type Message struct {
	Role    Role
	Content []ContentBlock
}

// ToolSchema describes a tool the model may call.
type ToolSchema struct {
	Name        string
	Description string
	// InputSchema is a JSON Schema object (map, not a string) describing
	// the tool's input parameters.
	InputSchema map[string]any
}

// StopReason explains why generation stopped.
type StopReason string

const (
	StopReasonEndTurn    StopReason = "end_turn"
	StopReasonToolUse    StopReason = "tool_use"
	StopReasonMaxTokens  StopReason = "max_tokens"
	StopReasonStopSeq    StopReason = "stop_sequence"
)

// Usage holds token-consumption statistics for one request.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// StreamEventType is the discriminant for StreamEvent.
type StreamEventType string

const (
	EventText       StreamEventType = "text"        // incremental text delta
	EventThinking   StreamEventType = "thinking"    // incremental thinking delta
	EventToolCall   StreamEventType = "tool_call"   // complete tool call ready
	EventDone       StreamEventType = "done"        // stream finished
	EventError      StreamEventType = "error"       // terminal error
)

// StreamEvent is emitted on the channel returned by Provider.Stream.
type StreamEvent struct {
	Type StreamEventType

	// EventText / EventThinking
	Delta string

	// EventToolCall — fully assembled when emitted
	ToolCallID string
	ToolName   string
	ToolInput  map[string]any

	// EventDone
	StopReason StopReason
	Usage      Usage

	// EventError
	Err error
}

// Model describes a model offered by a provider.
type Model struct {
	ID          string
	DisplayName string
	ContextLen  int // max context window in tokens, 0 = unknown
}

// Request bundles all inputs for a single inference call.
type Request struct {
	Model      string
	System     string
	Messages   []Message
	Tools      []ToolSchema
	MaxTokens  int
	Temperature *float64 // nil = provider default
}

// Response is the non-streaming result of Provider.Complete.
type Response struct {
	Content    []ContentBlock
	StopReason StopReason
	Usage      Usage
}

// Provider is the unified interface every LLM backend must satisfy.
type Provider interface {
	// Name returns the canonical provider identifier (e.g. "anthropic").
	Name() string

	// ListModels returns the models available from this provider.
	ListModels(ctx context.Context) ([]Model, error)

	// Stream starts a streaming inference request and returns a channel of
	// events. The channel is closed after an EventDone or EventError event.
	// Callers must drain or stop reading before the context is cancelled.
	Stream(ctx context.Context, req Request) (<-chan StreamEvent, error)

	// Complete performs a non-streaming inference request.
	Complete(ctx context.Context, req Request) (Response, error)
}
