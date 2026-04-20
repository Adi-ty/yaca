package agent

import (
	"context"

	"github.com/Adi-ty/yaca/ai"
)

// EventType is the discriminant for Event.
type EventType string

const (
	EventAgentStart    EventType = "agent_start"    // loop begins for a new user message
	EventTurnStart     EventType = "turn_start"     // one LLM call starts
	EventTextDelta     EventType = "text_delta"     // incremental assistant text
	EventThinkingDelta EventType = "thinking_delta" // incremental thinking text
	EventToolCallStart EventType = "tool_call_start" // tool call identified in stream
	EventToolCallEnd   EventType = "tool_call_end"  // tool execution finished
	EventToolResult    EventType = "tool_result"    // tool result appended to conversation
	EventTurnEnd       EventType = "turn_end"       // one LLM call completed
	EventAgentEnd      EventType = "agent_end"      // loop finished (no more tool calls)
	EventError         EventType = "error"          // terminal error; loop stops
)

// Event is emitted on every subscriber channel during agent execution.
type Event struct {
	Type EventType

	// EventTextDelta, EventThinkingDelta
	Delta string

	// EventToolCallStart, EventToolCallEnd, EventToolResult
	ToolCallID string
	ToolName   string
	ToolInput  map[string]any

	// EventToolCallEnd, EventToolResult
	ToolResult string
	IsError    bool

	// EventError
	Err error

	// EventTurnEnd carries the turn's token usage; EventAgentEnd carries the
	// cumulative total across all turns in this Send() call.
	Usage ai.Usage
}

// Tool is an executable capability injected into the agent at startup.
// The agent package never imports tools/ — tools are provided by the caller.
type Tool struct {
	Name        string
	Description string
	InputSchema map[string]any
	Execute     func(ctx context.Context, input map[string]any) (string, error)
}

// State is a point-in-time snapshot of the agent's conversation history.
type State struct {
	Model    string
	System   string
	Messages []ai.Message
}

// Config bundles everything New() needs to construct an Agent.
type Config struct {
	Provider    ai.Provider
	Model       string
	System      string
	Tools       []Tool
	MaxTokens   int
	Temperature *float64
}
