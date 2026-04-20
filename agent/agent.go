package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/Adi-ty/yaca/ai"
)

// Agent runs the tool-call loop for a single conversation session.
type Agent struct {
	cfg Config

	// mu protects state.Messages — taken as RLock for reads, Lock for appends.
	mu    sync.RWMutex
	state State

	// subsMu protects the subs slice itself; individual channel sends happen
	// outside this lock so a slow subscriber cannot starve others.
	subsMu sync.Mutex
	subs   []chan Event

	// loopMu serialises concurrent Send calls so only one loop runs at a time.
	loopMu sync.Mutex
}

// New constructs an Agent from cfg.
func New(cfg Config) *Agent {
	return &Agent{
		cfg: cfg,
		state: State{
			Model:  cfg.Model,
			System: cfg.System,
		},
	}
}

// State returns a shallow-copy snapshot safe to read from any goroutine.
func (a *Agent) State() State {
	a.mu.RLock()
	defer a.mu.RUnlock()
	msgs := make([]ai.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)
	return State{
		Model:    a.state.Model,
		System:   a.state.System,
		Messages: msgs,
	}
}

// Subscribe returns a channel that receives every Event for the lifetime of
// this agent. The channel is buffered; callers should read promptly.
func (a *Agent) Subscribe() <-chan Event {
	ch := make(chan Event, 256)
	a.subsMu.Lock()
	a.subs = append(a.subs, ch)
	a.subsMu.Unlock()
	return ch
}

// Send appends userText as a user message and starts the agent loop in a
// goroutine. Concurrent calls are serialised (the second waits for the first).
func (a *Agent) Send(ctx context.Context, userText string) {
	go func() {
		a.loopMu.Lock()
		defer a.loopMu.Unlock()
		a.loop(ctx, userText)
	}()
}

// ── core loop ─────────────────────────────────────────────────────────────────

func (a *Agent) loop(ctx context.Context, userText string) {
	a.emit(Event{Type: EventAgentStart})

	a.appendMessage(ai.Message{
		Role:    ai.RoleUser,
		Content: []ai.ContentBlock{{Type: ai.ContentTypeText, Text: userText}},
	})

	for {
		a.emit(Event{Type: EventTurnStart})

		// Build request from a snapshot so the mutex isn't held during I/O.
		a.mu.RLock()
		req := ai.Request{
			Model:       a.state.Model,
			System:      a.state.System,
			Messages:    append([]ai.Message(nil), a.state.Messages...),
			Tools:       a.toolSchemas(),
			MaxTokens:   a.cfg.MaxTokens,
			Temperature: a.cfg.Temperature,
		}
		a.mu.RUnlock()

		ch, err := a.cfg.Provider.Stream(ctx, req)
		if err != nil {
			a.emit(Event{Type: EventError, Err: fmt.Errorf("agent: start stream: %w", err)})
			a.emit(Event{Type: EventAgentEnd})
			return
		}

		assistantBlocks, ok := a.drainStream(ch)
		if !ok {
			// EventError already emitted inside drainStream.
			a.emit(Event{Type: EventAgentEnd})
			return
		}

		a.appendMessage(ai.Message{Role: ai.RoleAssistant, Content: assistantBlocks})
		a.emit(Event{Type: EventTurnEnd})

		toolCalls := filterType(assistantBlocks, ai.ContentTypeToolCall)
		if len(toolCalls) == 0 {
			a.emit(Event{Type: EventAgentEnd})
			return
		}

		toolResults := a.executeTools(ctx, toolCalls)
		a.appendMessage(ai.Message{Role: ai.RoleUser, Content: toolResults})
	}
}

// drainStream reads from ch, emits subscriber events, and returns the fully
// assembled assistant content blocks. Returns false on EventError.
func (a *Agent) drainStream(ch <-chan ai.StreamEvent) ([]ai.ContentBlock, bool) {
	var blocks []ai.ContentBlock
	var textBuf, thinkBuf strings.Builder

	flushText := func() {
		if textBuf.Len() > 0 {
			blocks = append(blocks, ai.ContentBlock{Type: ai.ContentTypeText, Text: textBuf.String()})
			textBuf.Reset()
		}
	}
	flushThink := func() {
		if thinkBuf.Len() > 0 {
			blocks = append(blocks, ai.ContentBlock{Type: ai.ContentTypeThinking, Text: thinkBuf.String()})
			thinkBuf.Reset()
		}
	}

	for event := range ch {
		switch event.Type {
		case ai.EventText:
			textBuf.WriteString(event.Delta)
			a.emit(Event{Type: EventTextDelta, Delta: event.Delta})

		case ai.EventThinking:
			thinkBuf.WriteString(event.Delta)
			a.emit(Event{Type: EventThinkingDelta, Delta: event.Delta})

		case ai.EventToolCall:
			flushThink()
			flushText()
			blocks = append(blocks, ai.ContentBlock{
				Type:       ai.ContentTypeToolCall,
				ToolCallID: event.ToolCallID,
				ToolName:   event.ToolName,
				ToolInput:  event.ToolInput,
			})
			a.emit(Event{
				Type:       EventToolCallStart,
				ToolCallID: event.ToolCallID,
				ToolName:   event.ToolName,
				ToolInput:  event.ToolInput,
			})

		case ai.EventError:
			a.emit(Event{Type: EventError, Err: event.Err})
			return nil, false
		}
	}

	flushThink()
	flushText()
	return blocks, true
}

// executeTools runs each tool call sequentially and returns the result blocks.
func (a *Agent) executeTools(ctx context.Context, calls []ai.ContentBlock) []ai.ContentBlock {
	results := make([]ai.ContentBlock, len(calls))
	for i, call := range calls {
		result := a.executeTool(ctx, call)
		results[i] = result
		a.emit(Event{
			Type:       EventToolCallEnd,
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			ToolInput:  call.ToolInput,
			ToolResult: result.ToolResultContent,
			IsError:    result.IsError,
		})
		a.emit(Event{
			Type:       EventToolResult,
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			ToolResult: result.ToolResultContent,
			IsError:    result.IsError,
		})
	}
	return results
}

// executeTool dispatches a single tool call and returns a ToolResult block.
func (a *Agent) executeTool(ctx context.Context, call ai.ContentBlock) ai.ContentBlock {
	result := ai.ContentBlock{
		Type:         ai.ContentTypeToolResult,
		ToolResultID: call.ToolCallID,
	}
	for i := range a.cfg.Tools {
		if a.cfg.Tools[i].Name == call.ToolName {
			output, err := a.cfg.Tools[i].Execute(ctx, call.ToolInput)
			if err != nil {
				result.ToolResultContent = err.Error()
				result.IsError = true
			} else {
				result.ToolResultContent = output
			}
			return result
		}
	}
	result.ToolResultContent = fmt.Sprintf("tool %q not found", call.ToolName)
	result.IsError = true
	return result
}

// ── helpers ───────────────────────────────────────────────────────────────────

func (a *Agent) appendMessage(msg ai.Message) {
	a.mu.Lock()
	a.state.Messages = append(a.state.Messages, msg)
	a.mu.Unlock()
}

// emit copies the subscriber slice (under lock) then sends outside the lock so
// a slow subscriber cannot block other sends or a concurrent Subscribe call.
func (a *Agent) emit(e Event) {
	a.subsMu.Lock()
	subs := make([]chan Event, len(a.subs))
	copy(subs, a.subs)
	a.subsMu.Unlock()

	for _, ch := range subs {
		ch <- e
	}
}

func (a *Agent) toolSchemas() []ai.ToolSchema {
	schemas := make([]ai.ToolSchema, len(a.cfg.Tools))
	for i, t := range a.cfg.Tools {
		schemas[i] = ai.ToolSchema{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
	return schemas
}

// filterType returns all blocks whose Type matches t.
func filterType(blocks []ai.ContentBlock, t ai.ContentType) []ai.ContentBlock {
	var out []ai.ContentBlock
	for _, b := range blocks {
		if b.Type == t {
			out = append(out, b)
		}
	}
	return out
}
