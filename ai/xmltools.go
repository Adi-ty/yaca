package ai

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// This file is the single source of truth for YACA's tool-calling protocol.
//
// Every provider — Anthropic, OpenAI, Ollama, and any OpenAI-compatible
// endpoint — uses the same text-based convention rather than each backend's
// native function-calling wire format. Tools are described in the system prompt
// inside a <tools> block; the model requests one by emitting a <tool_call>
// block; results are returned in <tool_response> blocks.
//
// Local models are inconsistent: some emit the <tool_call> tags as instructed,
// others emit a bare (often markdown-fenced) JSON object. The parser handles
// both — bare JSON is only promoted to a tool call when its "name" matches a
// known tool, which avoids misreading ordinary JSON in an answer.

// ToolStopSeq is sent as a stop sequence whenever tools are available so a
// compliant model halts immediately after emitting a tool call instead of
// hallucinating the tool result and a final answer in the same generation.
const ToolStopSeq = "</tool_call>"

const (
	toolCallOpen  = "<tool_call>"
	toolCallClose = "</tool_call>"
)

// fenceLineRE matches a markdown code-fence line (``` or ```json) on its own.
var fenceLineRE = regexp.MustCompile("(?m)^[ \t]*```[A-Za-z0-9_-]*[ \t]*$")

// StopSequences returns the stop sequences to send for req: the caller's
// sequences plus ToolStopSeq when tools are available.
func StopSequences(req Request) []string {
	if len(req.Tools) == 0 {
		return req.Stop
	}
	return append(append([]string(nil), req.Stop...), ToolStopSeq)
}

// ToolNameSet returns the set of tool names, used to validate bare JSON calls.
func ToolNameSet(tools []ToolSchema) map[string]bool {
	set := make(map[string]bool, len(tools))
	for _, t := range tools {
		set[t.Name] = true
	}
	return set
}

// BuildToolSystemPrompt augments base with the tool-calling contract and a
// <tools> block describing every available tool. With no tools it returns base
// unchanged (e.g. the summariser used by Compact).
func BuildToolSystemPrompt(base string, tools []ToolSchema) string {
	if len(tools) == 0 {
		return base
	}
	var sb strings.Builder
	if base != "" {
		sb.WriteString(base)
		sb.WriteString("\n\n")
	}
	sb.WriteString(`## Tool use

You operate in a loop: emit a tool call, receive its result in a <tool_response>
block, then continue until the task is done.

To call a tool, output EXACTLY one block and nothing after it:
<tool_call>
{"name": "TOOL_NAME", "arguments": {"arg": "value"}}
</tool_call>

Rules:
- Emit ONE <tool_call> block, then stop and wait for the <tool_response>.
- "arguments" MUST be a JSON object using the exact parameter names listed below.
- Never paste file contents as prose or suggest commands for the user to run —
  use the write and bash tools instead.
- When the task is complete, reply with a short plain-text summary and no tool call.

## Available tools
<tools>
`)
	for _, t := range tools {
		sb.WriteString(toolSignature(t))
	}
	sb.WriteString("</tools>")
	return sb.String()
}

// toolSignature renders one tool as a compact, model-friendly signature listing
// its parameters, types, and whether each is required.
func toolSignature(t ToolSchema) string {
	var sb strings.Builder
	sb.WriteString("\n" + t.Name + " — " + t.Description + "\n")

	props, _ := t.InputSchema["properties"].(map[string]any)
	required := requiredSet(t.InputSchema["required"])

	for name, def := range props {
		dm, _ := def.(map[string]any)
		typ, _ := dm["type"].(string)
		desc, _ := dm["description"].(string)
		flag := ""
		if !required[name] {
			flag = " (optional)"
		}
		sb.WriteString("    • " + name + " (" + typ + ")" + flag + ": " + desc + "\n")
	}
	return sb.String()
}

// requiredSet normalises a JSON-Schema "required" value (which may be []string
// when built in-process or []any after a JSON round-trip) into a lookup set.
func requiredSet(v any) map[string]bool {
	set := map[string]bool{}
	switch list := v.(type) {
	case []string:
		for _, s := range list {
			set[s] = true
		}
	case []any:
		for _, e := range list {
			if s, ok := e.(string); ok {
				set[s] = true
			}
		}
	}
	return set
}

// ── message flattening ──────────────────────────────────────────────────────

// FlatMessage is a role + plain-text turn. Under the unified protocol the whole
// conversation is represented as text turns regardless of provider.
type FlatMessage struct {
	Role Role
	Text string
}

// FlattenMessages converts structured history into plain-text turns, serialising
// tool calls as <tool_call> blocks and tool results as <tool_response> blocks so
// a text-only model sees a consistent transcript. Messages whose content
// produces no text are dropped.
func FlattenMessages(msgs []Message) []FlatMessage {
	out := make([]FlatMessage, 0, len(msgs))
	for _, m := range msgs {
		var parts []string
		for _, cb := range m.Content {
			switch cb.Type {
			case ContentTypeText, ContentTypeThinking:
				if cb.Text != "" {
					parts = append(parts, cb.Text)
				}
			case ContentTypeToolCall:
				parts = append(parts, SerializeToolCall(cb))
			case ContentTypeToolResult:
				parts = append(parts, SerializeToolResult(cb))
			}
		}
		if len(parts) == 0 {
			continue
		}
		out = append(out, FlatMessage{Role: m.Role, Text: strings.Join(parts, "\n")})
	}
	return out
}

// SerializeToolCall renders an assistant tool call as a <tool_call> block.
func SerializeToolCall(cb ContentBlock) string {
	args, err := json.Marshal(cb.ToolInput)
	if err != nil || cb.ToolInput == nil {
		args = []byte("{}")
	}
	return toolCallOpen + "\n" +
		`{"name": "` + cb.ToolName + `", "arguments": ` + string(args) + "}\n" +
		toolCallClose
}

// SerializeToolResult renders a tool result as a <tool_response> block.
func SerializeToolResult(cb ContentBlock) string {
	prefix := ""
	if cb.IsError {
		prefix = "ERROR: "
	}
	return "<tool_response>\n" + prefix + cb.ToolResultContent + "\n</tool_response>"
}

// ── response parsing ──────────────────────────────────────────────────────────

// ParseAssistantText splits a complete model response into content blocks (text
// interleaved with tool calls). toolNames lists the registered tools so a bare
// JSON object can be recognised as a call. It is the non-streaming equivalent of
// what providers assemble live.
func ParseAssistantText(text string, toolNames ...string) []ContentBlock {
	known := make(map[string]bool, len(toolNames))
	for _, n := range toolNames {
		known[n] = true
	}
	return segmentsToBlocks(extractSegments(text, known))
}

// toolStreamer accumulates a provider's text deltas and, on flush, emits clean
// EventText + EventToolCall events. Tool calls are only recognised once the full
// turn is buffered, so a bare JSON object split across deltas is handled and no
// partial <tool_call> tag ever leaks to the caller.
type toolStreamer struct {
	send    func(StreamEvent) bool
	known   map[string]bool
	buf     strings.Builder
	callIdx int
}

func newToolStreamer(send func(StreamEvent) bool, known map[string]bool) *toolStreamer {
	return &toolStreamer{send: send, known: known}
}

// feed buffers a delta. Nothing is emitted until flush, because a tool call may
// be a bare JSON object whose shape is only clear once the turn is complete.
func (s *toolStreamer) feed(delta string) bool {
	s.buf.WriteString(delta)
	return true
}

// flush parses the buffered turn and emits its text and tool-call events.
func (s *toolStreamer) flush() bool {
	for _, seg := range extractSegments(s.buf.String(), s.known) {
		switch v := seg.(type) {
		case textSeg:
			if !s.send(StreamEvent{Type: EventText, Delta: v.text}) {
				return false
			}
		case callSeg:
			s.callIdx++
			if !s.send(StreamEvent{
				Type:       EventToolCall,
				ToolCallID: fmt.Sprintf("call_%d", s.callIdx),
				ToolName:   v.name,
				ToolInput:  v.args,
			}) {
				return false
			}
		}
	}
	return true
}

// ── segment extraction ────────────────────────────────────────────────────────

type textSeg struct{ text string }
type callSeg struct {
	name string
	args map[string]any
}

// segmentsToBlocks converts segments into ContentBlocks, merging adjacent text.
func segmentsToBlocks(segs []any) []ContentBlock {
	var blocks []ContentBlock
	var buf strings.Builder
	idx := 0
	flush := func() {
		if buf.Len() > 0 {
			blocks = append(blocks, ContentBlock{Type: ContentTypeText, Text: buf.String()})
			buf.Reset()
		}
	}
	for _, seg := range segs {
		switch v := seg.(type) {
		case textSeg:
			buf.WriteString(v.text)
		case callSeg:
			flush()
			idx++
			blocks = append(blocks, ContentBlock{
				Type:       ContentTypeToolCall,
				ToolCallID: fmt.Sprintf("call_%d", idx),
				ToolName:   v.name,
				ToolInput:  v.args,
			})
		}
	}
	flush()
	return blocks
}

// extractSegments splits text into alternating text and tool-call segments,
// preferring explicit <tool_call> tags and falling back to bare JSON detection.
func extractSegments(text string, known map[string]bool) []any {
	if strings.Contains(text, toolCallOpen) {
		return extractTagged(text)
	}
	return extractBare(text, known)
}

// extractTagged parses explicit <tool_call>…</tool_call> blocks. A trailing open
// tag with no closing tag (the ToolStopSeq strips it) is treated as complete.
func extractTagged(text string) []any {
	var segs []any
	rem := text
	for {
		i := strings.Index(rem, toolCallOpen)
		if i == -1 {
			segs = appendText(segs, rem)
			break
		}
		segs = appendText(segs, rem[:i])
		rem = rem[i+len(toolCallOpen):]

		body, closed := rem, false
		if before, after, found := strings.Cut(rem, toolCallClose); found {
			body, rem, closed = before, after, true
		}
		if name, args, ok := parseToolCallBody(body); ok {
			segs = append(segs, callSeg{name, args})
		} else {
			segs = appendText(segs, toolCallOpen+body)
		}
		if !closed {
			break
		}
	}
	return segs
}

// extractBare detects a tool call expressed as a bare (possibly fenced) JSON
// object. It first confirms a known-tool object exists so ordinary answers —
// including ones containing code fences or JSON — are returned untouched.
func extractBare(text string, known map[string]bool) []any {
	if !containsToolJSON(text, known) {
		return []any{textSeg{text}}
	}
	clean := fenceLineRE.ReplaceAllString(text, "")

	var segs []any
	rem := clean
	for {
		start := strings.IndexByte(rem, '{')
		if start == -1 {
			segs = appendText(segs, rem)
			break
		}
		obj := firstJSONObject(rem[start:])
		if obj == "" {
			segs = appendText(segs, rem)
			break
		}
		if name, args, ok := parseToolCallBody(obj); ok && known[name] {
			segs = appendText(segs, rem[:start])
			segs = append(segs, callSeg{name, args})
		} else {
			segs = appendText(segs, rem[:start+len(obj)])
		}
		rem = rem[start+len(obj):]
	}
	return segs
}

// containsToolJSON reports whether text contains a balanced JSON object whose
// "name" is a known tool.
func containsToolJSON(text string, known map[string]bool) bool {
	rem := text
	for {
		start := strings.IndexByte(rem, '{')
		if start == -1 {
			return false
		}
		obj := firstJSONObject(rem[start:])
		if obj == "" {
			return false
		}
		if name, _, ok := parseToolCallBody(obj); ok && known[name] {
			return true
		}
		rem = rem[start+len(obj):]
	}
}

// appendText appends s as a text segment unless it is blank.
func appendText(segs []any, s string) []any {
	if strings.TrimSpace(s) == "" {
		return segs
	}
	return append(segs, textSeg{s})
}

// parseToolCallBody extracts the tool name and arguments from a JSON tool call,
// tolerating markdown fences and trailing commentary.
func parseToolCallBody(body string) (string, map[string]any, bool) {
	raw := stripFences(strings.TrimSpace(body))

	obj, ok := unmarshalObject(raw)
	if !ok {
		if span := firstJSONObject(raw); span != "" {
			obj, ok = unmarshalObject(span)
		}
	}
	if !ok {
		return "", nil, false
	}

	name, _ := obj["name"].(string)
	if name == "" {
		return "", nil, false
	}
	return name, extractArgs(obj), true
}

// extractArgs pulls the arguments object, tolerating common model variations:
// an "arguments" or "parameters" key, supplied as an object or a JSON string.
func extractArgs(obj map[string]any) map[string]any {
	raw, ok := obj["arguments"]
	if !ok {
		raw = obj["parameters"]
	}
	switch v := raw.(type) {
	case map[string]any:
		return v
	case string:
		if m, ok := unmarshalObject(v); ok {
			return m
		}
	}
	return map[string]any{}
}

func unmarshalObject(s string) (map[string]any, bool) {
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil, false
	}
	return m, true
}

// stripFences removes a surrounding ``` / ```json markdown code fence if present.
func stripFences(s string) string {
	if !strings.HasPrefix(s, "```") {
		return s
	}
	if nl := strings.IndexByte(s, '\n'); nl != -1 {
		s = s[nl+1:]
	}
	s = strings.TrimSuffix(strings.TrimRight(s, "\n "), "```")
	return strings.TrimSpace(s)
}

// firstJSONObject returns the first balanced {…} span in s (respecting strings
// and escapes), or "" if there is none.
func firstJSONObject(s string) string {
	start := strings.IndexByte(s, '{')
	if start == -1 {
		return ""
	}
	depth, inStr, esc := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return ""
}
