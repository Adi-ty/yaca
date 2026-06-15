package tests

import (
	"strings"
	"testing"

	"github.com/Adi-ty/yaca/ai"
)

// firstToolCall returns the first tool-call block in blocks, or fails.
func firstToolCall(t *testing.T, blocks []ai.ContentBlock) ai.ContentBlock {
	t.Helper()
	for _, b := range blocks {
		if b.Type == ai.ContentTypeToolCall {
			return b
		}
	}
	t.Fatalf("no tool call found in %+v", blocks)
	return ai.ContentBlock{}
}

func joinText(blocks []ai.ContentBlock) string {
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type == ai.ContentTypeText {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

func countToolCalls(blocks []ai.ContentBlock) int {
	n := 0
	for _, b := range blocks {
		if b.Type == ai.ContentTypeToolCall {
			n++
		}
	}
	return n
}

func TestParse_PlainTextNoTool(t *testing.T) {
	blocks := ai.ParseAssistantText("Just a normal answer with no tool call.")
	if countToolCalls(blocks) != 0 {
		t.Fatalf("expected no tool calls, got %d", countToolCalls(blocks))
	}
	if got := joinText(blocks); got != "Just a normal answer with no tool call." {
		t.Errorf("text mismatch: %q", got)
	}
}

func TestParse_SingleToolCall(t *testing.T) {
	in := `Sure.
<tool_call>
{"name": "write", "arguments": {"path": "a.txt", "content": "hi"}}
</tool_call>`
	blocks := ai.ParseAssistantText(in)
	tc := firstToolCall(t, blocks)
	if tc.ToolName != "write" {
		t.Fatalf("name = %q, want write", tc.ToolName)
	}
	if tc.ToolInput["path"] != "a.txt" || tc.ToolInput["content"] != "hi" {
		t.Errorf("args mismatch: %+v", tc.ToolInput)
	}
	if !strings.Contains(joinText(blocks), "Sure.") {
		t.Errorf("leading text lost: %q", joinText(blocks))
	}
}

func TestParse_ArgumentsAsJSONString(t *testing.T) {
	// Some models emit arguments as a JSON-encoded string rather than an object.
	in := `<tool_call>{"name": "read", "arguments": "{\"path\": \"go.mod\"}"}</tool_call>`
	tc := firstToolCall(t, ai.ParseAssistantText(in))
	if tc.ToolInput["path"] != "go.mod" {
		t.Errorf("stringified args not parsed: %+v", tc.ToolInput)
	}
}

func TestParse_ParametersAlias(t *testing.T) {
	in := `<tool_call>{"name": "read", "parameters": {"path": "go.mod"}}</tool_call>`
	tc := firstToolCall(t, ai.ParseAssistantText(in))
	if tc.ToolInput["path"] != "go.mod" {
		t.Errorf("parameters alias not parsed: %+v", tc.ToolInput)
	}
}

func TestParse_FencedJSON(t *testing.T) {
	in := "<tool_call>\n```json\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n```\n</tool_call>"
	tc := firstToolCall(t, ai.ParseAssistantText(in))
	if tc.ToolName != "bash" || tc.ToolInput["command"] != "ls" {
		t.Errorf("fenced JSON not parsed: %+v", tc)
	}
}

func TestParse_TrailingProse(t *testing.T) {
	// A balanced object followed by commentary should still parse.
	in := `<tool_call>{"name": "read", "arguments": {"path": "go.mod"}}  (reading the module file)</tool_call>`
	tc := firstToolCall(t, ai.ParseAssistantText(in))
	if tc.ToolName != "read" {
		t.Errorf("trailing prose broke parse: %+v", tc)
	}
}

func TestParse_UnclosedTagAfterStop(t *testing.T) {
	// The ToolStopSeq strips the closing tag, so a complete call ends mid-stream
	// with an open <tool_call> and no </tool_call>.
	in := `<tool_call>
{"name": "glob", "arguments": {"pattern": "**/*.go"}}`
	tc := firstToolCall(t, ai.ParseAssistantText(in))
	if tc.ToolName != "glob" || tc.ToolInput["pattern"] != "**/*.go" {
		t.Errorf("unclosed tool call not recovered: %+v", tc)
	}
}

func TestParse_MultipleToolCalls(t *testing.T) {
	in := `<tool_call>{"name": "read", "arguments": {"path": "a"}}</tool_call>
<tool_call>{"name": "read", "arguments": {"path": "b"}}</tool_call>`
	blocks := ai.ParseAssistantText(in)
	if n := countToolCalls(blocks); n != 2 {
		t.Fatalf("expected 2 tool calls, got %d", n)
	}
}

func TestParse_MalformedStaysText(t *testing.T) {
	// Not valid JSON and no balanced object — must not become a tool call.
	in := `<tool_call>not json at all</tool_call>`
	blocks := ai.ParseAssistantText(in)
	if countToolCalls(blocks) != 0 {
		t.Fatalf("malformed tool call should not parse, got %d calls", countToolCalls(blocks))
	}
	if !strings.Contains(joinText(blocks), "not json at all") {
		t.Errorf("malformed content lost: %q", joinText(blocks))
	}
}

func TestParse_BareJSONCall(t *testing.T) {
	// Many local models (e.g. qwen2.5-coder) emit the call as bare JSON with no
	// <tool_call> tags. It must still parse when the name is a known tool.
	in := `{"name": "write", "arguments": {"path": "hello.txt", "content": "HELLO"}}`
	tc := firstToolCall(t, ai.ParseAssistantText(in, "write", "read"))
	if tc.ToolName != "write" || tc.ToolInput["path"] != "hello.txt" {
		t.Errorf("bare JSON call not parsed: %+v", tc)
	}
}

func TestParse_FencedBareJSONCall(t *testing.T) {
	in := "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"x\"}}\n```"
	tc := firstToolCall(t, ai.ParseAssistantText(in, "write"))
	if tc.ToolName != "write" || tc.ToolInput["content"] != "x" {
		t.Errorf("fenced bare JSON call not parsed: %+v", tc)
	}
}

func TestParse_BareJSON_UnknownNameStaysText(t *testing.T) {
	// A JSON object whose name is not a registered tool must NOT be treated as a
	// call (avoids misreading JSON content in an ordinary answer).
	in := `Here is an example: {"name": "not_a_tool", "arguments": {}}`
	blocks := ai.ParseAssistantText(in, "write", "read")
	if countToolCalls(blocks) != 0 {
		t.Fatalf("unknown-name JSON should stay text, got %d calls", countToolCalls(blocks))
	}
}

func TestParse_PlainAnswerWithCodeFenceUntouched(t *testing.T) {
	// An ordinary answer with a code fence and no tool call must be preserved.
	in := "Here is code:\n```go\nfunc main() {}\n```\nDone."
	blocks := ai.ParseAssistantText(in, "write", "read")
	if countToolCalls(blocks) != 0 {
		t.Fatalf("plain answer should have no tool calls, got %d", countToolCalls(blocks))
	}
	if got := joinText(blocks); !strings.Contains(got, "```go") || !strings.Contains(got, "func main() {}") {
		t.Errorf("code fence not preserved: %q", got)
	}
}

func TestBuildToolSystemPrompt(t *testing.T) {
	tools := []ai.ToolSchema{{
		Name:        "read",
		Description: "Read a file",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"path": map[string]any{"type": "string", "description": "file path"}},
			"required":   []string{"path"},
		},
	}}
	out := ai.BuildToolSystemPrompt("BASE", tools)
	for _, want := range []string{"BASE", "<tools>", "read", "Read a file", "path", "<tool_call>"} {
		if !strings.Contains(out, want) {
			t.Errorf("system prompt missing %q\n%s", want, out)
		}
	}
}

func TestStopSequences(t *testing.T) {
	withTools := ai.StopSequences(ai.Request{Tools: []ai.ToolSchema{{Name: "x"}}})
	if len(withTools) != 1 || withTools[0] != ai.ToolStopSeq {
		t.Errorf("expected [%q], got %v", ai.ToolStopSeq, withTools)
	}
	if got := ai.StopSequences(ai.Request{}); len(got) != 0 {
		t.Errorf("expected no stop sequences without tools, got %v", got)
	}
}

func TestFlattenMessages_RoundTrip(t *testing.T) {
	msgs := []ai.Message{
		{Role: ai.RoleUser, Content: []ai.ContentBlock{{Type: ai.ContentTypeText, Text: "do it"}}},
		{Role: ai.RoleAssistant, Content: []ai.ContentBlock{{
			Type: ai.ContentTypeToolCall, ToolName: "read", ToolInput: map[string]any{"path": "go.mod"},
		}}},
		{Role: ai.RoleUser, Content: []ai.ContentBlock{{
			Type: ai.ContentTypeToolResult, ToolResultContent: "module yaca",
		}}},
	}
	flat := ai.FlattenMessages(msgs)
	if len(flat) != 3 {
		t.Fatalf("expected 3 flat messages, got %d", len(flat))
	}
	if !strings.Contains(flat[1].Text, "<tool_call>") || !strings.Contains(flat[1].Text, "read") {
		t.Errorf("assistant tool call not serialised: %q", flat[1].Text)
	}
	if !strings.Contains(flat[2].Text, "<tool_response>") || !strings.Contains(flat[2].Text, "module yaca") {
		t.Errorf("tool result not serialised: %q", flat[2].Text)
	}

	// The serialised tool call must round-trip back through the parser.
	tc := firstToolCall(t, ai.ParseAssistantText(flat[1].Text))
	if tc.ToolName != "read" || tc.ToolInput["path"] != "go.mod" {
		t.Errorf("round-trip failed: %+v", tc)
	}
}
