package tests

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Adi-ty/yaca/agent"
	"github.com/Adi-ty/yaca/ai"
	"github.com/Adi-ty/yaca/tools"
)

// TestOllamaE2E_WriteTool drives the full agent loop against a live Ollama
// instance using the unified XML tool-calling path. It is opt-in: set
// YACA_OLLAMA_E2E=1 (and run `ollama serve`) to enable it. Override the model
// with YACA_OLLAMA_MODEL (default qwen2.5-coder:7b).
func TestOllamaE2E_WriteTool(t *testing.T) {
	if os.Getenv("YACA_OLLAMA_E2E") == "" {
		t.Skip("set YACA_OLLAMA_E2E=1 and run `ollama serve` to enable the live Ollama test")
	}
	model := os.Getenv("YACA_OLLAMA_MODEL")
	if model == "" {
		model = "qwen2.5-coder:7b"
	}

	dir := t.TempDir()
	oldWd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldWd) //nolint

	ag := agent.New(agent.Config{
		Provider:  ai.NewOllama(""),
		Model:     model,
		System:    "You are YACA, a coding agent. Act using tools.",
		Tools:     tools.All(),
		MaxTokens: 2048,
		MaxTurns:  8,
	})

	var mu sync.Mutex
	var toolNames []string
	done := make(chan struct{})

	go func() {
		for ev := range ag.Subscribe() {
			switch ev.Type {
			case agent.EventToolCallStart:
				mu.Lock()
				toolNames = append(toolNames, ev.ToolName)
				mu.Unlock()
				fmt.Printf("[tool call] %s %v\n", ev.ToolName, ev.ToolInput)
			case agent.EventToolResult:
				fmt.Printf("[tool result err=%v] %s\n", ev.IsError, truncate(ev.ToolResult, 100))
			case agent.EventError:
				fmt.Printf("[error] %v\n", ev.Err)
			case agent.EventAgentEnd:
				close(done)
				return
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	ag.Send(ctx, "Create a file named hello.txt whose contents are exactly: HELLO")

	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("timed out waiting for the agent to finish")
	}

	mu.Lock()
	calls := strings.Join(toolNames, ", ")
	mu.Unlock()
	if calls == "" {
		t.Fatal("model never emitted a tool call (XML tool-calling not working)")
	}

	data, err := os.ReadFile(filepath.Join(dir, "hello.txt"))
	if err != nil {
		t.Fatalf("hello.txt not created (tools used: %s): %v", calls, err)
	}
	if !strings.Contains(string(data), "HELLO") {
		t.Errorf("hello.txt = %q, want it to contain HELLO (tools used: %s)", data, calls)
	}
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}
