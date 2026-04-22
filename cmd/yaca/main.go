package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/joho/godotenv"

	"github.com/Adi-ty/yaca/agent"
	"github.com/Adi-ty/yaca/ai"
	"github.com/Adi-ty/yaca/session"
	"github.com/Adi-ty/yaca/tools"
	"github.com/Adi-ty/yaca/tui"
)

const baseSystemPrompt = `You are YACA, a coding assistant. Use tools for all file and shell work — never guess.

Available tools (use exact names): read, write, edit, bash, glob, grep, list_dir, fetch, memory_read, memory_write

Rules you must follow:
1. Always call read on a file before calling edit on it.
2. For write: the "content" field must contain the complete file contents.
3. For edit: "old_string" must be an exact substring of the file; "new_string" replaces it.
4. For bash: run the command and show the output to the user.
5. Never invent file contents. If unsure, read first.
6. Use memory_write to save important project context, decisions, or notes that should persist across sessions.
7. Use fetch to retrieve documentation, API specs, or remote files when needed.`

func main() {
	// .env is optional; ignore "not found" errors.
	_ = godotenv.Load()

	providerName, modelName, provider := detectProvider()
	ai.Register(provider)

	// Build enriched system prompt with working directory and git context.
	systemPrompt := buildSystemPrompt()

	ag := agent.New(agent.Config{
		Provider:  provider,
		Model:     modelName,
		System:    systemPrompt,
		Tools:     tools.All(),
		MaxTokens: 8192,
	})

	// Load the most recent session (if any).
	sess, _ := session.Latest()
	if sess != nil {
		ag.LoadMessages(sess.Messages)
	} else {
		sess = session.New(modelName)
	}

	// Auto-save session after each completed agent turn.
	go func() {
		for ev := range ag.Subscribe() {
			if ev.Type == agent.EventAgentEnd {
				sess.Messages = ag.State().Messages
				_ = sess.Save()
			}
		}
	}()

	// Callback used by the TUI's /new command.
	onNewSession := func() string {
		sess = session.New(modelName)
		ag.Reset()
		return sess.ID
	}

	m := tui.New(ag, providerName, modelName)
	m.SessionID = sess.ID
	m.OnNewSession = onNewSession

	p := tea.NewProgram(m, tea.WithAltScreen())
	m.SetProgram(p)

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "yaca: %v\n", err)
		os.Exit(1)
	}
}

// buildSystemPrompt returns the base prompt enriched with cwd and git context.
func buildSystemPrompt() string {
	var sb strings.Builder
	sb.WriteString(baseSystemPrompt)

	cwd, err := os.Getwd()
	if err == nil {
		sb.WriteString("\n\nWorking directory: " + cwd)
	}

	if out, err := exec.Command("git", "status", "--short").Output(); err == nil && len(out) > 0 {
		sb.WriteString("\nGit status:\n" + string(out))
	}

	return sb.String()
}

// detectProvider checks env vars in priority order and returns the provider
// name, default model, and a ready-to-use ai.Provider.
func detectProvider() (name, model string, p ai.Provider) {
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		model = envOr("YACA_MODEL", "claude-3-5-sonnet-20241022")
		return "anthropic", model, ai.NewAnthropicProvider(key)
	}
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		model = envOr("YACA_MODEL", "gpt-4o")
		return "openai", model, ai.NewOpenAI(key)
	}
	// Fallback: Ollama on localhost.
	model = envOr("YACA_MODEL", "llama3.2")
	return "ollama", model, ai.NewOllama("")
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
