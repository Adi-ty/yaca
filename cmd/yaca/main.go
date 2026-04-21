package main

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/Adi-ty/yaca/agent"
	"github.com/Adi-ty/yaca/ai"
	"github.com/Adi-ty/yaca/tools"
	"github.com/Adi-ty/yaca/tui"
)

const systemPrompt = `You are YACA (Yet Another Coding Assistant), an expert software engineer.
You have access to tools: read, write, edit, bash, glob, grep, list_dir.
Use the exact tool names. Always use tools for file and shell work instead of guessing.
Always read files before editing. Prefer edit over write for existing files.
When running bash, show the user what you ran. Be concise but thorough.`

func main() {
	// .env is optional; ignore "not found" errors.
	_ = godotenv.Load()

	providerName, modelName, provider := detectProvider()
	ai.Register(provider)

	ag := agent.New(agent.Config{
		Provider:  provider,
		Model:     modelName,
		System:    systemPrompt,
		Tools:     tools.All(),
		MaxTokens: 8192,
	})

	m := tui.New(ag, providerName, modelName)
	p := tea.NewProgram(m, tea.WithAltScreen())
	m.SetProgram(p)

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "yaca: %v\n", err)
		os.Exit(1)
	}
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
