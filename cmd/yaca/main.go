package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/joho/godotenv"

	"github.com/Adi-ty/yaca/agent"
	"github.com/Adi-ty/yaca/ai"
	"github.com/Adi-ty/yaca/config"
	"github.com/Adi-ty/yaca/session"
	"github.com/Adi-ty/yaca/tools"
	"github.com/Adi-ty/yaca/tui"
)

// baseSystemPrompt holds YACA's behavioural rules. The catalogue of tools and
// the exact <tool_call> format are appended automatically by the provider
// (see ai.BuildToolSystemPrompt), so they are deliberately not duplicated here.
const baseSystemPrompt = `You are YACA, an autonomous coding agent operating in a terminal. You complete tasks by calling tools — never by describing what the user should do.

Operating rules:
- To act — create or edit files, run commands, search, browse, remember — CALL A TOOL. The available tools and the exact call format are described below.
- Prefer doing over explaining: do not ask for confirmation on a clear instruction, and do not narrate a plan you could simply execute.
- CREATE a file → write (the "content" argument holds the complete file text).
- EDIT a file → read it first, then edit with an exact old_string/new_string pair. If an edit fails because old_string is missing or not unique, re-read the file and copy the text verbatim with more surrounding context.
- RUN a command → bash. Never tell the user to run it themselves.
- SEARCH → glob or grep. Never guess file paths.
- Never paste file contents in a markdown code block instead of calling write.

Work in steps: chain tool calls until the task is complete. When finished, reply with a short plain-text summary of what changed (files, commands, results).`

func main() {
	// .env is optional.
	_ = godotenv.Load()

	// ── CLI flags ──────────────────────────────────────────────────────────────
	var (
		flagDir      string
		flagModel    string
		flagProvider string
		flagBaseURL  string
		flagContinue bool
		flagSetup    bool
		flagVersion  bool
	)
	flag.StringVar(&flagDir, "dir", "", "project `directory` to work in (default: current directory)")
	flag.StringVar(&flagDir, "d", "", "project `directory` (shorthand)")
	flag.StringVar(&flagModel, "model", "", "override `model` (e.g. qwen2.5-coder:7b)")
	flag.StringVar(&flagModel, "m", "", "override model (shorthand)")
	flag.StringVar(&flagProvider, "provider", "", "override `provider` (anthropic, openai, ollama)")
	flag.StringVar(&flagProvider, "p", "", "override provider (shorthand)")
	flag.StringVar(&flagBaseURL, "base-url", "", "custom OpenAI-compatible base `url` (e.g. http://localhost:1234/v1 for LM Studio/vLLM)")
	flag.BoolVar(&flagContinue, "continue", false, "continue last session for this project")
	flag.BoolVar(&flagContinue, "c", false, "continue last session (shorthand)")
	flag.BoolVar(&flagSetup, "setup", false, "reconfigure provider/API key")
	flag.BoolVar(&flagVersion, "version", false, "print version and exit")
	flag.BoolVar(&flagVersion, "v", false, "print version (shorthand)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: yaca [options] [directory]\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  yaca                               # use current directory\n")
		fmt.Fprintf(os.Stderr, "  yaca ./myproject                   # open myproject\n")
		fmt.Fprintf(os.Stderr, "  yaca -c ./myproject                # continue last session\n")
		fmt.Fprintf(os.Stderr, "  yaca -p ollama -m qwen2.5-coder:7b # use a specific ollama model\n")
		fmt.Fprintf(os.Stderr, "  yaca --setup                       # reconfigure provider\n")
	}
	flag.Parse()

	if flagVersion {
		fmt.Printf("yaca %s\n", config.Version)
		return
	}

	// Positional arg sets the project directory.
	if flag.NArg() > 0 && flagDir == "" {
		flagDir = flag.Arg(0)
	}

	// Change into the project directory early so all relative paths work.
	if flagDir != "" {
		abs, err := filepath.Abs(flagDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "yaca: invalid path %q: %v\n", flagDir, err)
			os.Exit(1)
		}
		if _, err := os.Stat(abs); err != nil {
			fmt.Fprintf(os.Stderr, "yaca: directory not found: %s\n", abs)
			os.Exit(1)
		}
		if err := os.Chdir(abs); err != nil {
			fmt.Fprintf(os.Stderr, "yaca: chdir %s: %v\n", abs, err)
			os.Exit(1)
		}
	}

	// ── Provider / API key resolution ──────────────────────────────────────────
	cfg := resolveConfig(flagSetup, flagProvider, flagBaseURL)
	if flagProvider != "" {
		cfg.Provider = flagProvider
		// Ollama needs no API key; clear any stale key from a previous provider.
		if flagProvider == "ollama" {
			cfg.APIKey = ""
		}
	}
	if flagBaseURL != "" {
		cfg.BaseURL = flagBaseURL
	}
	if flagModel != "" {
		cfg.Model = flagModel
	}
	if cfg.Model == "" {
		cfg.Model = config.DefaultModel(cfg.Provider)
	}

	providerName, modelName, provider := buildProvider(cfg)

	// ── Agent setup ────────────────────────────────────────────────────────────
	systemPrompt := buildSystemPrompt()

	ag := agent.New(agent.Config{
		Provider:  provider,
		Model:     modelName,
		System:    systemPrompt,
		Tools:     tools.All(),
		MaxTokens: 8192,
		MaxTurns:  25,
		// Auto-summarise history once it grows past ~40k chars (~10k tokens) so
		// long sessions stay within local models' context windows.
		CompactThreshold: 40000,
	})

	// ── Session: load project-scoped history ───────────────────────────────────
	cwd, _ := os.Getwd()
	var sess *session.Session
	if flagContinue {
		sess, _ = session.Latest(cwd)
	}
	if sess == nil {
		sess = session.New(modelName, cwd)
	} else {
		ag.LoadMessages(sess.Messages)
	}

	// Auto-save after each completed agent turn.
	go func() {
		for ev := range ag.Subscribe() {
			if ev.Type == agent.EventAgentEnd {
				sess.Messages = ag.State().Messages
				_ = sess.Save()
			}
		}
	}()

	// ── TUI ────────────────────────────────────────────────────────────────────
	onNewSession := func() string {
		sess = session.New(modelName, cwd)
		ag.Reset()
		return sess.ID
	}

	m := tui.New(ag, providerName, modelName, cwd)
	m.SessionID = sess.ID
	m.OnNewSession = onNewSession

	p := tea.NewProgram(m, tea.WithAltScreen())
	m.SetProgram(p)

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "yaca: %v\n", err)
		os.Exit(1)
	}
}

// resolveConfig loads credentials with this priority:
//  1. Explicit keyless endpoints chosen via flags (ollama, or openai --base-url)
//  2. Environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY)
//  3. ~/.yaca/config.json
//  4. In-app setup wizard (if --setup or nothing configured)
func resolveConfig(forceSetup bool, flagProvider, flagBaseURL string) *config.Config {
	// Explicit local / custom OpenAI-compatible targets need no wizard or key.
	if !forceSetup {
		switch flagProvider {
		case "ollama":
			return &config.Config{Provider: "ollama"}
		case "openai":
			if flagBaseURL != "" {
				return &config.Config{Provider: "openai", BaseURL: flagBaseURL, APIKey: os.Getenv("OPENAI_API_KEY")}
			}
		}
	}

	// Env vars take highest priority.
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" && !forceSetup {
		return &config.Config{Provider: "anthropic", APIKey: key}
	}
	if key := os.Getenv("OPENAI_API_KEY"); key != "" && !forceSetup {
		return &config.Config{Provider: "openai", APIKey: key}
	}

	// Try config file.
	if !forceSetup {
		if cfg, err := config.Load(); err == nil && cfg != nil {
			if cfg.APIKey != "" || cfg.Provider == "ollama" {
				return cfg
			}
		}
	}

	// Run setup wizard.
	cfg, err := tui.RunSetup()
	if err != nil {
		fmt.Fprintf(os.Stderr, "yaca: setup failed: %v\n", err)
		os.Exit(1)
	}
	if cfg == nil {
		// User quit during setup — fall back to Ollama.
		cfg = &config.Config{Provider: "ollama", Model: config.DefaultModel("ollama")}
	}
	// Persist non-skip config.
	if cfg.Provider != "" {
		if saveErr := config.Save(cfg); saveErr != nil {
			fmt.Fprintf(os.Stderr, "yaca: warning: could not save config: %v\n", saveErr)
		}
	}
	return cfg
}

// buildProvider constructs the ai.Provider from the resolved config.
func buildProvider(cfg *config.Config) (name, model string, p ai.Provider) {
	switch cfg.Provider {
	case "anthropic":
		return "anthropic", cfg.Model, ai.NewAnthropicProvider(cfg.APIKey)
	case "openai":
		baseURL := cfg.BaseURL
		if baseURL == "" {
			baseURL = envOr("OPENAI_BASE_URL", "")
		}
		return "openai", cfg.Model, ai.NewOpenAI(cfg.APIKey, baseURL)
	default:
		url := cfg.OllamaURL
		if url == "" {
			url = envOr("OLLAMA_URL", "")
		}
		return "ollama", cfg.Model, ai.NewOllama(url)
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

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
