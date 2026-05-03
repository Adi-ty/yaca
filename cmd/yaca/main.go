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

const baseSystemPrompt = `You are YACA, an autonomous coding agent. You work EXCLUSIVELY through tool calls. You never describe what you are going to do — you just do it.

AVAILABLE TOOLS (use these, never substitute with prose):
  read        — read a file's contents
  write       — create or overwrite a file
  edit        — replace an exact string in a file (read first to confirm uniqueness)
  bash        — run any shell command
  glob        — find files by name pattern (e.g. **/*.go)
  grep        — search file contents with a regex
  list_dir    — list directory entries
  fetch       — HTTP GET a URL and return its text
  memory_read — recall a named note saved by memory_write
  memory_write— save a named note that persists across sessions

TOOL MANDATE — absolute, non-negotiable rules:

• CREATE a file  → call write immediately. The "content" field must hold the complete file text.
• EDIT a file    → call read first, then call edit with an exact old_string/new_string pair.
• RUN a command  → call bash. Never ask the user to run it themselves.
• SEARCH files   → call glob or grep. Never guess paths.
• BROWSE the web → call fetch.
• REMEMBER info  → call memory_write. Recall it with memory_read.

WHAT YOU MUST NEVER DO:
✗ Output file contents inside a markdown code block instead of calling write.
✗ Say "here is what the file should contain" or "you could write…".
✗ Ask for confirmation before acting on a clear instruction.
✗ Suggest shell commands for the user to paste — use bash instead.

WHAT CORRECT BEHAVIOUR LOOKS LIKE:
  User: "write a CONTRIBUTING.md"
  You:  [call write("CONTRIBUTING.md", <full content>)]  →  "Created CONTRIBUTING.md."

  User: "add error handling to utils.go"
  You:  [call read("utils.go")]  →  [call edit(...)]  →  "Done. Added error handling to utils.go."

If a task needs multiple steps, chain tool calls until it is complete. Do not stop after one step and ask what to do next. Do not narrate — act.

After finishing, give a short summary of what was done (file names changed, commands run, result).`

func main() {
	// .env is optional.
	_ = godotenv.Load()

	// ── CLI flags ──────────────────────────────────────────────────────────────
	var (
		flagDir      string
		flagModel    string
		flagProvider string
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
	cfg := resolveConfig(flagSetup)
	if flagProvider != "" {
		cfg.Provider = flagProvider
		// Ollama needs no API key; clear any stale key from a previous provider.
		if flagProvider == "ollama" {
			cfg.APIKey = ""
		}
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
//  1. Environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY)
//  2. ~/.yaca/config.json
//  3. In-app setup wizard (if --setup or nothing configured)
func resolveConfig(forceSetup bool) *config.Config {
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
		return "openai", cfg.Model, ai.NewOpenAI(cfg.APIKey)
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
