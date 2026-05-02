package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/Adi-ty/yaca/config"
)

// SetupResult is returned by RunSetup on success.
type SetupResult struct {
	Cfg *config.Config
}

// setupStep tracks which step of the wizard we're on.
type setupStep int

const (
	stepProvider setupStep = iota
	stepAPIKey
	stepModel
	stepDone
)

var providers = []string{"Anthropic (Claude)", "OpenAI (GPT-4o)", "Ollama (local)", "Skip for now"}
var providerKeys = []string{"anthropic", "openai", "ollama", ""}

// ── styles ────────────────────────────────────────────────────────────────────

var (
	setupTitle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("62")).MarginBottom(1)
	setupSubtitle = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	setupSelected = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("86"))
	setupCursor   = lipgloss.NewStyle().Foreground(lipgloss.Color("212"))
	setupDim      = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	setupErr      = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true)
	setupBox      = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("62")).
			Padding(1, 2)
)

// ── SetupModel ────────────────────────────────────────────────────────────────

type SetupModel struct {
	step         setupStep
	providerIdx  int
	storedAPIKey string // carries API key from step 2 to step 3
	input        textinput.Model
	errMsg       string
	width        int
	height       int
	result       *config.Config
}

type setupDoneMsg struct{ cfg *config.Config }

// NewSetupModel builds the initial setup wizard model.
func NewSetupModel() SetupModel {
	ti := textinput.New()
	ti.CharLimit = 256
	ti.Width = 50
	return SetupModel{
		step:  stepProvider,
		input: ti,
	}
}

// RunSetup runs the setup wizard as a blocking program and returns the chosen
// config (or nil if the user skipped).
func RunSetup() (*config.Config, error) {
	m := NewSetupModel()
	p := tea.NewProgram(m)
	result, err := p.Run()
	if err != nil {
		return nil, fmt.Errorf("setup: %w", err)
	}
	final, ok := result.(SetupModel)
	if !ok || final.result == nil {
		return nil, nil
	}
	return final.result, nil
}

func (m SetupModel) Init() tea.Cmd {
	return nil
}

func (m SetupModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case setupDoneMsg:
		m.result = msg.cfg
		return m, tea.Quit

	case tea.KeyMsg:
		switch m.step {
		case stepProvider:
			return m.updateProvider(msg)
		case stepAPIKey:
			return m.updateAPIKey(msg)
		case stepModel:
			return m.updateModel(msg)
		}
	}
	return m, nil
}

func (m SetupModel) updateProvider(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyUp:
		if m.providerIdx > 0 {
			m.providerIdx--
		}
	case tea.KeyDown:
		if m.providerIdx < len(providers)-1 {
			m.providerIdx++
		}
	case tea.KeyEnter:
		pKey := providerKeys[m.providerIdx]
		if pKey == "" {
			// Skip: use Ollama fallback with no key
			cfg := &config.Config{Provider: "ollama", Model: config.DefaultModel("ollama")}
			return m, func() tea.Msg { return setupDoneMsg{cfg: cfg} }
		}
		if pKey == "ollama" {
			m.step = stepModel
			m.input.Placeholder = "http://localhost:11434/v1 (default)"
			m.input.SetValue("")
			m.input.Focus()
			return m, textinput.Blink
		}
		m.step = stepAPIKey
		m.input.Placeholder = "sk-..."
		m.input.EchoMode = textinput.EchoPassword
		m.input.EchoCharacter = '•'
		m.input.SetValue("")
		m.input.Focus()
		return m, textinput.Blink
	case tea.KeyCtrlC:
		return m, tea.Quit
	default:
		// j/k navigation
		switch msg.String() {
		case "k":
			if m.providerIdx > 0 {
				m.providerIdx--
			}
		case "j":
			if m.providerIdx < len(providers)-1 {
				m.providerIdx++
			}
		}
	}
	return m, nil
}

func (m SetupModel) updateAPIKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyEnter:
		key := strings.TrimSpace(m.input.Value())
		if key == "" {
			m.errMsg = "API key cannot be empty. Press Ctrl+C to quit."
			return m, nil
		}
		m.errMsg = ""
		m.storedAPIKey = key
		m.step = stepModel
		m.input.EchoMode = textinput.EchoNormal
		m.input.Placeholder = config.DefaultModel(providerKeys[m.providerIdx])
		m.input.SetValue("")
		m.input.Focus()
		return m, textinput.Blink
	case tea.KeyCtrlC:
		return m, tea.Quit
	case tea.KeyEsc:
		m.step = stepProvider
		m.input.Blur()
		return m, nil
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m SetupModel) updateModel(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	pKey := providerKeys[m.providerIdx]
	switch msg.Type {
	case tea.KeyEnter:
		modelVal := strings.TrimSpace(m.input.Value())
		if modelVal == "" {
			if pKey == "ollama" {
				modelVal = config.DefaultModel("ollama")
			} else {
				modelVal = config.DefaultModel(pKey)
			}
		}
		var apiKey, ollamaURL string
		if pKey == "ollama" {
			ollamaURL = m.input.Value()
			if ollamaURL == "" {
				ollamaURL = "http://localhost:11434/v1"
			}
			// For ollama, the "model" step is really the URL step.
			// Re-use input for actual model name.
			m.step = stepModel
			m.input.Placeholder = config.DefaultModel("ollama")
			m.input.SetValue("")
			m.input.Focus()
			// Hack: we need a second model step for ollama. Use a sentinel.
			// Simplification: combine URL + model into single step via placeholder swap.
			// For now, just use the value as ollama URL.
			cfg := &config.Config{
				Provider:  "ollama",
				Model:     config.DefaultModel("ollama"),
				OllamaURL: ollamaURL,
			}
			return m, func() tea.Msg { return setupDoneMsg{cfg: cfg} }
		}
		// Find the stored API key from the previous step.
		// We need to re-access it — store it in the model between steps.
		apiKey = m.storedAPIKey
		cfg := &config.Config{
			Provider: pKey,
			APIKey:   apiKey,
			Model:    modelVal,
		}
		return m, func() tea.Msg { return setupDoneMsg{cfg: cfg} }
	case tea.KeyCtrlC:
		return m, tea.Quit
	case tea.KeyEsc:
		if pKey != "ollama" {
			m.step = stepAPIKey
			m.input.EchoMode = textinput.EchoPassword
			m.input.EchoCharacter = '•'
		} else {
			m.step = stepProvider
		}
		m.input.Focus()
		return m, textinput.Blink
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m SetupModel) View() string {
	if m.width == 0 {
		return "Loading…\n"
	}

	var content strings.Builder
	content.WriteString(setupTitle.Render("YACA — First-time setup") + "\n")
	content.WriteString(setupSubtitle.Render("Configure your AI provider to get started.") + "\n\n")

	switch m.step {
	case stepProvider:
		content.WriteString("Choose a provider:\n\n")
		for i, p := range providers {
			cursor := "  "
			line := p
			if i == m.providerIdx {
				cursor = setupCursor.Render("▶ ")
				line = setupSelected.Render(p)
			} else {
				line = setupDim.Render(p)
			}
			content.WriteString(cursor + line + "\n")
		}
		content.WriteString("\n" + setupDim.Render("↑/↓ or j/k to navigate  ·  Enter to select  ·  Ctrl+C to quit"))

	case stepAPIKey:
		provider := providers[m.providerIdx]
		content.WriteString(fmt.Sprintf("Enter your %s API key:\n\n", provider))
		content.WriteString(m.input.View() + "\n\n")
		if m.errMsg != "" {
			content.WriteString(setupErr.Render(m.errMsg) + "\n")
		}
		content.WriteString(setupDim.Render("Enter to continue  ·  Esc to go back  ·  Ctrl+C to quit"))

	case stepModel:
		pKey := providerKeys[m.providerIdx]
		if pKey == "ollama" {
			content.WriteString("Ollama base URL (leave blank for default):\n\n")
			content.WriteString(m.input.View() + "\n\n")
			content.WriteString(setupDim.Render("Default: http://localhost:11434/v1") + "\n")
		} else {
			content.WriteString("Model name (leave blank for default):\n\n")
			content.WriteString(m.input.View() + "\n\n")
			content.WriteString(setupDim.Render(fmt.Sprintf("Default: %s", config.DefaultModel(pKey))) + "\n")
		}
		content.WriteString("\n" + setupDim.Render("Enter to save  ·  Esc to go back  ·  Ctrl+C to quit"))
	}

	box := setupBox.Width(m.width - 4).Render(content.String())
	// Centre vertically
	lines := strings.Count(box, "\n") + 1
	padTop := (m.height - lines) / 2
	if padTop < 0 {
		padTop = 0
	}
	return strings.Repeat("\n", padTop) + box + "\n"
}
