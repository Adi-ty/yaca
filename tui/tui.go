package tui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/Adi-ty/yaca/agent"
)

// Fixed line counts for layout arithmetic.
const (
	headerLines = 2 // title line + separator
	inputLines  = 3 // textarea height
	footerLines = 2 // separator + status line
)

// ── styles ────────────────────────────────────────────────────────────────────

var (
	sHeader     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("62"))
	sDim        = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	sUser       = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("86"))
	sAssistant  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("212"))
	sThinking   = lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Italic(true)
	sToolCall   = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	sToolResult = lipgloss.NewStyle().Foreground(lipgloss.Color("82"))
	sError      = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true)
	sStatus     = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	sSep        = lipgloss.NewStyle().Foreground(lipgloss.Color("238"))
)

// ── message types ─────────────────────────────────────────────────────────────

// agentEventMsg carries an agent.Event into Bubbletea's Update loop.
type agentEventMsg struct{ ev agent.Event }

// ── Model ─────────────────────────────────────────────────────────────────────

// Model is the root Bubbletea model for YACA's terminal UI.
type Model struct {
	ag           *agent.Agent
	providerName string
	modelName    string

	vp    viewport.Model
	ta    textarea.Model
	ready bool
	width int
	height int

	// content holds all completed, rendered conversation turns.
	content string

	// In-flight turn state (reset by flushTurn on EventTurnEnd).
	streaming bool
	thinkBuf  string // accumulated thinking deltas
	streamBuf string // accumulated text deltas
	toolLines string // ⚙ indicator lines for tool calls in this turn

	// Status bar.
	statusMsg    string
	inputTokens  int
	outputTokens int
}

// New builds the initial Model. Call SetProgram after tea.NewProgram.
func New(ag *agent.Agent, providerName, modelName string) Model {
	ta := textarea.New()
	ta.Placeholder = "Message… (Enter to send, Ctrl+C to quit)"
	ta.Focus()
	ta.ShowLineNumbers = false
	ta.SetHeight(inputLines)
	// Disable Enter→newline so we can intercept Enter to send.
	ta.KeyMap.InsertNewline.SetEnabled(false)

	return Model{
		ag:           ag,
		providerName: providerName,
		modelName:    modelName,
		ta:           ta,
		statusMsg:    "Ready",
	}
}

// SetProgram stores the tea.Program reference and starts the goroutine that
// relays agent events into Bubbletea's message loop via p.Send.
func (m *Model) SetProgram(p *tea.Program) {
	ch := m.ag.Subscribe()
	go func() {
		for ev := range ch {
			p.Send(agentEventMsg{ev})
		}
	}()
}

// ── Bubbletea interface ───────────────────────────────────────────────────────

func (m Model) Init() tea.Cmd {
	return textarea.Blink
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		vpH := m.vpHeight()
		if !m.ready {
			m.vp = viewport.New(m.width, vpH)
			m.ready = true
		} else {
			m.vp.Width = m.width
			m.vp.Height = vpH
		}
		m.ta.SetWidth(m.width)
		m.vp.SetContent(m.buildViewContent())

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit

		case tea.KeyEnter:
			// Send on Enter (only when not streaming and input is non-empty).
			if !m.streaming {
				if text := strings.TrimSpace(m.ta.Value()); text != "" {
					m = m.doSend(text)
					m.vp.SetContent(m.buildViewContent())
					m.vp.GotoBottom()
				}
			}

		default:
			var taCmd tea.Cmd
			m.ta, taCmd = m.ta.Update(msg)
			cmds = append(cmds, taCmd)
		}

	case agentEventMsg:
		m = m.onAgentEvent(msg.ev)
		m.vp.SetContent(m.buildViewContent())
		m.vp.GotoBottom()
	}

	// Always forward to viewport so PgUp/PgDn/mouse scrolling works.
	var vpCmd tea.Cmd
	m.vp, vpCmd = m.vp.Update(msg)
	cmds = append(cmds, vpCmd)

	return m, tea.Batch(cmds...)
}

func (m Model) View() string {
	if !m.ready {
		return "\nInitializing…\n"
	}
	sep := sSep.Render(strings.Repeat("─", m.width))
	return strings.Join([]string{
		m.renderHeader(),
		sep,
		m.vp.View(),
		sep,
		m.ta.View(),
		sStatus.Render(m.statusMsg),
	}, "\n")
}

// ── agent event handling ──────────────────────────────────────────────────────

func (m Model) onAgentEvent(ev agent.Event) Model {
	switch ev.Type {

	case agent.EventTextDelta:
		m.streamBuf += ev.Delta

	case agent.EventThinkingDelta:
		m.thinkBuf += ev.Delta

	case agent.EventToolCallStart:
		m.toolLines += sToolCall.Render("⚙  "+ev.ToolName+"(…)") + "\n"

	case agent.EventToolCallEnd:
		// Append the execution result under the completed history.
		preview := ev.ToolResult
		if runes := []rune(preview); len(runes) > 200 {
			preview = string(runes[:200]) + "…"
		}
		label := "✓  " + ev.ToolName + ": " + preview
		if ev.IsError {
			label = "✗  " + ev.ToolName + ": " + preview
			m.content += sError.Render(label) + "\n"
		} else {
			m.content += sToolResult.Render(label) + "\n"
		}

	case agent.EventTurnEnd:
		m = m.flushTurn()

	case agent.EventAgentEnd:
		m.streaming = false
		if ev.Usage.InputTokens > 0 || ev.Usage.OutputTokens > 0 {
			m.inputTokens = ev.Usage.InputTokens
			m.outputTokens = ev.Usage.OutputTokens
		}
		m.statusMsg = m.doneStatus()

	case agent.EventError:
		m.streaming = false
		errMsg := "unknown error"
		if ev.Err != nil {
			errMsg = ev.Err.Error()
		}
		m.content += sError.Render("Error: "+errMsg) + "\n\n"
		m.statusMsg = "Error — see above"
	}
	return m
}

// flushTurn moves the in-flight turn buffers into the completed content string.
func (m Model) flushTurn() Model {
	hasContent := m.thinkBuf != "" || m.streamBuf != "" || m.toolLines != ""
	if hasContent {
		var sb strings.Builder
		sb.WriteString(sAssistant.Render("Assistant") + "\n")
		if m.thinkBuf != "" {
			sb.WriteString(sThinking.Render(m.thinkBuf) + "\n")
		}
		if m.streamBuf != "" {
			sb.WriteString(m.streamBuf + "\n")
		}
		if m.toolLines != "" {
			sb.WriteString(m.toolLines)
		}
		m.content += sb.String() + "\n"
	}
	m.thinkBuf = ""
	m.streamBuf = ""
	m.toolLines = ""
	return m
}

// doSend records the user message, marks streaming active, and fires the agent.
func (m Model) doSend(text string) Model {
	m.ta.Reset()
	m.content += sUser.Render("You") + "\n" + text + "\n\n"
	m.streaming = true
	m.statusMsg = "Streaming…"
	// agent.Send is non-blocking (starts its own goroutine internally).
	m.ag.Send(context.Background(), text)
	return m
}

// ── rendering helpers ─────────────────────────────────────────────────────────

// buildViewContent returns the full viewport string: completed history plus the
// in-flight streaming turn (if any).
func (m Model) buildViewContent() string {
	if !m.streaming || (m.thinkBuf == "" && m.streamBuf == "" && m.toolLines == "") {
		return m.content
	}
	var sb strings.Builder
	sb.WriteString(m.content)
	sb.WriteString(sAssistant.Render("Assistant") + "\n")
	if m.thinkBuf != "" {
		sb.WriteString(sThinking.Render(m.thinkBuf) + "\n")
	}
	if m.streamBuf != "" {
		sb.WriteString(m.streamBuf)
		if !strings.HasSuffix(m.streamBuf, "\n") && m.toolLines != "" {
			sb.WriteString("\n")
		}
	}
	if m.toolLines != "" {
		sb.WriteString(m.toolLines)
	}
	return sb.String()
}

func (m Model) renderHeader() string {
	return sHeader.Render("YACA") + sDim.Render("  "+m.providerName+" / "+m.modelName)
}

func (m Model) doneStatus() string {
	if m.inputTokens == 0 && m.outputTokens == 0 {
		return "Done"
	}
	cost := m.computeCost()
	if cost >= 0 {
		return fmt.Sprintf("Done  ·  %d in / %d out tokens  ·  $%.4f",
			m.inputTokens, m.outputTokens, cost)
	}
	return fmt.Sprintf("Done  ·  %d in / %d out tokens", m.inputTokens, m.outputTokens)
}

func (m Model) computeCost() float64 {
	rates, ok := tokenCostRates[m.modelName]
	if !ok {
		return -1
	}
	return float64(m.inputTokens)*rates[0]/1e6 + float64(m.outputTokens)*rates[1]/1e6
}

func (m Model) vpHeight() int {
	h := m.height - headerLines - inputLines - footerLines
	if h < 1 {
		return 1
	}
	return h
}

// tokenCostRates maps model ID → [inputPer1M, outputPer1M] in USD.
var tokenCostRates = map[string][2]float64{
	"claude-opus-4-5":            {15.0, 75.0},
	"claude-sonnet-4-5":          {3.0, 15.0},
	"claude-haiku-4-5-20251001":  {0.8, 4.0},
	"claude-3-5-sonnet-20241022": {3.0, 15.0},
	"claude-3-5-haiku-20241022":  {0.8, 4.0},
	"claude-3-opus-20240229":     {15.0, 75.0},
	"gpt-4o":                     {2.5, 10.0},
	"gpt-4o-mini":                {0.15, 0.60},
}
