package tui

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"github.com/Adi-ty/yaca/agent"
)

// Fixed line counts for layout arithmetic.
const (
	headerLines = 2 // title + separator
	inputLines  = 3 // textarea height
	footerLines = 2 // separator + status
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
	sPath       = lipgloss.NewStyle().Foreground(lipgloss.Color("69"))
)

// ── message types ─────────────────────────────────────────────────────────────

type agentEventMsg struct{ ev agent.Event }
type compactDoneMsg struct{ err error }

// ── Model ─────────────────────────────────────────────────────────────────────

type Model struct {
	ag           *agent.Agent
	providerName string
	modelName    string
	projectDir   string // absolute path of current project

	vp     viewport.Model
	ta     textarea.Model
	ready  bool
	width  int
	height int

	// Glamour renderer — recreated on resize.
	renderer *glamour.TermRenderer

	// content holds all completed, rendered conversation turns.
	content string

	// In-flight turn state (reset by flushTurn on EventTurnEnd).
	streaming bool
	thinkBuf  string
	streamBuf string
	toolLines string

	// Status bar.
	statusMsg    string
	inputTokens  int
	outputTokens int

	// Session management.
	SessionID    string
	OnNewSession func() string
}

// New builds the initial Model.
func New(ag *agent.Agent, providerName, modelName, projectDir string) Model {
	ta := textarea.New()
	ta.Placeholder = "Message… (Enter to send, Shift+Enter for newline, Ctrl+C to quit)"
	ta.Focus()
	ta.ShowLineNumbers = false
	ta.SetHeight(inputLines)
	// Keep Shift+Enter for newlines; intercept plain Enter to send.
	ta.KeyMap.InsertNewline.SetEnabled(false)

	abs, _ := filepath.Abs(projectDir)
	return Model{
		ag:           ag,
		providerName: providerName,
		modelName:    modelName,
		projectDir:   abs,
		ta:           ta,
		statusMsg:    "Ready",
	}
}

// SetProgram bridges agent events into Bubbletea's message loop.
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
		// Recreate glamour renderer with correct width.
		m.renderer = newRenderer(m.width)
		m.vp.SetContent(m.buildViewContent())

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit

		case tea.KeyEnter:
			if !m.streaming {
				if text := strings.TrimSpace(m.ta.Value()); text != "" {
					m.ta.Reset()
					if strings.HasPrefix(text, "/") {
						var cmd tea.Cmd
						m, cmd = m.handleSlashCommand(text)
						cmds = append(cmds, cmd)
					} else {
						m = m.doSend(text)
					}
					m.vp.SetContent(m.buildViewContent())
					m.vp.GotoBottom()
				}
			}

		default:
			// Shift+Enter inserts a newline.
			if msg.Type == tea.KeyShiftDown || msg.Alt && msg.Type == tea.KeyEnter {
				m.ta.InsertString("\n")
			} else {
				var taCmd tea.Cmd
				m.ta, taCmd = m.ta.Update(msg)
				cmds = append(cmds, taCmd)
			}
		}

	case compactDoneMsg:
		if msg.err != nil {
			m.content += sError.Render("Compact failed: "+msg.err.Error()) + "\n\n"
			m.statusMsg = "Error"
		} else {
			state := m.ag.State()
			m.content += sDim.Render(fmt.Sprintf("Context compacted — %d messages retained.", len(state.Messages))) + "\n\n"
			m.statusMsg = "Ready"
		}
		m.vp.SetContent(m.buildViewContent())
		m.vp.GotoBottom()

	case agentEventMsg:
		m = m.onAgentEvent(msg.ev)
		m.vp.SetContent(m.buildViewContent())
		m.vp.GotoBottom()
	}

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
		label := toolCallLabel(ev.ToolName, ev.ToolInput)
		m.toolLines += sToolCall.Render("⤷  "+label) + "\n"

	case agent.EventToolCallEnd:
		// Append to toolLines so ordering is correct: start (⤷) always
		// precedes result (✓/✗) and the whole block lands inside the
		// "Assistant" section that flushTurn creates on EventTurnEnd.
		label := toolResultLabel(ev.ToolName, ev.ToolInput, ev.ToolResult)
		if ev.IsError {
			m.toolLines += sError.Render("✗  "+label) + "\n"
		} else {
			m.toolLines += sToolResult.Render("✓  "+label) + "\n"
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

// flushTurn moves in-flight buffers into the immutable content string,
// running the text through glamour for markdown rendering.
func (m Model) flushTurn() Model {
	hasContent := m.thinkBuf != "" || m.streamBuf != "" || m.toolLines != ""
	if hasContent {
		var sb strings.Builder
		sb.WriteString(sAssistant.Render("Assistant") + "\n")
		if m.thinkBuf != "" {
			sb.WriteString(sThinking.Render("💭 "+m.thinkBuf) + "\n")
		}
		if m.streamBuf != "" {
			rendered := m.renderMarkdown(m.streamBuf)
			sb.WriteString(rendered)
			if !strings.HasSuffix(rendered, "\n") {
				sb.WriteByte('\n')
			}
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

// renderMarkdown runs text through glamour if a renderer is available,
// falling back to plain text on any error.
func (m Model) renderMarkdown(text string) string {
	if m.renderer == nil {
		return text
	}
	rendered, err := m.renderer.Render(text)
	if err != nil {
		return text
	}
	return rendered
}

// doSend fires the agent and records the user message.
func (m Model) doSend(text string) Model {
	m.content += sUser.Render("You") + "\n" + text + "\n\n"
	m.streaming = true
	m.statusMsg = "Streaming…"
	m.ag.Send(context.Background(), text)
	return m
}

// handleSlashCommand processes /commands.
func (m Model) handleSlashCommand(text string) (Model, tea.Cmd) {
	parts := strings.Fields(text)
	if len(parts) == 0 {
		return m, nil
	}
	switch parts[0] {
	case "/compact":
		m.statusMsg = "Compacting…"
		return m, func() tea.Msg {
			err := m.ag.Compact(context.Background())
			return compactDoneMsg{err: err}
		}

	case "/new":
		if m.OnNewSession != nil {
			m.SessionID = m.OnNewSession()
		}
		m.content = ""
		m.streaming = false
		m.statusMsg = "Ready"
		m.content += sDim.Render("New session: "+m.SessionID) + "\n\n"
		return m, nil

	case "/session":
		state := m.ag.State()
		id := m.SessionID
		if id == "" {
			id = "(unknown)"
		}
		m.content += sDim.Render(fmt.Sprintf(
			"Session: %s  ·  %d messages  ·  %d in / %d out tokens",
			id, len(state.Messages), m.inputTokens, m.outputTokens,
		)) + "\n\n"
		return m, nil

	case "/help":
		help := "/compact  — summarise older messages to free context\n" +
			"/new      — start a fresh session\n" +
			"/session  — show session ID and token stats\n" +
			"/help     — show this help"
		m.content += sDim.Render(help) + "\n\n"
		return m, nil

	default:
		m.content += sError.Render("Unknown command: "+parts[0]+" (try /help)") + "\n\n"
		return m, nil
	}
}

// ── rendering helpers ─────────────────────────────────────────────────────────

func (m Model) buildViewContent() string {
	if !m.streaming || (m.thinkBuf == "" && m.streamBuf == "" && m.toolLines == "") {
		return m.content
	}
	var sb strings.Builder
	sb.WriteString(m.content)
	sb.WriteString(sAssistant.Render("Assistant") + "\n")
	if m.thinkBuf != "" {
		sb.WriteString(sThinking.Render("💭 "+m.thinkBuf) + "\n")
	}
	if m.streamBuf != "" {
		// Show raw streaming text (not yet complete for glamour).
		sb.WriteString(m.streamBuf)
		if !strings.HasSuffix(m.streamBuf, "\n") && m.toolLines != "" {
			sb.WriteByte('\n')
		}
	}
	if m.toolLines != "" {
		sb.WriteString(m.toolLines)
	}
	return sb.String()
}

func (m Model) renderHeader() string {
	dir := m.projectDir
	if dir == "" {
		dir = "."
	}
	// Show only last 2 path components to save space.
	parts := strings.Split(filepath.ToSlash(dir), "/")
	if len(parts) > 2 {
		dir = "…/" + strings.Join(parts[len(parts)-2:], "/")
	}

	sess := ""
	if m.SessionID != "" {
		id := m.SessionID
		if len(id) > 8 {
			id = id[len(id)-8:]
		}
		sess = "  · sess:" + id
	}

	return sHeader.Render("YACA") +
		sDim.Render("  "+m.providerName+" / "+m.modelName+sess) +
		"  " + sPath.Render(dir)
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

// newRenderer creates a glamour renderer for the given terminal width.
func newRenderer(width int) *glamour.TermRenderer {
	w := width - 4
	if w < 40 {
		w = 40
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(w),
	)
	if err != nil {
		return nil
	}
	return r
}

// ── tool label helpers ────────────────────────────────────────────────────────

// toolCallLabel returns a compact one-line description of an in-progress call.
func toolCallLabel(name string, input map[string]any) string {
	switch name {
	case "read":
		return "read  " + sPath.Render(shortPath(strIn(input, "path")))
	case "write":
		return "write  " + sPath.Render(shortPath(strIn(input, "path")))
	case "edit":
		return "edit  " + sPath.Render(shortPath(strIn(input, "path")))
	case "bash":
		cmd := strIn(input, "command")
		if len(cmd) > 60 {
			cmd = cmd[:60] + "…"
		}
		return "bash  " + cmd
	case "glob":
		return "glob  " + strIn(input, "pattern")
	case "grep":
		pat := strIn(input, "pattern")
		if inc := strIn(input, "include"); inc != "" {
			pat += "  [" + inc + "]"
		}
		return "grep  " + pat
	case "list_dir":
		return "list_dir  " + sPath.Render(shortPath(strIn(input, "path")))
	case "fetch":
		u := strIn(input, "url")
		if len(u) > 60 {
			u = u[:60] + "…"
		}
		return "fetch  " + u
	case "memory_read":
		return "memory_read  " + strIn(input, "name")
	case "memory_write":
		return "memory_write  " + strIn(input, "name")
	}
	return name
}

// toolResultLabel returns a compact summary after a tool finishes.
func toolResultLabel(name string, input map[string]any, result string) string {
	switch name {
	case "read":
		lines := strings.Count(result, "\n") + 1
		return fmt.Sprintf("read  %s  (%d lines)", sPath.Render(shortPath(strIn(input, "path"))), lines)
	case "write":
		return "write  " + sPath.Render(shortPath(strIn(input, "path")))
	case "edit":
		return "edit  " + sPath.Render(shortPath(strIn(input, "path")))
	case "bash":
		cmd := strIn(input, "command")
		if len(cmd) > 40 {
			cmd = cmd[:40] + "…"
		}
		preview := firstLine(result)
		if preview != "" {
			return "bash  " + cmd + "\n     " + sDim.Render(preview)
		}
		return "bash  " + cmd
	case "glob":
		n := len(strings.Split(strings.TrimSpace(result), "\n"))
		if result == "no files found" {
			n = 0
		}
		return fmt.Sprintf("glob  %s  (%d files)", strIn(input, "pattern"), n)
	case "grep":
		n := len(strings.Split(strings.TrimSpace(result), "\n"))
		if result == "no matches found" {
			n = 0
		}
		return fmt.Sprintf("grep  %s  (%d matches)", strIn(input, "pattern"), n)
	case "fetch":
		u := strIn(input, "url")
		if len(u) > 50 {
			u = u[:50] + "…"
		}
		bytes := len(result)
		return fmt.Sprintf("fetch  %s  (%d chars)", u, bytes)
	case "memory_write":
		return "memory_write  " + strIn(input, "name")
	case "memory_read":
		return "memory_read  " + strIn(input, "name")
	}
	preview := result
	if len(preview) > 80 {
		preview = preview[:80] + "…"
	}
	return name + "  " + preview
}

func strIn(input map[string]any, key string) string {
	v, _ := input[key].(string)
	return v
}

func shortPath(p string) string {
	if p == "" {
		return ""
	}
	// Show at most last 3 path components.
	parts := strings.Split(filepath.ToSlash(p), "/")
	if len(parts) > 3 {
		return "…/" + strings.Join(parts[len(parts)-3:], "/")
	}
	return p
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		s = s[:idx]
	}
	if len(s) > 80 {
		s = s[:80] + "…"
	}
	return s
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
