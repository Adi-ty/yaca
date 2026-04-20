package tui

import (
	_ "github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	_ "github.com/charmbracelet/lipgloss"
)

// Model is the root Bubbletea model for the TUI.
type Model struct{}

func (m Model) Init() tea.Cmd                           { return nil }
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) { return m, nil }
func (m Model) View() string                            { return "" }
