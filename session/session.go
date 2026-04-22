package session

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/Adi-ty/yaca/ai"
)

// Session captures a full conversation history with metadata.
type Session struct {
	ID        string       `json:"id"`
	CreatedAt time.Time    `json:"created_at"`
	Model     string       `json:"model"`
	Messages  []ai.Message `json:"messages"`
}

// dir returns (creating if needed) the ~/.yaca/sessions directory.
func dir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("session: home dir: %w", err)
	}
	d := filepath.Join(home, ".yaca", "sessions")
	if err := os.MkdirAll(d, 0o755); err != nil {
		return "", fmt.Errorf("session: create dir: %w", err)
	}
	return d, nil
}

// New creates a fresh in-memory session; call Save to persist it.
func New(model string) *Session {
	return &Session{
		ID:        time.Now().Format("20060102-150405"),
		CreatedAt: time.Now(),
		Model:     model,
	}
}

// Save writes the session to ~/.yaca/sessions/<id>.json.
func (s *Session) Save() error {
	d, err := dir()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("session: marshal: %w", err)
	}
	return os.WriteFile(filepath.Join(d, s.ID+".json"), data, 0o644)
}

// Load reads a session from disk by ID.
func Load(id string) (*Session, error) {
	d, err := dir()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(d, id+".json"))
	if err != nil {
		return nil, fmt.Errorf("session: read %s: %w", id, err)
	}
	var s Session
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("session: unmarshal %s: %w", id, err)
	}
	return &s, nil
}

// Latest returns the most recently saved session, or nil if none exist.
func Latest() (*Session, error) {
	d, err := dir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(d)
	if err != nil {
		return nil, fmt.Errorf("session: list: %w", err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		return nil, nil
	}
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	id := names[0][:len(names[0])-5] // strip .json
	return Load(id)
}
