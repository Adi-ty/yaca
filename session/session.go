// Package session persists conversation history per project.
// Sessions are stored at ~/.yaca/sessions/<project-hash>/<timestamp>.json
// where the hash is derived from the absolute project path, so each
// directory gets its own isolated session history.
package session

import (
	"crypto/sha256"
	"encoding/hex"
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
	ID          string       `json:"id"`
	CreatedAt   time.Time    `json:"created_at"`
	Model       string       `json:"model"`
	ProjectPath string       `json:"project_path"`
	Messages    []ai.Message `json:"messages"`
}

// projectHash returns a 16-character hex hash of the absolute path.
// This creates a stable short identifier for the sessions subdirectory
// while making collisions between different project paths impractical.
func projectHash(projectPath string) string {
	abs, err := filepath.Abs(projectPath)
	if err != nil {
		abs = projectPath
	}
	h := sha256.Sum256([]byte(abs))
	return hex.EncodeToString(h[:8]) // 8 bytes = 16 hex chars
}

// dir returns (creating if needed) the ~/.yaca/sessions/<hash> directory
// for the given project path.
func dir(projectPath string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("session: home dir: %w", err)
	}
	hash := projectHash(projectPath)
	d := filepath.Join(home, ".yaca", "sessions", hash)
	if err := os.MkdirAll(d, 0o755); err != nil {
		return "", fmt.Errorf("session: create dir: %w", err)
	}
	return d, nil
}

// New creates a fresh in-memory session for projectPath; call Save to persist it.
func New(model, projectPath string) *Session {
	abs, _ := filepath.Abs(projectPath)
	// Include milliseconds to avoid collisions when /new is invoked rapidly.
	return &Session{
		ID:          time.Now().Format("20060102-150405.000"),
		CreatedAt:   time.Now(),
		Model:       model,
		ProjectPath: abs,
	}
}

// Save writes the session to ~/.yaca/sessions/<hash>/<id>.json.
func (s *Session) Save() error {
	d, err := dir(s.ProjectPath)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("session: marshal: %w", err)
	}
	return os.WriteFile(filepath.Join(d, s.ID+".json"), data, 0o644)
}

// Load reads a session from disk by project path and ID.
func Load(projectPath, id string) (*Session, error) {
	d, err := dir(projectPath)
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

// Latest returns the most recently saved session for projectPath, or nil
// if no sessions exist for this project.
func Latest(projectPath string) (*Session, error) {
	d, err := dir(projectPath)
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
	return Load(projectPath, id)
}
