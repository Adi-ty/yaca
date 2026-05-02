// Package config manages persistent YACA configuration stored at
// ~/.yaca/config.json. This includes provider selection and API keys.
// The file is written with mode 0600 so keys are not world-readable.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const Version = "0.1.0"

// Config holds the persisted provider credentials and preferences.
type Config struct {
	Provider  string `json:"provider"`            // "anthropic" | "openai" | "ollama"
	APIKey    string `json:"api_key,omitempty"`   // not set for Ollama
	Model     string `json:"model,omitempty"`     // empty means use provider default
	OllamaURL string `json:"ollama_url,omitempty"` // defaults to http://localhost:11434/v1
}

// DefaultModel returns the recommended default model for each provider.
func DefaultModel(provider string) string {
	switch provider {
	case "anthropic":
		return "claude-3-5-sonnet-20241022"
	case "openai":
		return "gpt-4o"
	case "ollama":
		return "llama3.2"
	default:
		return ""
	}
}

// configPath returns ~/.yaca/config.json.
func configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("config: home dir: %w", err)
	}
	dir := filepath.Join(home, ".yaca")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("config: create dir: %w", err)
	}
	return filepath.Join(dir, "config.json"), nil
}

// Load reads the config from disk. Returns nil, nil if the file does not exist.
func Load() (*Config, error) {
	path, err := configPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("config: read: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("config: parse: %w", err)
	}
	return &cfg, nil
}

// Save writes cfg to disk with mode 0600.
func Save(cfg *Config) error {
	path, err := configPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("config: marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("config: write: %w", err)
	}
	return nil
}
