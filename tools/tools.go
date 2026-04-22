package tools

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/Adi-ty/yaca/agent"
)

// All returns every built-in tool.
func All() []agent.Tool {
	return []agent.Tool{
		ReadTool(),
		WriteTool(),
		EditTool(),
		BashTool(),
		GlobTool(),
		GrepTool(),
		ListDirTool(),
		FetchTool(),
		MemoryReadTool(),
		MemoryWriteTool(),
	}
}

// ── ReadTool ──────────────────────────────────────────────────────────────────

func ReadTool() agent.Tool {
	return agent.Tool{
		Name:        "read",
		Description: "Read the full contents of a file and return them as a string.",
		InputSchema: objectSchema(
			propMap{"path": strProp("Absolute path to the file to read")},
			[]string{"path"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			path, err := strParam(input, "path")
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("read: %w", err)
			}
			return string(data), nil
		},
	}
}

// ── WriteTool ─────────────────────────────────────────────────────────────────

func WriteTool() agent.Tool {
	return agent.Tool{
		Name:        "write",
		Description: "Write content to a file, creating parent directories as needed. Overwrites existing files.",
		InputSchema: objectSchema(
			propMap{
				"path":    strProp("Absolute path of the file to write"),
				"content": strProp("Content to write to the file"),
			},
			[]string{"path", "content"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			path, err := strParam(input, "path")
			if err != nil {
				return "", err
			}
			content, err := strParam(input, "content")
			if err != nil {
				return "", err
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return "", fmt.Errorf("write: create parent dirs: %w", err)
			}
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				return "", fmt.Errorf("write: %w", err)
			}
			return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil
		},
	}
}

// ── EditTool ──────────────────────────────────────────────────────────────────

func EditTool() agent.Tool {
	return agent.Tool{
		Name:        "edit",
		Description: "Replace an exact string in a file. old_string must appear exactly once; use read first to verify uniqueness.",
		InputSchema: objectSchema(
			propMap{
				"path":       strProp("Absolute path to the file to edit"),
				"old_string": strProp("Exact string to find and replace (must appear exactly once in the file)"),
				"new_string": strProp("Replacement string"),
			},
			[]string{"path", "old_string", "new_string"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			path, err := strParam(input, "path")
			if err != nil {
				return "", err
			}
			oldStr, err := strParam(input, "old_string")
			if err != nil {
				return "", err
			}
			newStr, err := strParam(input, "new_string")
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("edit: read: %w", err)
			}
			content := string(data)
			n := strings.Count(content, oldStr)
			switch {
			case n == 0:
				return "", fmt.Errorf("edit: old_string not found in %s", path)
			case n > 1:
				return "", fmt.Errorf("edit: old_string appears %d times in %s (must appear exactly once)", n, path)
			}
			updated := strings.Replace(content, oldStr, newStr, 1)
			if err := os.WriteFile(path, []byte(updated), 0o644); err != nil {
				return "", fmt.Errorf("edit: write: %w", err)
			}
			return fmt.Sprintf("edited %s", path), nil
		},
	}
}

// ── BashTool ──────────────────────────────────────────────────────────────────

func BashTool() agent.Tool {
	return agent.Tool{
		Name:        "bash",
		Description: "Execute a bash command and return combined stdout and stderr. Non-zero exit codes are surfaced in output, not as errors.",
		InputSchema: objectSchema(
			propMap{"command": strProp("The bash command to execute")},
			[]string{"command"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			command, err := strParam(input, "command")
			if err != nil {
				return "", err
			}
			cmd := exec.CommandContext(ctx, "bash", "-c", command)
			out, cmdErr := cmd.CombinedOutput()
			output := string(out)
			// Context cancellation / timeout is a real error worth surfacing.
			if ctx.Err() != nil {
				return output, fmt.Errorf("bash: %w", ctx.Err())
			}
			// Non-zero exit is not an agent-level error — stderr is in output.
			_ = cmdErr
			return output, nil
		},
	}
}

// ── GlobTool ──────────────────────────────────────────────────────────────────

func GlobTool() agent.Tool {
	return agent.Tool{
		Name:        "glob",
		Description: "Find files matching a glob pattern. Skips .git, node_modules, and vendor. Use ** to match across directories.",
		InputSchema: objectSchema(
			propMap{
				"pattern": strProp("Glob pattern, e.g. *.go or **/*.ts"),
				"dir":     strProp("Root directory to search (defaults to current directory)"),
			},
			[]string{"pattern"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			pattern, err := strParam(input, "pattern")
			if err != nil {
				return "", err
			}
			root := strParamOpt(input, "dir", ".")

			var matches []string
			walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return nil // skip unreadable entries
				}
				if d.IsDir() {
					if skipDir[d.Name()] {
						return filepath.SkipDir
					}
					return nil
				}
				rel, _ := filepath.Rel(root, path)
				if globMatch(pattern, rel) {
					matches = append(matches, path)
				}
				return nil
			})
			if walkErr != nil {
				return "", fmt.Errorf("glob: %w", walkErr)
			}
			if len(matches) == 0 {
				return "no files found", nil
			}
			return strings.Join(matches, "\n"), nil
		},
	}
}

// ── GrepTool ──────────────────────────────────────────────────────────────────

func GrepTool() agent.Tool {
	return agent.Tool{
		Name:        "grep",
		Description: "Search files for a regular expression. Returns up to 100 matching lines as file:line: text.",
		InputSchema: objectSchema(
			propMap{
				"pattern": strProp("Regular expression to search for"),
				"dir":     strProp("Root directory to search (defaults to current directory)"),
				"include": strProp("Glob pattern to restrict which files are searched, e.g. *.go"),
			},
			[]string{"pattern"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			pattern, err := strParam(input, "pattern")
			if err != nil {
				return "", err
			}
			root := strParamOpt(input, "dir", ".")
			include := strParamOpt(input, "include", "")

			re, err := regexp.Compile(pattern)
			if err != nil {
				return "", fmt.Errorf("grep: invalid regex: %w", err)
			}

			const maxResults = 100
			var results []string

			walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return nil
				}
				if d.IsDir() {
					if skipDir[d.Name()] {
						return filepath.SkipDir
					}
					return nil
				}
				if include != "" {
					if ok, _ := filepath.Match(include, d.Name()); !ok {
						return nil
					}
				}
				if len(results) >= maxResults {
					return filepath.SkipAll
				}
				data, err := os.ReadFile(path)
				if err != nil {
					return nil
				}
				for i, line := range strings.Split(string(data), "\n") {
					if len(results) >= maxResults {
						break
					}
					if re.MatchString(line) {
						results = append(results, fmt.Sprintf("%s:%d: %s", path, i+1, line))
					}
				}
				return nil
			})
			if walkErr != nil {
				return "", fmt.Errorf("grep: %w", walkErr)
			}
			if len(results) == 0 {
				return "no matches found", nil
			}
			return strings.Join(results, "\n"), nil
		},
	}
}

// ── ListDirTool ───────────────────────────────────────────────────────────────

func ListDirTool() agent.Tool {
	return agent.Tool{
		Name:        "list_dir",
		Description: "List the entries of a directory. Directories are shown with a trailing /.",
		InputSchema: objectSchema(
			propMap{"path": strProp("Path to the directory to list")},
			[]string{"path"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			path, err := strParam(input, "path")
			if err != nil {
				return "", err
			}
			entries, err := os.ReadDir(path)
			if err != nil {
				return "", fmt.Errorf("list_dir: %w", err)
			}
			if len(entries) == 0 {
				return "(empty directory)", nil
			}
			var sb strings.Builder
			for _, e := range entries {
				name := e.Name()
				if e.IsDir() {
					name += "/"
				}
				sb.WriteString(name)
				sb.WriteByte('\n')
			}
			return strings.TrimRight(sb.String(), "\n"), nil
		},
	}
}

// ── shared helpers ────────────────────────────────────────────────────────────

var skipDir = map[string]bool{
	".git":         true,
	"node_modules": true,
	"vendor":       true,
}

type propMap = map[string]any

func objectSchema(properties propMap, required []string) map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
}

func strProp(description string) map[string]any {
	return map[string]any{"type": "string", "description": description}
}

func strParam(input map[string]any, key string) (string, error) {
	v, ok := input[key]
	if !ok {
		return "", fmt.Errorf("missing required parameter %q", key)
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("parameter %q must be a string, got %T", key, v)
	}
	return s, nil
}

func strParamOpt(input map[string]any, key, def string) string {
	v, ok := input[key]
	if !ok {
		return def
	}
	s, ok := v.(string)
	if !ok {
		return def
	}
	return s
}

// globMatch reports whether relPath matches pattern.
//
// Rules (applied in order, first match wins):
//  1. Strip a leading "**/" prefix — we walk all subdirs anyway.
//  2. If pattern still has no separator, match against the basename only.
//  3. Otherwise match against the full relative path (forward-slash normalised).
func globMatch(pattern, relPath string) bool {
	pat := filepath.ToSlash(strings.TrimPrefix(filepath.ToSlash(pattern), "**/"))
	rel := filepath.ToSlash(relPath)

	if !strings.Contains(pat, "/") {
		ok, _ := filepath.Match(pat, filepath.Base(rel))
		return ok
	}
	ok, _ := filepath.Match(pat, rel)
	return ok
}

// ── FetchTool ─────────────────────────────────────────────────────────────────

var htmlTagRE = regexp.MustCompile(`<[^>]*>`)

func FetchTool() agent.Tool {
	return agent.Tool{
		Name:        "fetch",
		Description: "Fetch a URL and return its content. Useful for reading documentation, APIs, or GitHub files. HTML is stripped to plain text. Response is truncated at 20 000 characters.",
		InputSchema: objectSchema(
			propMap{"url": strProp("The URL to fetch")},
			[]string{"url"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			url, err := strParam(input, "url")
			if err != nil {
				return "", err
			}
			client := &http.Client{Timeout: 15 * time.Second}
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			if err != nil {
				return "", fmt.Errorf("fetch: build request: %w", err)
			}
			req.Header.Set("User-Agent", "yaca/1.0")
			resp, err := client.Do(req)
			if err != nil {
				return "", fmt.Errorf("fetch: %w", err)
			}
			defer resp.Body.Close()

			const maxBytes = 20_000
			body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
			if err != nil {
				return "", fmt.Errorf("fetch: read body: %w", err)
			}
			text := string(body)
			if len(body) > maxBytes {
				text = text[:maxBytes] + "\n[truncated]"
			}

			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				preview := text
				if len(preview) > 500 {
					preview = preview[:500]
				}
				return fmt.Sprintf("HTTP %d: %s", resp.StatusCode, preview), nil
			}

			if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "text/html") {
				text = htmlTagRE.ReplaceAllString(text, "")
				// Collapse runs of whitespace/blank lines.
				wsRE := regexp.MustCompile(`\n{3,}`)
				text = wsRE.ReplaceAllString(strings.TrimSpace(text), "\n\n")
			}
			return text, nil
		},
	}
}

// ── MemoryReadTool / MemoryWriteTool ──────────────────────────────────────────

func memoryDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("memory: home dir: %w", err)
	}
	d := filepath.Join(home, ".yaca", "memory")
	if err := os.MkdirAll(d, 0o755); err != nil {
		return "", fmt.Errorf("memory: create dir: %w", err)
	}
	return d, nil
}

func MemoryReadTool() agent.Tool {
	return agent.Tool{
		Name:        "memory_read",
		Description: "Read a named memory entry saved by memory_write. Returns the content or a 'not found' message.",
		InputSchema: objectSchema(
			propMap{"name": strProp("Memory entry name (alphanumeric, hyphens, underscores)")},
			[]string{"name"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			name, err := strParam(input, "name")
			if err != nil {
				return "", err
			}
			d, err := memoryDir()
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(filepath.Join(d, name+".md"))
			if os.IsNotExist(err) {
				return "(no memory named " + name + ")", nil
			}
			if err != nil {
				return "", fmt.Errorf("memory_read: %w", err)
			}
			return string(data), nil
		},
	}
}

func MemoryWriteTool() agent.Tool {
	return agent.Tool{
		Name:        "memory_write",
		Description: "Save a named memory entry that persists across sessions. Use this to remember important project context, decisions, or notes.",
		InputSchema: objectSchema(
			propMap{
				"name":    strProp("Memory entry name (alphanumeric, hyphens, underscores)"),
				"content": strProp("Content to save"),
			},
			[]string{"name", "content"},
		),
		Execute: func(ctx context.Context, input map[string]any) (string, error) {
			name, err := strParam(input, "name")
			if err != nil {
				return "", err
			}
			content, err := strParam(input, "content")
			if err != nil {
				return "", err
			}
			d, err := memoryDir()
			if err != nil {
				return "", err
			}
			if err := os.WriteFile(filepath.Join(d, name+".md"), []byte(content), 0o644); err != nil {
				return "", fmt.Errorf("memory_write: %w", err)
			}
			return "saved memory: " + name, nil
		},
	}
}
