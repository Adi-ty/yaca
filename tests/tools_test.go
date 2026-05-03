package tests

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Adi-ty/yaca/tools"
)

var ctx = context.Background()

// tool returns the named tool's Execute func or fails immediately.
func tool(t *testing.T, name string) func(map[string]any) (string, error) {
	t.Helper()
	for _, tool := range tools.All() {
		if tool.Name == name {
			return func(input map[string]any) (string, error) {
				return tool.Execute(ctx, input)
			}
		}
	}
	t.Fatalf("tool %q not found", name)
	return nil
}

// projectRoot returns the absolute path of the repo root (one dir above tests/).
func projectRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Dir(wd)
}

// ── ReadTool ──────────────────────────────────────────────────────────────────

func TestReadTool(t *testing.T) {
	read := tool(t, "read")
	root := projectRoot(t)

	out, err := read(map[string]any{"path": filepath.Join(root, "go.mod")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "github.com/Adi-ty/yaca") {
		t.Errorf("go.mod content missing module path; got:\n%s", out)
	}
}

func TestReadTool_MissingFile(t *testing.T) {
	read := tool(t, "read")

	_, err := read(map[string]any{"path": "/nonexistent/path/file.txt"})
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

// ── WriteTool ─────────────────────────────────────────────────────────────────

func TestWriteTool(t *testing.T) {
	write := tool(t, "write")

	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "hello.txt")

	out, err := write(map[string]any{"path": path, "content": "hello world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "hello.txt") {
		t.Errorf("expected path in output, got %q", out)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("file not created: %v", err)
	}
	if string(data) != "hello world" {
		t.Errorf("content mismatch: got %q", data)
	}
}

// ── EditTool ──────────────────────────────────────────────────────────────────

func TestEditTool(t *testing.T) {
	write := tool(t, "write")
	edit := tool(t, "edit")

	dir := t.TempDir()
	path := filepath.Join(dir, "edit_test.txt")

	_, err := write(map[string]any{"path": path, "content": "foo bar baz"})
	if err != nil {
		t.Fatal(err)
	}

	_, err = edit(map[string]any{
		"path":       path,
		"old_string": "bar",
		"new_string": "qux",
	})
	if err != nil {
		t.Fatalf("edit error: %v", err)
	}

	data, _ := os.ReadFile(path)
	if string(data) != "foo qux baz" {
		t.Errorf("expected %q, got %q", "foo qux baz", data)
	}
}

func TestEditTool_NotFound(t *testing.T) {
	write := tool(t, "write")
	edit := tool(t, "edit")

	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	write(map[string]any{"path": path, "content": "hello"}) //nolint

	_, err := edit(map[string]any{
		"path":       path,
		"old_string": "missing",
		"new_string": "x",
	})
	if err == nil {
		t.Fatal("expected error for missing old_string")
	}
}

func TestEditTool_Ambiguous(t *testing.T) {
	write := tool(t, "write")
	edit := tool(t, "edit")

	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	write(map[string]any{"path": path, "content": "ab ab"}) //nolint

	_, err := edit(map[string]any{
		"path":       path,
		"old_string": "ab",
		"new_string": "cd",
	})
	if err == nil {
		t.Fatal("expected error for ambiguous old_string")
	}
}

// ── BashTool ──────────────────────────────────────────────────────────────────

func TestBashTool(t *testing.T) {
	bash := tool(t, "bash")

	out, err := bash(map[string]any{"command": "echo hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.TrimSpace(out) != "hello" {
		t.Errorf("expected %q, got %q", "hello", strings.TrimSpace(out))
	}
}

func TestBashTool_StderrIncluded(t *testing.T) {
	bash := tool(t, "bash")

	// A command that writes to both stdout and stderr.
	out, err := bash(map[string]any{"command": "echo out; echo err >&2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "out") || !strings.Contains(out, "err") {
		t.Errorf("expected combined output, got %q", out)
	}
}

func TestBashTool_NonZeroExit(t *testing.T) {
	bash := tool(t, "bash")

	// Non-zero exit should NOT produce a Go error — output carries the info.
	out, err := bash(map[string]any{"command": "exit 1"})
	if err != nil {
		t.Fatalf("non-zero exit should not return Go error; got: %v", err)
	}
	_ = out
}

// ── GlobTool ─────────────────────────────────────────────────────────────────

func TestGlobTool_GoFiles(t *testing.T) {
	glob := tool(t, "glob")
	root := projectRoot(t)

	out, err := glob(map[string]any{"pattern": "*.go", "dir": root})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The project has .go files; at minimum main.go should be found.
	if out == "no files found" {
		t.Fatal("expected .go files, got none")
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	for _, l := range lines {
		if !strings.HasSuffix(l, ".go") {
			t.Errorf("non-.go file in results: %q", l)
		}
	}
}

func TestGlobTool_NoMatch(t *testing.T) {
	glob := tool(t, "glob")

	out, err := glob(map[string]any{"pattern": "*.zzznomatch", "dir": t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if out != "no files found" {
		t.Errorf("expected no-match message, got %q", out)
	}
}

func TestGlobTool_SkipsGit(t *testing.T) {
	glob := tool(t, "glob")
	root := projectRoot(t)

	out, _ := glob(map[string]any{"pattern": "*.go", "dir": root})
	if strings.Contains(out, ".git/") {
		t.Error("glob should not recurse into .git/")
	}
}

// ── ListDirTool ───────────────────────────────────────────────────────────────

func TestListDirTool(t *testing.T) {
	list := tool(t, "list_dir")
	root := projectRoot(t)

	out, err := list(map[string]any{"path": root})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Root must contain go.mod and at least one directory.
	if !strings.Contains(out, "go.mod") {
		t.Errorf("expected go.mod in listing, got:\n%s", out)
	}
	// Directories should be shown with a trailing slash.
	if !strings.Contains(out, "agent/") && !strings.Contains(out, "ai/") {
		t.Errorf("expected directories with trailing /, got:\n%s", out)
	}
}

func TestListDirTool_MissingDir(t *testing.T) {
	list := tool(t, "list_dir")

	_, err := list(map[string]any{"path": "/nonexistent/dir"})
	if err == nil {
		t.Fatal("expected error for missing directory")
	}
}

// ── ReadTool truncation ───────────────────────────────────────────────────────

func TestReadTool_Truncation(t *testing.T) {
	read := tool(t, "read")

	dir := t.TempDir()
	path := filepath.Join(dir, "big.txt")

	// Write 2100 lines so truncation triggers.
	var sb strings.Builder
	for i := 1; i <= 2100; i++ {
		sb.WriteString("line\n")
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	out, err := read(map[string]any{"path": path})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(out, "[truncated: showing first 2000 of") {
		t.Errorf("expected truncation header, got: %q", out[:min(80, len(out))])
	}
	// Strip the header line before counting content lines.
	bodyStart := strings.Index(out, "\n")
	body := out[bodyStart+1:]
	got := len(strings.Split(strings.TrimRight(body, "\n"), "\n"))
	if got != 2000 {
		t.Errorf("expected 2000 lines after truncation, got %d", got)
	}
}

func TestReadTool_NoTruncation(t *testing.T) {
	read := tool(t, "read")

	dir := t.TempDir()
	path := filepath.Join(dir, "small.txt")

	var sb strings.Builder
	for i := 1; i <= 100; i++ {
		sb.WriteString("line\n")
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	out, err := read(map[string]any{"path": path})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(out, "[truncated") {
		t.Error("small file should not be truncated")
	}
}

// ── MemoryTool name validation ────────────────────────────────────────────────

func TestMemoryWriteTool_ValidName(t *testing.T) {
	memWrite := tool(t, "memory_write")
	memRead := tool(t, "memory_read")

	// Valid names: alphanumeric, hyphens, underscores.
	for _, name := range []string{"foo", "my-note", "project_ctx", "Note123"} {
		_, err := memWrite(map[string]any{"name": name, "content": "hello"})
		if err != nil {
			t.Errorf("valid name %q rejected: %v", name, err)
		}
		out, err := memRead(map[string]any{"name": name})
		if err != nil {
			t.Errorf("memory_read failed for valid name %q: %v", name, err)
		}
		if out != "hello" {
			t.Errorf("memory_read returned %q, want %q", out, "hello")
		}
	}
}

func TestMemoryWriteTool_InvalidName(t *testing.T) {
	memWrite := tool(t, "memory_write")

	// These names must be rejected to prevent path traversal.
	for _, name := range []string{"../x", "../../etc/passwd", "foo/bar", "a b", "foo.md", ""} {
		_, err := memWrite(map[string]any{"name": name, "content": "bad"})
		if err == nil {
			t.Errorf("invalid name %q should have been rejected", name)
		}
	}
}

func TestMemoryReadTool_InvalidName(t *testing.T) {
	memRead := tool(t, "memory_read")

	for _, name := range []string{"../x", "../../etc/passwd", "foo/bar", ""} {
		_, err := memRead(map[string]any{"name": name})
		if err == nil {
			t.Errorf("invalid name %q should have been rejected", name)
		}
	}
}

func TestMemoryReadTool_NotFound(t *testing.T) {
	memRead := tool(t, "memory_read")

	out, err := memRead(map[string]any{"name": "does-not-exist-xyz"})
	if err != nil {
		t.Fatalf("not-found should not error, got: %v", err)
	}
	if !strings.Contains(out, "no memory named") {
		t.Errorf("expected not-found message, got: %q", out)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
