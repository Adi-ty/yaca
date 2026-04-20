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
