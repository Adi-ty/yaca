# YACA — Yet Another Coding Assistant

YACA is a terminal-based AI coding agent written in Go. It wraps any OpenAI-compatible LLM — Anthropic Claude, OpenAI GPT, or a local Ollama model — in a Bubbletea chat UI and gives the model a set of filesystem and shell tools (read, write, edit, bash, glob, grep, list) so it can autonomously explore and modify a codebase in response to natural-language requests.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  cmd/yaca/   entry point · provider detection · wiring   │
└──────────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  tui/   Bubbletea UI       │
         │  viewport · textarea       │
         │  relays agent.Events       │
         └─────────────┬──────────────┘
                       │  <-chan agent.Event
         ┌─────────────▼──────────────┐
         │  agent/   loop             │
         │  Stream → collect → tools  │
         │        → append → repeat   │
         └──────┬───────────┬─────────┘
                │           │ injected via Config.Tools
   ┌────────────▼──┐  ┌─────▼─────────────────┐
   │  ai/          │  │  tools/               │
   │  Provider     │  │  read  write  edit    │
   │  Anthropic    │  │  bash  glob   grep    │
   │  OpenAI       │  │  list_dir             │
   │  Ollama       │  └───────────────────────┘
   └───────────────┘
```

Dependency flow is strictly downward: `tui → agent → ai`. `tools/` is injected into `agent.Config` at startup so the core loop never imports tools or UI code.

## Install

```bash
git clone https://github.com/Adi-ty/yaca
cd yaca
go build -o yaca ./cmd/yaca
```

Requires Go 1.24+.

## Configure

```bash
cp .env.example .env
$EDITOR .env          # paste your API key
```

YACA selects the provider from the first matching environment variable:

| Priority | Variable | Provider | Default model |
|----------|----------|----------|---------------|
| 1 | `ANTHROPIC_API_KEY` | Anthropic Claude | `claude-3-5-sonnet-20241022` |
| 2 | `OPENAI_API_KEY` | OpenAI | `gpt-4o` |
| 3 | *(neither set)* | Ollama (localhost) | `llama3.2` |

Override the model without changing the key:

```bash
YACA_MODEL=claude-opus-4-5 ./yaca
YACA_MODEL=gpt-4o-mini     ./yaca
YACA_MODEL=llama3.2        ./yaca
```

## Run

```bash
./yaca
# or without a binary:
go run ./cmd/yaca
```

**Keys:** `Enter` sends · `Ctrl+C` quits · `PgUp` / `PgDn` scrolls history.

## Switch providers

Set the relevant env var before launching — there is no in-session `/provider` command yet:

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-... ./yaca

# OpenAI
OPENAI_API_KEY=sk-... ./yaca

# Ollama (no key required, model must be pulled first)
ollama pull llama3.2
./yaca
```

## Test

```bash
go test ./...
```

The suite in `tests/` exercises all seven tools without requiring an API key.
