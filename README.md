# mailroom-ml

Machine-learning training, evaluation, and deployment for the Digital Mailroom **ModernBERT ingest fast-path classifier** (shared encoder + conditional subclass heads).

## Prerequisites

- Python **3.11+** (`pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- Optional: Modal account + `HF_TOKEN` for cloud training (`deploy/README.md`)

## Quick start

```bash
git clone https://github.com/LLM-Mailroom-Services/mailroom-ml.git
cd mailroom-ml
uv sync --extra dev
uv run pytest -m "not fullcorpus"
```

## Commands

The full command surface (staging data, local train/eval, Modal deploy, ONNX export) lives in **[AGENTS.md](AGENTS.md)**. Start there for Hub pins, taxonomy law, and runbooks.

## Governance

Mission tracking: `governance/TASKS.md` and `governance/M9a-HANDOFF.md` (active #112 pass).
