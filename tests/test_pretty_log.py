"""Hermetic tests for LLM Mailroom training log TUI."""
from __future__ import annotations

import json
from pathlib import Path

from training.pretty_log import (
    epoch_strip,
    parse_log_step_line,
    progress_bar,
    render_owl,
    render_step_event,
    render_watch_snapshot,
    stage_gremlin,
    use_color,
)


def test_use_color_respects_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert use_color(stream=type("S", (), {"isatty": lambda self: True})()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert use_color(stream=type("S", (), {"isatty": lambda self: True})()) is False


def test_epoch_strip_and_progress_bar():
    assert "1/3" in epoch_strip(1, 3)
    assert "█" in epoch_strip(1, 3)
    bar = progress_bar(50, 4221)
    assert "micro [" in bar
    assert "50/4221" in bar
    assert "░" in bar
    assert "█" in progress_bar(500, 4221)


def test_render_step_plain_no_ansi():
    row = {
        "epoch": 1,
        "epochs": 3,
        "step": 50,
        "loss": 2.085931,
        "steps_per_sec": 0.2679,
        "samples_per_sec": 1.07,
        "eta": "4h 19m 28s",
        "micro_done": 50,
        "micro_planned": 4221,
        "event": "step",
    }
    out = render_step_event(row, on=False)
    assert "\033[" not in out
    assert "step 50" in out
    assert "2.0859" in out
    assert "4h 19m 28s" in out
    assert "1/3" in out
    assert "SORTER" in out
    assert "micro [" in out


def test_parse_log_step_line_from_trainer_text():
    line = "  step 50 loss 2.0859 0.27 steps/s 1.1 samples/s ETA 4h 19m 28s"
    parsed = parse_log_step_line(line)
    assert parsed is not None
    assert parsed["step"] == 50
    assert parsed["loss"] == 2.0859


def test_render_watch_from_fixture_jsonl(tmp_path: Path):
    jsonl = tmp_path / "train_steps.jsonl"
    row = {
        "event": "step",
        "epoch": 1,
        "epochs": 3,
        "step": 25,
        "loss": 2.005745,
        "eta": "4h 20m 27s",
        "steps_per_sec": 0.2685,
        "samples_per_sec": 1.07,
        "micro_done": 25,
        "micro_planned": 4221,
    }
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    from training.pretty_log import read_last_jsonl

    last = read_last_jsonl(jsonl)
    assert last is not None
    out = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-local logs/x.log",
        trainer_lines=["999 python training/train_modernbert.py"],
        log_tail=[],
        jsonl_row=last,
        run_tag="m9a-local-014429",
        on=False,
    )
    assert "LLM MAILROOM" in out
    assert "plurality vote" in out
    assert "INBOX" in out and "SORTER" in out
    assert "step 25" in out
    assert "m9a-local-014429" in out
    assert "▸TRAIN◂" in out
    assert "stage ▸TRAIN◂" in out
