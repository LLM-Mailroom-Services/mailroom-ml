"""Hermetic tests for LLM Mailroom training log TUI."""
from __future__ import annotations

import json
import re
from pathlib import Path

from training.pretty_log import (
    ALT_ENTER,
    ALT_LEAVE,
    CURSOR_HOME,
    epoch_strip,
    job_completed_summary,
    parse_log_step_line,
    progress_bar,
    render_epoch_event,
    render_hardware_panel,
    render_header_banner,
    render_job_completed_banner,
    render_step_event,
    render_watch_snapshot,
    stage_gremlin,
    use_color,
)

_ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")

_CRINGE = (
    "(o,o)(o,o)",  # double-head ban; single (o,o) owl is intentional
    "(  V  V  )",
    "(◕‿◕)",
    "(>^.^)>",
    "(o_o)?",
    "(^._.^)",
    "♡",
)


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _framed_widths(text: str) -> list[int]:
    widths: list[int] = []
    for line in _strip(text).splitlines():
        if line[:1] in {"╔", "╚", "║", "╠"}:
            widths.append(len(line))
    return widths


def _step_row() -> dict:
    return {
        "epoch": 1,
        "epochs": 3,
        "step": 50,
        "loss": 2.085931,
        "steps_per_sec": 0.2679,
        "samples_per_sec": 1.07,
        "eta": "4h 19m 28s",
        "micro_done": 50,
        "micro_planned": 4221,
        "wall_s": 120.5,
        "event": "step",
    }


def _epoch_row() -> dict:
    return {
        "event": "epoch",
        "epoch": 1,
        "epochs": 3,
        "loss": 2.0,
        "loss_endpoint": 1.95,
        "val_loss": 1.9,
        "lr": 2e-5,
        "doc_type_window_acc": 0.82,
        "doc_type_doc_acc": 0.81,
        "doc_type_macro_f1_observed": 0.78,
        "doc_type_ece": 0.05,
        "doc_type_ece_calibrated": 0.02,
        "subclass_objective": 0.65,
        "per_head_macro_f1_observed": {"head_a": 0.70},
        "per_head_ece_calibrated": {"head_a": 0.03},
        "selected_this_epoch": True,
        "selection_epoch": 1,
        "epoch_wall_s": 300,
    }


def _hw_sample() -> dict:
    return {
        "available": True,
        "gpu_index": "0",
        "gpu_name": "NVIDIA Test GPU",
        "gpu_util": "42",
        "gpu_mem_used": "8000",
        "gpu_mem_total": "24576",
        "gpu_temp": "65",
        "gpu_power": "200",
        "cpu_pct": "12.5",
        "rss": "2048M",
        "pid": 123,
        "host_mem_used": 16000,
        "host_mem_total": 64000,
        "load1": "1.23",
    }


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
    assert "(o,o) step 50" in out
    assert "sorting…" in out
    assert "2.0859" in out
    assert "4h 19m 28s" in out
    assert "1/3" in out
    assert "SORTER" in out
    assert "micro [" in out
    for face in _CRINGE:
        assert face not in out
    blink = render_step_event(row, on=False, blink=True)
    assert "(-,-) step 50" in blink
    assert "(-,-) sorting…" in blink
    idle = render_step_event({**row, "micro_done": None, "micro_planned": None}, on=False)
    assert "(o,o) waiting on the next tray" in idle

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
    assert "MAILROOM" in out
    assert "DIGITAL MAILROOM" in out
    assert "plurality vote" in out or "ModernBERT" in out
    assert "INBOX" in out and "SORTER" in out
    assert "(o,o) step 25" in out
    assert "(o,o)  stage ▸TRAIN◂" in out
    assert "(o,o) INBOX" in out
    assert "sorting…" in out
    assert "m9a-local-014429" in out
    assert "▸TRAIN◂" in out
    assert "stage ▸TRAIN◂" in out
    for face in _CRINGE:
        assert face not in out

def test_header_wordmark_aligned_no_faces():
    for cols, compact in ((58, True), (80, False), (120, False)):
        lines = render_header_banner(
            run_tag="m9a-test",
            on=False,
            width=cols,
            compact=compact,
        )
        stripped = [_strip(x) for x in lines]
        widths = [len(x) for x in stripped]
        assert len(set(widths)) == 1, (cols, widths)
        assert all(w == cols for w in widths)
        joined = "\n".join(stripped)
        assert "MAILROOM" in joined
        assert "DIGITAL MAILROOM" in joined
        assert "INBOX" in joined and "SORTER" in joined and "TRAY" in joined
        # One symmetrical owl mark (open or blink), never the double-head.
        assert "(o,o)" in joined or "(-,-)" in joined
        assert "(o,o)(o,o)" not in joined
        for face in _CRINGE:
            assert face not in joined
        # Wordmark rows (between top/bottom frame, before DIGITAL) share one width.
        body = stripped[1:-1]
        assert body
        assert len({len(x) for x in body}) == 1


def test_brand_colors_truecolor_hermes_amber_wordmark():
    blue = "38;2;36;86;214"
    teal = "38;2;14;116;144"
    cyan = "38;2;56;224;214"
    gold = "38;2;245;196;69"
    amber = "38;2;255;176;0"  # #FFB000 Hermes face on wordmark/owl
    highlight = "38;2;255;224;138"  # #FFE08A top bevel
    extrude = "38;2;138;90;18"  # #8A5A12 dark bronze drop
    header = "\n".join(render_header_banner(run_tag="m9a-test", on=True, width=80))
    # Frame/pipeline stay blue-teal; wordmark is Hermes amber — not blue glyphs.
    assert blue in header  # frame
    assert teal in header  # INBOX → SORTER → TRAY
    assert amber in header
    assert highlight in header
    assert extrude in header
    assert extrude != amber
    assert gold in header or amber in header
    assert "38;5;214" not in header
    # Wordmark block uses amber RGB (high R, mid G, low B), not mailroom blue.
    assert "38;2;255;176;0" in header or "38;2;245;196;69" in header
    out = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=["123 python training/train_modernbert.py"],
        log_tail=[],
        jsonl_row=_step_row(),
        epoch_jsonl_row=None,
        run_tag="m9a-test",
        on=True,
        width=80,
        height=24,
        resources=_hw_sample(),
    )
    assert blue in out
    assert teal in out
    assert cyan in out
    assert amber in out
    assert extrude in out
    assert "38;5;214" not in out
    for face in _CRINGE:
        assert face not in _strip(out)


def test_width_tiers_narrow_mini_wide_hardware():
    step = _step_row()
    hw = _hw_sample()
    narrow = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=step,
        run_tag="m9a-test",
        on=False,
        width=58,
        height=16,
        resources=hw,
    )
    narrow_stripped = _strip(narrow)
    assert "MAILROOM" in narrow_stripped
    assert "DIGITAL MAILROOM" in narrow_stripped
    assert "INBOX" in narrow_stripped
    assert "HW " in narrow_stripped
    assert "(o,o)" in narrow_stripped
    assert "HARDWARE" not in narrow_stripped
    for face in _CRINGE:
        assert face not in narrow_stripped
    medium = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=step,
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=hw,
    )
    medium_stripped = _strip(medium)
    assert "MAILROOM" in medium_stripped
    assert "(o,o) HARDWARE" in medium_stripped
    assert "(o,o)  stage ▸TRAIN◂" in medium_stripped
    assert "(o,o) step 50" in medium_stripped
    assert "(o,o) INBOX" in medium_stripped
    for face in _CRINGE:
        assert face not in medium_stripped
    wide = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=step,
        run_tag="m9a-test",
        on=False,
        width=120,
        height=30,
        resources=hw,
    )
    wide_stripped = _strip(wide)
    assert "MAILROOM" in wide_stripped
    assert "(o,o) HARDWARE" in wide_stripped
    assert any("HARDWARE" in line and "step 50" in line for line in wide_stripped.splitlines())
    for face in _CRINGE:
        assert face not in wide_stripped
    for cols, rows in ((58, 16), (80, 24), (120, 30)):
        out = render_watch_snapshot(
            timestamp="2026-09-27 01:00:00",
            lock_line="123 m9a-test logs/x.log",
            trainer_lines=[],
            log_tail=[],
            jsonl_row=step,
            run_tag="m9a-test",
            on=False,
            width=cols,
            height=rows,
            resources=hw,
        )
        stripped = _strip(out)
        for line in stripped.splitlines():
            assert len(line) <= cols
        framed = _framed_widths(out)
        assert framed
        assert len(set(framed)) == 1, (cols, set(framed))


def test_monitor_owl_blink_and_loading():
    step = _step_row()
    open_snap = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=step,
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=_hw_sample(),
        blink=False,
    )
    assert "(o,o)  stage ▸TRAIN◂" in open_snap
    assert "(o,o) step 50" in open_snap
    assert "(o,o) sorting…" in open_snap
    assert "(o,o) HARDWARE" in open_snap
    assert "(o,o) INBOX → SORTER → TRAY" in open_snap
    blink_snap = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=step,
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=_hw_sample(),
        blink=True,
    )
    assert "(-,-)  stage ▸TRAIN◂" in blink_snap
    assert "(-,-) step 50" in blink_snap
    assert "(-,-) sorting…" in blink_snap
    assert "(-,-) HARDWARE" in blink_snap
    assert "(-,-) INBOX → SORTER → TRAY" in blink_snap
    for face in _CRINGE:
        assert face not in open_snap
        assert face not in blink_snap


def test_mailroom_m_has_two_peaks_not_n():
    """Both M glyphs must show outer stems + center valley (not an N diagonal)."""
    from training.pretty_log import _WM_FONT_3, _WM_FONT_4

    for font in (_WM_FONT_3, _WM_FONT_4):
        m = font["M"]
        # Middle peak row(s) must have a center solid between two gaps / stems.
        # Row index 1 is the first inner peak row for both fonts.
        peak = m[1]
        assert peak.count("█") >= 4, peak
        # Must not be the old N-shaped ███ ██ / ██ ███ pair.
        assert peak != "███ ██"
        assert m[2] != "██ ███"
    header = "\n".join(render_header_banner(run_tag="m9a-test", on=False, width=80))
    # Face bitmap of M appears (plain uses █); last letter region has two peaks.
    assert "███ ███" in header or "██ ██" in header
    assert "(o,o)" in header
    assert "(o,o)(o,o)" not in header


def test_wordmark_fonts_uniform_glyph_dimensions():
    """Every letter in each wordmark font tier must share one row count and
    one column count (catches thin/1-col strokes and width mismatches like
    the old 7-col M sitting inside a font of 6-col letters)."""
    from training.pretty_log import _WM_FONT_3, _WM_FONT_4, _assert_uniform_font

    letters = "THEMAILRO"
    for font in (_WM_FONT_3, _WM_FONT_4):
        glyphs = {ch: font[ch] for ch in letters}
        row_counts = {len(rows) for rows in glyphs.values()}
        col_counts = {len(row) for rows in glyphs.values() for row in rows}
        assert len(row_counts) == 1, row_counts
        assert len(col_counts) == 1, col_counts
        # Validator itself must accept a uniform font and reject a mismatched one.
        _assert_uniform_font(glyphs)
        broken = dict(glyphs)
        broken["M"] = [row + "█" for row in broken["M"]]
        try:
            _assert_uniform_font(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("expected _assert_uniform_font to reject mismatched widths")


def test_epoch_extras_lr_f1_ece_subclass():
    row = _epoch_row()
    for on in (False, True):
        out = render_epoch_event(row, on=on)
        stripped = _strip(out)
        assert "(o,o) epoch 1" in stripped
        assert "2.00e-05" in stripped  # lr
        assert "F1" in stripped and "0.7800" in stripped  # macro-F1
        assert "cal" in stripped and "0.0200" in stripped  # calibrated ECE
        assert "subclass_obj" in stripped and "0.6500" in stripped
    snap = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=None,
        epoch_jsonl_row=row,
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=None,
    )
    stripped = _strip(snap)
    assert "2.00e-05" in stripped
    assert "0.7800" in stripped
    assert "0.0200" in stripped
    assert "0.6500" in stripped


def test_injected_hardware_no_subprocess(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("must use injected dict, not nvidia-smi/ps")

    monkeypatch.setattr("training.pretty_log.subprocess.run", _fail)
    hw = _hw_sample()
    full = render_hardware_panel(hw, on=False)
    assert "NVIDIA Test GPU" in full
    assert "42%" in full
    assert "(o,o) HARDWARE" in full
    compact = render_hardware_panel(hw, on=False, width=58, compact=True)
    assert "(o,o) HW " in compact
    assert "42%" in compact
    snap = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=_step_row(),
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=hw,
    )
    assert "NVIDIA Test GPU" in snap
    assert "42%" in snap


def test_plain_output_free_of_alt_screen_sequences():
    step = _step_row()
    out = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=step,
        epoch_jsonl_row=_epoch_row(),
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=_hw_sample(),
    )
    assert "\033[" not in out
    assert "?1049" not in out
    assert ALT_ENTER not in out
    assert ALT_LEAVE not in out
    assert CURSOR_HOME not in out
    assert "\033[2J" not in out
    assert "\033[?25" not in out
    assert "MAILROOM" in out
    assert "█" in out or "░" in out
    for face in _CRINGE:
        assert face not in out
    colored = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=step,
        run_tag="m9a-test",
        on=True,
        width=80,
        height=24,
        resources=_hw_sample(),
    )
    assert "?1049" not in colored
    assert CURSOR_HOME not in colored


def test_job_completed_banner_in_snapshot():
    summary = {
        "epochs_run": 1,
        "epochs": [{"epoch": 1}],
        "wall_s": 5578.9,
        "test_metrics": {
            "n_docs": 323,
            "doc_type_acc": 0.8669,
            "subclass_acc_conditional": 0.475,
        },
    }
    assert job_completed_summary(
        run_tag="m9a-smoke",
        trainer_lines=[],
        jsonl_row={"epoch": 1, "epochs": 1, "micro_done": 100, "micro_planned": 100},
        log_tail=["run summary: device=cuda"],
        summary_path=None,
    ) == {}
    snap = render_watch_snapshot(
        timestamp="2026-09-27 04:00:00",
        lock_line=None,
        trainer_lines=[],
        log_tail=["run summary: device=cuda"],
        jsonl_row={"epoch": 1, "epochs": 1, "micro_done": 100, "micro_planned": 100},
        run_tag="m9a-smoke",
        on=False,
        width=80,
        height=24,
        resources=None,
        summary_path=None,
    )
    assert "JOB COMPLETED" in snap
    assert "stage ▸DONE◂" in snap
    banner = render_job_completed_banner(summary, run_tag="m9a-smoke", on=False, width=60)
    assert "test doc_type acc 0.8669" in banner


def test_pipeline_stages_no_gremlins():
    assert stage_gremlin("TRAIN") == ""
    assert stage_gremlin("VAL") == ""
    train_snap = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=_step_row(),
        epoch_jsonl_row=None,
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=None,
    )
    assert "stage ▸TRAIN◂" in train_snap
    assert "(o,o)  stage ▸TRAIN◂" in train_snap
    assert "(o,o) INBOX" in train_snap
    assert "TRAIN" in train_snap and "VAL" in train_snap
    assert "TEST" in train_snap and "GATE" in train_snap and "HUB" in train_snap
    for face in _CRINGE:
        assert face not in train_snap
    val_snap = render_watch_snapshot(
        timestamp="2026-09-27 01:00:00",
        lock_line="123 m9a-test logs/x.log",
        trainer_lines=[],
        log_tail=[],
        jsonl_row=_step_row(),
        epoch_jsonl_row=_epoch_row(),
        run_tag="m9a-test",
        on=False,
        width=80,
        height=24,
        resources=None,
    )
    assert "stage ▸VAL◂" in val_snap
    assert "(o,o)  stage ▸VAL◂" in val_snap
    for face in _CRINGE:
        assert face not in val_snap
