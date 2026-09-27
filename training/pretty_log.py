#!/usr/bin/env python3
"""LLM Mailroom pixel-owl live TUI for M9a training monitors (stdout only; never mutates logs)."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Rounded inner panels
TL, TR, BL, BR = "╭", "╮", "╰", "╯"
H, V = "─", "│"
# Outer mailroom frame (double line)
DTL, DTR, DBL, DBR = "╔", "╗", "╚", "╝"
DH, DV = "═", "║"

PANEL_W = 58
_INNER_BOX_W = 52
MIN_W = 58
MAX_W = 120

# Mailroom brand (blue -> teal gradient + gold), truecolor only.
BLUE = (36, 86, 214)      # #2456d6
TEAL = (14, 116, 144)     # #0e7490
CYAN = (56, 224, 214)     # #38e0d6
NAVY = (11, 42, 74)       # #0b2a4a
GOLD = (245, 196, 69)     # #f5c445
SNOW = (251, 252, 254)    # #fbfcfe
SKY = (124, 155, 255)     # #7c9bff (dark-theme brand accent)

ALT_ENTER = "\033[?1049h"
ALT_LEAVE = "\033[?1049l"
CURSOR_HOME = "\033[H"
CLEAR_SCREEN = "\033[2J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

_ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")


def use_color(stream: Any = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    fh = stream or sys.stdout
    return hasattr(fh, "isatty") and fh.isatty()


def _c(code: str, text: str, *, on: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if on else text


def _rgb(role: tuple[int, int, int]) -> str:
    r, g, b = role
    return f"38;2;{r};{g};{b}"


def _rgb_bg(role: tuple[int, int, int]) -> str:
    r, g, b = role
    return f"48;2;{r};{g};{b}"


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def palette(on: bool) -> dict[str, Any]:
    """Mailroom blue-to-teal brand with snow owl white and gold (truecolor when TTY)."""
    return {
        "title": lambda t: _c(_rgb(BLUE), t, on=on),
        "snow": lambda t: _c(_rgb(SNOW), t, on=on),
        "cream": lambda t: _c(_rgb(SNOW), t, on=on),
        "accent": lambda t: _c(_rgb(CYAN), t, on=on),
        "teal": lambda t: _c(_rgb(TEAL), t, on=on),
        "cyan": lambda t: _c(_rgb(CYAN), t, on=on),
        "mint": lambda t: _c(_rgb(SKY), t, on=on),
        "gold": lambda t: _c(_rgb(GOLD), t, on=on),
        "stamp": lambda t: _c(_rgb(CYAN), t, on=on),
        "blue": lambda t: _c(_rgb(BLUE), t, on=on),
        "navy": lambda t: _c(_rgb(NAVY), t, on=on),
        "dim": lambda t: _c("2", t, on=on),
        "warn": lambda t: _c(_rgb(GOLD), t, on=on),
        "frame": lambda t: _c(_rgb(BLUE), t, on=on),
    }


def mailroom_tagline() -> str:
    return "ingest fast-path · plurality vote · calibration"


def mailroom_pipeline_hint(active: str = "TRAIN") -> str:
    stages = ["TRAIN", "VAL", "TEST", "GATE", "HUB"]
    parts: list[str] = []
    for s in stages:
        if s == active:
            parts.append(f"▸{s}◂")
        else:
            parts.append(s)
    return " · ".join(parts)


def mailroom_route_banner() -> str:
    return "INBOX → SORTER → TRAY"


# ---------------------------------------------------------------- owl + gremlins
# ASCII emoticon snowy owl (open eyes) and blink variant (closed eyes).
# Stays text-only so it renders over ssh, in pipes, and in --plain mode.
OWL_FULL_OPEN: list[str] = [
    r"  ___  ___  ",
    r" (o,o)(o,o) ",
    r" (  V  V  ) ",
    r"((__v__v__))",
    r"  (~~ ~~)   ",
    r" //(  )\\   ",
    r" ^^   ^^    ",
]

OWL_FULL_BLINK: list[str] = [
    r"  ___  ___  ",
    r" (-,-)(-,-) ",
    r" (  V  V  ) ",
    r"((__v__v__))",
    r"  (~~ ~~)   ",
    r" //(  )\\   ",
    r" ^^   ^^    ",
]

OWL_MINI_OPEN: list[str] = [
    r"(o,o)",
    r"( V )",
    r"(~~ )",
]

OWL_MINI_BLINK: list[str] = [
    r"(-,-)",
    r"( V )",
    r"(~~ )",
]

# Backwards-compatible aliases (pixel grids removed in favor of emoticons).
OWL_FULL = OWL_FULL_OPEN
OWL_MINI = OWL_MINI_OPEN

# Digital terminal gremlins: tiny mailroom helpers, one per pipeline stage.
STAGE_GREMLINS: dict[str, str] = {
    "TRAIN": "(>^.^)>",
    "VAL": "(o_o)?",
    "TEST": "(=_=)o",
    "GATE": "(¬‿¬)!",
    "HUB": "(^._.^)/",
}

FOOTER_GREMLIN = "(^._.^)ﾉ"


def stage_gremlin(stage: str) -> str:
    return STAGE_GREMLINS.get(stage, "(o_o)")


def render_owl_pixels(grid: list[str], *, on: bool, blink: bool = False) -> list[str]:
    """Backwards-compatible hook: colorize given ASCII lines (no pixel cells)."""
    p = palette(on)
    out: list[str] = []
    for line in grid:
        if not on:
            out.append(line)
            continue
        styled = line
        for eye in ("o,o", "O,O", "o,O", "-,-"):
            styled = styled.replace(eye, p["gold"](eye))
        for beak in ("V", "v"):
            styled = styled.replace(beak, p["gold"](beak))
        styled = styled.replace("~~", p["teal"]("~~"))
        styled = styled.replace("^", p["blue"]("^"))
        out.append(styled)
    return out


def render_owl(*, mini: bool = False, on: bool, blink: bool = False) -> list[str]:
    if mini:
        grid = OWL_MINI_BLINK if blink else OWL_MINI_OPEN
    else:
        grid = OWL_FULL_BLINK if blink else OWL_FULL_OPEN
    return render_owl_pixels(grid, on=on, blink=blink)


def owl_width(*, mini: bool = False) -> int:
    grid = OWL_MINI if mini else OWL_FULL
    return max(len(r) for r in grid)


# ---------------------------------------------------------------- sizing

def _term_size() -> tuple[int, int]:
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        return int(size.columns), int(size.lines)
    except Exception:
        return 80, 24


def _panel_width(cols: int) -> int:
    if cols < MIN_W:
        return max(cols, 40)
    return min(cols, MAX_W)


def _layout(cols: int, rows: int) -> str:
    if cols >= 100 and rows >= 28:
        return "wide"
    if cols < 70 or rows < 20:
        return "narrow"
    return "medium"


def epoch_strip(epoch: int | None, epochs: int | None) -> str:
    if not epoch or not epochs or int(epochs) <= 0:
        return ""
    e, n = int(epoch), int(epochs)
    e = max(1, min(e, n))
    strip = "".join("█" if i < e else "░" for i in range(n))
    return f"epoch [{strip}] {e}/{n}"


def progress_bar(
    micro_done: int | None,
    micro_planned: int | None,
    *,
    width: int = 22,
    tick: int = 0,
    on: bool = False,
) -> str:
    if not micro_done or not micro_planned or micro_planned <= 0:
        return ""
    ratio = min(1.0, micro_done / micro_planned)
    filled = int(width * ratio)
    if tick and on and filled > 0:
        # Animate only the leading cell; keep classic glyphs elsewhere.
        bar = "█" * (filled - 1) + ("▓" if tick % 2 else "█") + "░" * (width - filled)
    else:
        bar = "█" * filled + "░" * (width - filled)
    pct = 100.0 * micro_done / micro_planned
    return f"micro [{bar}] {pct:.1f}% ({micro_done}/{micro_planned})"


def _pad_visible(text: str, width: int) -> str:
    pad = width - _visible_len(text)
    if pad <= 0:
        return text
    return text + " " * pad


def _truncate_visible(text: str, width: int) -> str:
    if _visible_len(text) <= width:
        return text
    plain = _ANSI_RE.sub("", text)
    if len(plain) <= width:
        return text
    return plain[: max(0, width - 1)] + "…"


def _panel_row(content: str, *, width: int = PANEL_W, on: bool, sides: bool = True) -> str:
    inner = width - 2
    if sides:
        inner -= 2
    body = _truncate_visible(content, inner)
    body = _pad_visible(body, inner)
    if sides:
        row = f"{DV} {body} {DV}"
    else:
        row = body
    p = palette(on)
    if on and sides:
        return p["frame"](DV) + " " + body + " " + p["frame"](DV)
    return row


def _centered_row(owl_line: str, *, width: int, on: bool) -> str:
    inner = width - 4
    w = _visible_len(owl_line)
    if w >= inner:
        body = _truncate_visible(owl_line, inner)
    else:
        left = (inner - w) // 2
        right = inner - w - left
        body = " " * left + owl_line + " " * right
    p = palette(on)
    if on:
        return p["frame"](DV) + " " + body + " " + p["frame"](DV)
    return f"{DV} {body} {DV}"


def render_header_banner(
    *,
    run_tag: str | None,
    on: bool,
    width: int = PANEL_W,
    blink: bool = False,
    compact: bool = False,
) -> list[str]:
    p = palette(on)
    w = max(width, 20)
    inner = w - 2
    owl_lines = render_owl(mini=compact, on=on, blink=blink)

    line1_plain = "◈ LLM MAILROOM ◈  ModernBERT fast-path"
    if run_tag:
        tag = run_tag if len(run_tag) <= 24 else run_tag[:23] + "…"
        line1_plain = f"◈ LLM MAILROOM ◈  {tag}"
    line2_plain = f"✉ {mailroom_route_banner()} · {mailroom_tagline()}"
    if _visible_len(line2_plain) > inner - 2:
        line2_plain = f"✉ {mailroom_tagline()}"

    top = DTL + DH * inner + DTR
    bot = DBL + DH * inner + DBR
    rows: list[str] = [p["frame"](top) if on else top]
    for ol in owl_lines:
        rows.append(_centered_row(ol, width=w, on=on))
    if on:
        row1 = _panel_row(
            p["title"]("◈ LLM MAILROOM ◈")
            + p["snow"]("  ModernBERT fast-path  ")
            + (p["gold"](f"{run_tag}  ") if run_tag else ""),
            on=False,
            sides=True,
            width=w,
        )
        row2 = _panel_row(
            p["gold"]("✉ ")
            + p["dim"](mailroom_route_banner())
            + p["dim"](" · ")
            + p["accent"](mailroom_tagline()),
            on=False,
            sides=True,
            width=w,
        )
    else:
        row1 = DV + " " + _pad_visible(line1_plain, inner - 2) + " " + DV
        row2 = DV + " " + _pad_visible(line2_plain, inner - 2) + " " + DV
    rows.extend([row1, row2, p["frame"](bot) if on else bot])
    return rows


def render_status_bar(
    *,
    timestamp: str,
    stage: str,
    on: bool,
    width: int = PANEL_W,
) -> str:
    p = palette(on)
    left = f"watch @ {timestamp}"
    right = f"✉ {stage_gremlin(stage)} stage ▸{stage}◂"
    gap = width - 4 - len(left) - len(right)
    if gap < 1:
        gap = 1
    plain = left + " " * gap + right
    if not on:
        return _panel_row(plain, on=on, width=width)
    styled = (
        p["dim"]("✉ stage ")
        + p["stamp"](f"▸{stage}◂")
        + p["dim"](f" {stage_gremlin(stage)}")
    )
    gap2 = width - 4 - _visible_len(left) - _visible_len(styled)
    if gap2 < 1:
        gap2 = 1
    styled = p["dim"](left) + " " * gap2 + styled
    return _panel_row(styled, on=False, sides=True, width=width)


def _box(title: str, lines: list[str], *, width: int = _INNER_BOX_W, on: bool) -> str:
    p = palette(on)
    inner_w = max(width - 2, 20)
    title_plain = title.strip()
    top_inner = f" {title_plain} "
    if len(title_plain) > inner_w - 2:
        title_plain = title_plain[: inner_w - 3] + "…"
        top_inner = f" {title_plain} "
    pad = inner_w - len(top_inner)
    if on:
        top = TL + p["title"](title_plain) + H * max(0, pad) + TR
    else:
        top = TL + top_inner + H * pad + TR
    body: list[str] = []
    for line in lines:
        plain = _ANSI_RE.sub("", line) if on else line
        if len(plain) > inner_w:
            plain = plain[: inner_w - 1] + "…"
            if on:
                line = plain
        body.append(V + " " + (line if on else plain.ljust(inner_w - 1)) + V)
    bottom = BL + H * inner_w + BR
    return "\n".join([top, *body, bottom])


def _metric(label: str, value: str, *, label_w: int = 12, total_w: int = 48, on: bool = False) -> str:
    p = palette(on)
    val_w = max(total_w - label_w - 1, 8)
    plain = f"{label:<{label_w}} {value:>{val_w}}"
    if not on:
        return plain
    return p["dim"](f"{label:<{label_w}}") + " " + p["snow"](f"{value:>{val_w}}")


def fmt_loss(loss: float | None) -> str:
    if loss is None:
        return "—"
    return f"{loss:.4f}"


def fmt_rate(x: float | None, unit: str) -> str:
    if x is None:
        return f"— {unit}"
    return f"{x:.2f} {unit}"


def fmt_f1(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"


def fmt_lr(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.2e}"
    except (TypeError, ValueError):
        return "—"


def fmt_progress(micro_done: int | None, micro_planned: int | None) -> str:
    return progress_bar(micro_done, micro_planned)


def fmt_loss_delta(cur: Any, prev: Any) -> str:
    try:
        if cur is None or prev is None:
            return ""
        d = float(cur) - float(prev)
        sign = "+" if d >= 0 else ""
        return f" (Δ {sign}{d:.4f})"
    except (TypeError, ValueError):
        return ""


def step_spark(step: int) -> str:
    """Kawaii ASCII on milestone steps only."""
    if step <= 0:
        return ""
    if step == 1 or step % 500 == 0:
        return " ♡"
    if step % 100 == 0:
        return " ·"
    return ""


def _mini_owl_block(*, on: bool, blink: bool = False) -> list[str]:
    return render_owl(mini=True, on=on, blink=blink)


def _combine_owl_metrics(
    owl_lines: list[str],
    metric_lines: list[str],
    *,
    inner_w: int,
    on: bool,
) -> list[str]:
    owl_w = max((_visible_len(x) for x in owl_lines), default=0)
    gap = 2
    right_w = max(inner_w - 1 - owl_w - gap, 16)
    out: list[str] = []
    for i in range(max(len(owl_lines), len(metric_lines))):
        left = owl_lines[i] if i < len(owl_lines) else " " * owl_w
        right = metric_lines[i] if i < len(metric_lines) else ""
        right = _truncate_visible(right, right_w)
        right = _pad_visible(right, right_w)
        combined = left + " " * gap + right
        combined = _truncate_visible(combined, inner_w - 1)
        if _visible_len(combined) < inner_w - 1:
            combined = _pad_visible(combined, inner_w - 1)
        out.append(combined)
    return out


def render_step_event(
    row: dict,
    *,
    on: bool | None = None,
    width: int = _INNER_BOX_W,
    prev_row: dict | None = None,
    tick: int = 0,
    blink: bool = False,
    show_owl: bool = True,
) -> str:
    on = use_color() if on is None else on
    p = palette(on)
    epoch = row.get("epoch")
    epochs = row.get("epochs")
    step = row.get("step")
    loss = row.get("loss")
    eta = row.get("eta") or _fmt_duration(row.get("eta_s"))
    sps = row.get("steps_per_sec")
    samp = row.get("samples_per_sec")
    micro_done = row.get("micro_done")
    micro_planned = row.get("micro_planned")
    wall_s = row.get("wall_s")
    spark = step_spark(int(step or 0))

    ep_label = epoch_strip(
        int(epoch) if epoch is not None else None,
        int(epochs) if epochs is not None else None,
    )
    title = f"✉ step {step}{spark} → SORTER · doc_type slot"
    if not ep_label:
        ep_label = "routing windows → plurality vote"

    inner_w = max(width - 2, 20)
    content_w = inner_w - 1
    bar_w = 12 if width < 52 else (30 if width > 70 else 22)
    bar = progress_bar(micro_done, micro_planned, width=bar_w, tick=tick, on=bool(on))
    delta = fmt_loss_delta(loss, (prev_row or {}).get("loss"))
    loss_val = fmt_loss(float(loss) if loss is not None else None) + delta

    right_lines = [
        ep_label,
        f"loss {loss_val}  ·  ETA {eta}",
        f"{fmt_rate(float(sps) if sps is not None else None, 'steps/s')}"
        f"  ·  {fmt_rate(float(samp) if samp is not None else None, 'samples/s')}",
    ]
    if wall_s is not None:
        right_lines.append(f"wall {_fmt_duration(float(wall_s))}  ·  micro {micro_done}/{micro_planned}" if micro_done else f"wall {_fmt_duration(float(wall_s))}")

    if not on:
        if show_owl:
            owl_lines = _mini_owl_block(on=False, blink=blink)
            top = _combine_owl_metrics(owl_lines, right_lines, inner_w=inner_w, on=False)
            lines_plain = [*top, bar] if bar else top
            return _box(title, lines_plain, on=on, width=width)
        lines_plain = [*right_lines, bar] if bar else right_lines
        return _box(title, lines_plain, on=on, width=width)

    if show_owl:
        owl_lines = _mini_owl_block(on=True, blink=blink)
        styled_right = [
            p["accent"](right_lines[0]),
            p["mint"](f"loss {loss_val}") + p["dim"]("  ·  ") + p["gold"](f"ETA {eta}"),
            p["cyan"](right_lines[2]),
        ]
        if len(right_lines) > 3:
            styled_right.append(p["dim"](right_lines[3]))
        top = _combine_owl_metrics(owl_lines, styled_right, inner_w=inner_w, on=True)
        if bar:
            top.append(p["gold"](bar))
        return _box(title, top, on=on, width=width)

    lines_col = [
        p["accent"](ep_label),
        p["mint"](f"loss {loss_val}") + p["dim"]("  ·  ") + p["gold"](f"ETA {eta}"),
        p["cyan"](right_lines[2]),
    ]
    if len(right_lines) > 3:
        lines_col.append(p["dim"](right_lines[3]))
    if bar:
        lines_col.append(p["gold"](bar))
    return _box(title, lines_col, on=on, width=width)


def render_epoch_event(
    row: dict,
    *,
    on: bool | None = None,
    width: int = _INNER_BOX_W,
    blink: bool = False,
    show_owl: bool = True,
) -> str:
    on = use_color() if on is None else on
    p = palette(on)
    epoch = row.get("epoch", "?")
    epochs = row.get("epochs")
    val_loss = row.get("val_loss")
    train_loss = row.get("loss")
    loss_endpoint = row.get("loss_endpoint")
    lr = row.get("lr")
    selected = row.get("selected_this_epoch")
    gate = row.get("gate_met")
    selection_epoch = row.get("selection_epoch")
    eta_rem = row.get("eta_remaining") or row.get("eta")
    epoch_wall_s = row.get("epoch_wall_s")
    win_acc = row.get("doc_type_window_acc")
    doc_acc = row.get("doc_type_doc_acc")
    macro_f1 = row.get("doc_type_macro_f1_observed")
    ece = row.get("doc_type_ece")
    ece_cal = row.get("doc_type_ece_calibrated")
    sub_obj = row.get("subclass_objective")

    ep_strip = epoch_strip(
        int(epoch) if str(epoch).isdigit() else None,
        int(epochs) if epochs is not None else None,
    )
    title = f"✉ epoch {epoch} → TRAY · subclass labels on parcels"
    inner_w = max(width - 2, 20)
    content_w = inner_w - 1

    def row2(label: str, value: str) -> str:
        return _metric(label, value, label_w=12, total_w=content_w, on=bool(on))

    head_f1 = row.get("per_head_macro_f1_observed") or {}
    head_ece = row.get("per_head_ece_calibrated") or {}
    head_bits: list[str] = []
    if isinstance(head_f1, dict) and head_f1:
        for h in sorted(head_f1)[:6]:
            head_bits.append(f"{h}={fmt_f1(head_f1.get(h))}")
    elif row.get("subclass_objective") is not None:
        head_bits.append(f"subclass_obj={fmt_f1(sub_obj)}")

    lines: list[str] = [ep_strip or f"parcel batch {epoch} sorted"]
    lines.append(row2("train", fmt_loss(float(train_loss) if train_loss is not None else None)))
    if loss_endpoint is not None:
        lines.append(row2("endpoint", fmt_loss(float(loss_endpoint))))
    lines.append(row2("val", fmt_loss(float(val_loss) if val_loss is not None else None)))
    if lr is not None:
        lines.append(row2("lr", fmt_lr(lr)))
    if win_acc is not None or doc_acc is not None or macro_f1 is not None:
        lines.append(
            f"acc {fmt_f1(win_acc) if win_acc is not None else '—'}"
            f"  ·  doc {fmt_f1(doc_acc) if doc_acc is not None else '—'}"
            f"  ·  F1 {fmt_f1(macro_f1)}"
        )
    if ece is not None or ece_cal is not None:
        lines.append(f"ece {fmt_f1(ece)}  ·  cal {fmt_f1(ece_cal)}")
    if sub_obj is not None:
        lines.append(row2("subclass_obj", fmt_f1(sub_obj)))
    if head_bits:
        lines.append("heads " + " ".join(head_bits)[: max(content_w - 6, 16)])
    flags: list[str] = []
    if selected is not None:
        flags.append("selected ★" if selected else "not selected")
    if gate is not None:
        flags.append("gate ok" if gate else "gate pending")
    if selection_epoch:
        flags.append(f"best e={selection_epoch}")
    if flags:
        lines.append(" · ".join(flags))
    if epoch_wall_s is not None:
        lines.append(f"wall {_fmt_duration(float(epoch_wall_s))}" + (f"  ·  remaining ~ {eta_rem}" if eta_rem else ""))
    elif eta_rem:
        lines.append(f"remaining ~ {eta_rem}")

    if not on:
        if show_owl:
            owl_lines = _mini_owl_block(on=False, blink=blink)
            top = _combine_owl_metrics(owl_lines, lines[: len(owl_lines)], inner_w=inner_w, on=False)
            return _box(title, [*top, *lines[len(owl_lines):]], on=on, width=width)
        return _box(title, lines, on=on, width=width)

    styled = [p["accent"](lines[0])]
    for ln in lines[1:]:
        if ln.startswith("train") or ln.startswith("val") or ln.startswith("endpoint") or ln.startswith("lr"):
            styled.append(p["mint"](ln))
        elif ln.startswith("acc") or "F1" in ln or ln.startswith("ece"):
            styled.append(p["cyan"](ln))
        else:
            styled.append(p["gold"](ln))
    if show_owl:
        owl_lines = _mini_owl_block(on=True, blink=blink)
        top = _combine_owl_metrics(owl_lines, styled[: len(owl_lines)], inner_w=inner_w, on=True)
        return _box(title, [*top, *styled[len(owl_lines):]], on=on, width=width)
    return _box(title, styled, on=on, width=width)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    try:
        s = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "—"
    h, rem = s // 3600, s % 3600
    m, sec = rem // 60, rem % 60
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def _fmt_mem(mb: Any) -> str:
    try:
        v = float(mb)
    except (TypeError, ValueError):
        return "—"
    if v >= 1024:
        return f"{v / 1024:.1f}G"
    return f"{v:.0f}M"


def read_last_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def read_last_two_jsonl(path: Path) -> tuple[dict | None, dict | None]:
    if not path.is_file():
        return None, None
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return None, None
    if len(rows) == 1:
        return None, rows[0]
    return rows[-2], rows[-1]


def parse_log_step_line(line: str) -> dict | None:
    m = re.search(
        r"step\s+(\d+)\s+loss\s+([\d.]+)\s+([\d.]+)\s+steps/s\s+"
        r"([\d.]+)\s+samples/s\s+ETA\s+(.+?)\s*$",
        line.strip(),
    )
    if not m:
        return None
    return {
        "event": "step",
        "step": int(m.group(1)),
        "loss": float(m.group(2)),
        "steps_per_sec": float(m.group(3)),
        "samples_per_sec": float(m.group(4)),
        "eta": m.group(5).strip(),
    }


def _active_stage(jsonl_row: dict | None, epoch_jsonl_row: dict | None) -> str:
    if jsonl_row and jsonl_row.get("event") == "epoch":
        return "VAL"
    if epoch_jsonl_row and epoch_jsonl_row.get("event") == "epoch":
        return "VAL"
    return "TRAIN"


def _trainer_pid(lock_line: str | None) -> int | None:
    if not lock_line:
        return None
    try:
        return int(lock_line.strip().split()[0])
    except (ValueError, IndexError):
        return None


def sample_resources(
    lock_line: str | None = None,
    trainer_lines: list[str] | None = None,
) -> dict[str, Any]:
    """Read-only GPU/CPU/RAM sample. Never signals the trainer."""
    sample: dict[str, Any] = {"available": False}
    pid = _trainer_pid(lock_line)
    if trainer_lines:
        for t in trainer_lines:
            try:
                cand = int(str(t).strip().split()[0])
                pid = cand
                break
            except (ValueError, IndexError):
                continue
    # GPU via nvidia-smi (read-only query).
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first = proc.stdout.strip().splitlines()[0].split(",")
            vals = [v.strip() for v in first]
            if len(vals) >= 7:
                sample.update(
                    {
                        "gpu_index": vals[0],
                        "gpu_name": vals[1],
                        "gpu_util": vals[2],
                        "gpu_mem_used": vals[3],
                        "gpu_mem_total": vals[4],
                        "gpu_temp": vals[5],
                        "gpu_power": vals[6],
                        "available": True,
                    }
                )
    except Exception:
        pass
    # Trainer CPU/RSS via ps (read-only).
    if pid:
        try:
            proc = subprocess.run(
                ["ps", "-o", "%cpu=,rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                bits = proc.stdout.strip().split()
                if len(bits) >= 2:
                    sample["cpu_pct"] = bits[0]
                    try:
                        rss_kb = float(bits[1])
                        sample["rss"] = f"{rss_kb / 1024:.0f}M"
                    except ValueError:
                        sample["rss"] = bits[1]
                    sample["pid"] = pid
        except Exception:
            pass
    # Host RAM + load (read-only procfs).
    try:
        mem_total = mem_avail = None
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) // 1024
                if mem_total is not None and mem_avail is not None:
                    break
        if mem_total is not None and mem_avail is not None:
            sample["host_mem_used"] = mem_total - mem_avail
            sample["host_mem_total"] = mem_total
    except Exception:
        pass
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            sample["load1"] = fh.read().strip().split()[0]
    except Exception:
        pass
    return sample


def render_hardware_panel(
    resources: dict[str, Any] | None,
    *,
    on: bool,
    width: int = _INNER_BOX_W,
    compact: bool = False,
) -> str:
    p = palette(on)
    if compact:
        if not resources or not resources.get("available"):
            line = "HW gpu: unavailable · trainer: idle"
            return _panel_row(line, on=on, width=width)
        gpu = f"gpu {resources.get('gpu_util', '—')}% {_fmt_mem(resources.get('gpu_mem_used'))}/{_fmt_mem(resources.get('gpu_mem_total'))}"
        cpu = f"cpu {resources.get('cpu_pct', '—')}%"
        ram = "ram —"
        if resources.get("host_mem_used") and resources.get("host_mem_total"):
            ram = f"ram {_fmt_mem(resources['host_mem_used'])}/{_fmt_mem(resources['host_mem_total'])}"
        line = f"HW {gpu} · {cpu} · {ram}"
        if on:
            return _panel_row(p["cyan"](line), on=False, sides=True, width=width)
        return _panel_row(line, on=on, width=width)
    lines: list[str] = []
    if not resources or not resources.get("available"):
        lines = ["gpu: unavailable", "trainer: idle"]
    else:
        gpu_name = str(resources.get("gpu_name", "gpu"))[:28]
        lines.append(f"gpu {resources.get('gpu_util', '—')}% · {gpu_name}")
        lines.append(
            f"mem {_fmt_mem(resources.get('gpu_mem_used'))}/{_fmt_mem(resources.get('gpu_mem_total'))}"
            f"  ·  {resources.get('gpu_temp', '—')}C  ·  {resources.get('gpu_power', '—')}W"
        )
        cpu = resources.get("cpu_pct", "—")
        rss = resources.get("rss", "—")
        lines.append(f"trainer cpu {cpu}%  ·  rss {rss}")
        if resources.get("host_mem_used") and resources.get("host_mem_total"):
            lines.append(
                f"host {_fmt_mem(resources['host_mem_used'])}/{_fmt_mem(resources['host_mem_total'])}"
                f"  ·  load {resources.get('load1', '—')}"
            )
        else:
            lines.append(f"load {resources.get('load1', '—')}")
    if on:
        styled = [p["cyan"](lines[0])] + [p["dim"](x) for x in lines[1:]]
        return _box("HARDWARE", styled, on=on, width=width)
    return _box("HARDWARE", lines, on=on, width=width)


def _side_by_side(left: str, right: str, *, gap: int = 2) -> str:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    n = max(len(left_lines), len(right_lines))
    left_w = max((_visible_len(x) for x in left_lines), default=0)
    out: list[str] = []
    for i in range(n):
        l = left_lines[i] if i < len(left_lines) else " " * left_w
        r = right_lines[i] if i < len(right_lines) else ""
        l = _pad_visible(l, left_w)
        out.append(l + " " * gap + r)
    return "\n".join(out)


def render_watch_snapshot(
    *,
    timestamp: str,
    lock_line: str | None,
    trainer_lines: list[str],
    log_tail: list[str],
    jsonl_row: dict | None,
    epoch_jsonl_row: dict | None = None,
    run_tag: str | None = None,
    on: bool | None = None,
    width: int | None = None,
    height: int | None = None,
    resources: dict[str, Any] | None = None,
    prev_step_row: dict | None = None,
    blink: bool = False,
    tick: int = 0,
) -> str:
    on = use_color() if on is None else on
    p = palette(on)
    cols, rows = _term_size()
    if width is not None:
        cols = width
    if height is not None:
        rows = height
    panel_w = _panel_width(cols)
    layout = _layout(cols, rows)
    compact = layout == "narrow"
    box_w = max(panel_w - 6, 40)
    if compact:
        box_w = max(panel_w - 4, 30)

    chunks: list[str] = []
    chunks.extend(render_header_banner(run_tag=run_tag, on=on, width=panel_w, blink=blink, compact=compact))
    chunks.append("")

    stage = _active_stage(jsonl_row, epoch_jsonl_row)
    chunks.append(render_status_bar(timestamp=timestamp, stage=stage, on=on, width=panel_w))

    meta: list[str] = []
    if lock_line:
        meta.append(f"lock: {lock_line.strip()}")
    else:
        meta.append("lock: (none)")
    if trainer_lines:
        meta.append("trainer: running ✉")
        meta.extend(f"  {t}" for t in trainer_lines)
    else:
        meta.append("trainer: (not running)")
    for m in meta:
        if on and not m.startswith("  "):
            chunks.append(_panel_row(p["dim"](m), on=False, sides=True, width=panel_w))
        else:
            chunks.append(_panel_row(m, on=on, sides=True, width=panel_w))

    sep = "╠" + DH * (panel_w - 2) + "╣"
    chunks.append(p["frame"](sep) if on else sep)

    metric_text = ""
    if jsonl_row:
        ev = jsonl_row.get("event", "step")
        if ev == "epoch":
            metric_text = render_epoch_event(jsonl_row, on=on, width=box_w, blink=blink)
        else:
            metric_text = render_step_event(
                jsonl_row, on=on, width=box_w, prev_row=prev_step_row, tick=tick, blink=blink
            )
    elif log_tail:
        for line in reversed(log_tail):
            parsed = parse_log_step_line(line)
            if parsed:
                metric_text = render_step_event(
                    parsed, on=on, width=box_w, prev_row=None, tick=tick, blink=blink
                )
                break
        else:
            hint = _panel_row(p["dim"]("log (recent):") if on else "log (recent):", on=False, sides=True, width=panel_w)
            chunks.append(hint)
            chunks.extend(f"  {ln.rstrip()}" for ln in log_tail[-3:])

    epoch_text = ""
    if epoch_jsonl_row and epoch_jsonl_row.get("event") == "epoch":
        if not jsonl_row or jsonl_row.get("event") != "epoch":
            epoch_text = render_epoch_event(epoch_jsonl_row, on=on, width=box_w, blink=blink)

    hw_compact = compact
    hw_width = box_w
    if layout == "wide":
        left_w = int((panel_w - 4) * 0.62)
        right_w = panel_w - 4 - left_w - 2
        hw_width = max(right_w, 30)
        metric_w = max(left_w, 40)
        if metric_text:
            # Re-render metrics at column width for a clean split.
            if jsonl_row and jsonl_row.get("event", "step") == "epoch":
                metric_text = render_epoch_event(jsonl_row, on=on, width=metric_w, blink=blink)
            elif jsonl_row:
                metric_text = render_step_event(
                    jsonl_row, on=on, width=metric_w, prev_row=prev_step_row, tick=tick, blink=blink
                )
        hw_text = render_hardware_panel(resources, on=on, width=hw_width, compact=False)
        if metric_text:
            chunks.append(_side_by_side(metric_text, hw_text))
        else:
            chunks.append(hw_text)
        if epoch_text:
            chunks.append(epoch_text)
    else:
        if metric_text:
            chunks.append(metric_text)
        if epoch_text:
            chunks.append(epoch_text)
        if hw_compact:
            chunks.append(render_hardware_panel(resources, on=on, width=panel_w, compact=True))
        else:
            chunks.append(render_hardware_panel(resources, on=on, width=box_w, compact=False))

    footer = f"{FOOTER_GREMLIN} {stage_gremlin(stage)} {mailroom_route_banner()} · {mailroom_pipeline_hint(stage)}"
    if on:
        chunks.append(_panel_row(p["dim"](footer), on=False, sides=True, width=panel_w))
    else:
        chunks.append(_panel_row(footer, on=on, sides=True, width=panel_w))

    return "\n".join(chunks) + "\n"


def follow_jsonl(path: Path, *, from_end: bool = True) -> None:
    """Tail -f style reader; pretty-prints new step/epoch rows."""
    on = use_color()
    pos = 0
    if from_end and path.is_file():
        pos = path.stat().st_size
    while True:
        if not path.is_file():
            time.sleep(0.5)
            continue
        with path.open(encoding="utf-8") as fh:
            fh.seek(pos)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = row.get("event", "step")
                if ev == "epoch":
                    print(render_epoch_event(row, on=on), flush=True)
                elif ev == "step":
                    print(render_step_event(row, on=on), flush=True)
            pos = fh.tell()
        time.sleep(1.0)


def _file_sig(path: Path | None) -> tuple[float, int] | None:
    if path is None or not path.is_file():
        return None
    try:
        st = path.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def follow_live(
    *,
    log_path: Path | None,
    jsonl_path: Path | None,
    epoch_path: Path | None,
    run_tag: str | None,
    lock_path: Path | None,
    on: bool,
    interval: float = 2.0,
) -> int:
    tick = 0
    jsonl_pos = jsonl_path.stat().st_size if jsonl_path and jsonl_path.is_file() else 0
    epoch_pos = epoch_path.stat().st_size if epoch_path and epoch_path.is_file() else 0
    log_pos = log_path.stat().st_size if log_path and log_path.is_file() else 0
    last_sig = (
        _file_sig(log_path),
        _file_sig(jsonl_path),
        _file_sig(epoch_path),
    )
    use_alt = bool(on)
    stop = False

    def _leave(_s: Any = None, _f: Any = None) -> None:
        nonlocal stop
        stop = True

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _leave)
    signal.signal(signal.SIGTERM, _leave)
    try:
        if use_alt:
            sys.stdout.write(ALT_ENTER + HIDE_CURSOR)
            sys.stdout.flush()
        while not stop:
            # Drain new jsonl/log lines (positions only drive redraws).
            if jsonl_path and jsonl_path.is_file():
                try:
                    size = jsonl_path.stat().st_size
                    if size < jsonl_pos:
                        jsonl_pos = 0
                    else:
                        jsonl_pos = size
                except OSError:
                    pass
            if epoch_path and epoch_path.is_file():
                try:
                    size = epoch_path.stat().st_size
                    if size < epoch_pos:
                        epoch_pos = 0
                    else:
                        epoch_pos = size
                except OSError:
                    pass
            if log_path and log_path.is_file():
                try:
                    size = log_path.stat().st_size
                    if size < log_pos:
                        log_pos = 0
                    else:
                        log_pos = size
                except OSError:
                    pass
            sig = (_file_sig(log_path), _file_sig(jsonl_path), _file_sig(epoch_path))
            changed = sig != last_sig
            last_sig = sig

            lock_line = None
            if lock_path and lock_path.is_file():
                try:
                    lock_line = lock_path.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    lock_line = None
            trainer_lines: list[str] = []
            try:
                proc = subprocess.run(
                    ["pgrep", "-f", r"train_modernbert\.py"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                for pid in proc.stdout.strip().split():
                    try:
                        ps = subprocess.run(
                            ["ps", "-o", "args=", "-p", pid],
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                        args = ps.stdout.strip()
                        if "training/train_modernbert.py" in args:
                            trainer_lines.append(f"{pid} {args}")
                    except Exception:
                        continue
            except Exception:
                pass
            log_tail: list[str] = []
            if log_path and log_path.is_file():
                try:
                    log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
                except OSError:
                    log_tail = []
            prev_row, row = (None, None)
            if jsonl_path:
                prev_row, row = read_last_two_jsonl(jsonl_path)
                if row is None and prev_row is not None:
                    row, prev_row = prev_row, None
            epoch_row = read_last_jsonl(epoch_path) if epoch_path else None
            resources = sample_resources(lock_line, trainer_lines)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            blink = (tick % 6 == 5)
            text = render_watch_snapshot(
                timestamp=ts,
                lock_line=lock_line,
                trainer_lines=trainer_lines,
                log_tail=log_tail,
                jsonl_row=row,
                epoch_jsonl_row=epoch_row,
                run_tag=run_tag or (lock_line.split()[1] if lock_line and len(lock_line.split()) > 1 else None),
                on=on,
                resources=resources,
                prev_step_row=prev_row,
                blink=blink,
                tick=tick,
            )
            if use_alt:
                sys.stdout.write(CURSOR_HOME + CLEAR_SCREEN + text)
                sys.stdout.flush()
            else:
                sys.stdout.write(text + "\n")
                sys.stdout.flush()
            # Sleep in small slices so Ctrl-C stays responsive; hardware
            # ticks every `interval` even without new log lines.
            waited = 0.0
            step = 0.2
            while waited < (interval if not changed else interval) and not stop:
                time.sleep(step)
                waited += step
            tick += 1
    finally:
        try:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
        except Exception:
            pass
        if use_alt:
            sys.stdout.write(SHOW_CURSOR + ALT_LEAVE)
            sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM Mailroom pixel-owl live TUI for M9a logs (read-only).")
    ap.add_argument("--jsonl", type=Path, help="train_steps.jsonl path")
    ap.add_argument("--epoch-jsonl", type=Path, help="epoch_metrics.jsonl path")
    ap.add_argument("--log", type=Path, help="Plain trainer .log (fallback parse)")
    ap.add_argument("--once", action="store_true", help="Print latest event and exit")
    ap.add_argument("--follow", action="store_true", help="Follow logs for new events")
    ap.add_argument("--plain", action="store_true", help="Disable ANSI even on TTY")
    ap.add_argument("--timestamp", help="Header timestamp for watch snapshot")
    ap.add_argument("--run-tag", default="", help="Run tag from lock file (display only)")
    ap.add_argument("--lock-line", default="", help="Contents of logs/.m9a-train.lock")
    ap.add_argument("--lock-path", type=Path, help="Path to logs/.m9a-train.lock for live follow")
    ap.add_argument("--width", type=int, help="Override terminal columns (tests/manual)")
    ap.add_argument("--height", type=int, help="Override terminal rows (tests/manual)")
    ap.add_argument("--interval", type=float, default=2.0, help="Live refresh seconds")
    ap.add_argument(
        "--trainer-line",
        action="append",
        default=[],
        help="One trainer ps line (repeatable)",
    )
    args = ap.parse_args(argv)

    on = use_color() and not args.plain

    if args.follow:
        watch_live = bool(args.log or args.epoch_jsonl or args.lock_line or args.lock_path or args.run_tag)
        if not watch_live:
            if not args.jsonl:
                print("--follow requires --jsonl", file=sys.stderr)
                return 2
            follow_jsonl(args.jsonl)
            return 0
        return follow_live(
            log_path=args.log,
            jsonl_path=args.jsonl,
            epoch_path=args.epoch_jsonl,
            run_tag=args.run_tag or None,
            lock_path=args.lock_path,
            on=on,
            interval=max(0.5, float(args.interval or 2.0)),
        )

    prev_row, row = (None, None)
    if args.jsonl:
        prev_row, row = read_last_two_jsonl(args.jsonl)
        if row is None and prev_row is not None:
            row, prev_row = prev_row, None
    epoch_row = read_last_jsonl(args.epoch_jsonl) if args.epoch_jsonl else None
    log_tail: list[str] = []
    if args.log and args.log.is_file():
        log_tail = args.log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]

    watch_mode = bool(args.timestamp or args.lock_line or args.trainer_line or args.run_tag)
    if watch_mode or row or log_tail or epoch_row:
        ts = args.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resources: dict[str, Any] | None = None
        if not args.plain:
            try:
                resources = sample_resources(args.lock_line or None, list(args.trainer_line))
            except Exception:
                resources = None
        text = render_watch_snapshot(
            timestamp=ts,
            lock_line=args.lock_line or None,
            trainer_lines=list(args.trainer_line),
            log_tail=log_tail,
            jsonl_row=row,
            epoch_jsonl_row=epoch_row,
            run_tag=args.run_tag or None,
            on=on,
            width=args.width,
            height=args.height,
            resources=resources,
            prev_step_row=prev_row,
        )
        sys.stdout.write(text)
        return 0

    if args.jsonl:
        print(f"(no events in {args.jsonl})", file=sys.stderr)
        return 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
