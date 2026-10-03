#!/usr/bin/env python3
"""LLM Mailroom retro digital TUI for M9a training monitors (stdout only; never mutates logs)."""
from __future__ import annotations

import argparse
import ast
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
# Hermes / amber arcade marquee (face + bevel + extrusion — not blue/teal).
AMBER_FACE = (255, 176, 0)        # #FFB000 face front
AMBER_HI = (255, 224, 138)        # #FFE08A top/left bevel highlight
AMBER = (245, 196, 69)            # #F5C445 mid gold (owl / accents)
AMBER_MID = (232, 163, 23)        # #E8A317
AMBER_EXTRUDE = (138, 90, 18)     # #8A5A12 dark bronze 3D drop
AMBER_SHADOW = (196, 132, 29)     # #C4841D warmer bronze fallback
AMBER_DEEP = (184, 134, 11)       # #B8860B deep shade
CREAM = (255, 236, 200)           # soft cream for DIGITAL MAILROOM subtitle
SNOW = (251, 252, 254)    # #fbfcfe
MUTED = (92, 113, 132)    # #5c7184 site muted labels
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
    """Mailroom frame/labels in blue-teal; Hermes amber wordmark; cream body (truecolor when TTY)."""
    return {
        "title": lambda t: _c(_rgb(BLUE), t, on=on),
        "brand": lambda t: _c(_rgb(CREAM), t, on=on),
        "snow": lambda t: _c(_rgb(SNOW), t, on=on),
        "cream": lambda t: _c(_rgb(CREAM), t, on=on),
        "accent": lambda t: _c(_rgb(CYAN), t, on=on),
        "teal": lambda t: _c(_rgb(TEAL), t, on=on),
        "cyan": lambda t: _c(_rgb(CYAN), t, on=on),
        "mint": lambda t: _c(_rgb(SKY), t, on=on),
        "gold": lambda t: _c(_rgb(GOLD), t, on=on),
        "amber": lambda t: _c(_rgb(AMBER_FACE), t, on=on),
        "stamp": lambda t: _c(_rgb(CYAN), t, on=on),
        "blue": lambda t: _c(_rgb(BLUE), t, on=on),
        "navy": lambda t: _c(_rgb(NAVY), t, on=on),
        "muted": lambda t: _c(_rgb(MUTED), t, on=on),
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


# ---------------------------------------------------------------- wordmark + stamps
# One symmetrical snowy owl (not a double-head). Hermes amber arcade marquee.
# Every owl row is exactly 7 cells so the mark stays rectangular.
OWL_FULL_OPEN: list[str] = [
    "  ___  ",
    " (o,o) ",
    " (  V) ",
    " /)  ) ",
    " ^^ ^^ ",
]
OWL_FULL_BLINK: list[str] = [
    "  ___  ",
    " (-,-) ",
    " (  V) ",
    " /)  ) ",
    " ^^ ^^ ",
]
OWL_MINI_OPEN: list[str] = [
    " (o,o) ",
    " (  V) ",
    " ^^ ^^ ",
]
OWL_MINI_BLINK: list[str] = [
    " (-,-) ",
    " (  V) ",
    " ^^ ^^ ",
]
OWL_FULL = OWL_FULL_OPEN
OWL_MINI = OWL_MINI_OPEN

# Face-only chunky glyphs (█ = solid). Depth is composited as a +1,+1 extrusion.
# Compact / stacked: 5-row x 5-col heavy arcade blocks. Every letter uses the
# SAME 2-column stroke width for vertical stems and horizontal bars so no
# glyph reads "thinner" than its neighbors.
_WM_FONT_3: dict[str, list[str]] = {
    "T": ["█████", "█████", " ███ ", " ███ ", " ███ "],
    "H": ["██ ██", "██ ██", "█████", "██ ██", "██ ██"],
    "E": ["█████", "███  ", "████ ", "███  ", "█████"],
    # Two 2-col stems: open peaks at the top, a thick two-row arch bridging
    # them, then open legs below -- reads distinctly from H's thin, single
    # mid-height crossbar.
    "M": ["██ ██", "█████", "█████", "██ ██", "██ ██"],
    "A": [" ███ ", "██ ██", "█████", "██ ██", "██ ██"],
    "I": ["█████", " ███ ", " ███ ", " ███ ", "█████"],
    "L": ["██   ", "██   ", "██   ", "██   ", "█████"],
    "R": ["████ ", "██ ██", "█████", "██ █ ", "██  █"],
    "O": ["█████", "██ ██", "██ ██", "██ ██", "█████"],
    " ": ["  ", "  ", "  ", "  ", "  "],
}

# Wide single-row (80+ cols beside owl): 5-row face, 6 cols for every letter
# (M matched to the same 6-col width as the rest; two peaks + early bridge,
# same 2-column stroke width discipline as _WM_FONT_3).
_WM_FONT_4: dict[str, list[str]] = {
    "T": ["██████", "██████", " ████ ", " ████ ", " ████ "],
    "H": ["██  ██", "██  ██", "██████", "██  ██", "██  ██"],
    "E": ["██████", "████  ", "█████ ", "████  ", "██████"],
    # Two outer stems + a two-row arch bridge form two peaks. Not an N diagonal.
    "M": ["██  ██", "██████", "██████", "██  ██", "██  ██"],
    "A": [" ████ ", "██  ██", "██████", "██  ██", "██  ██"],
    "I": ["██████", " ████ ", " ████ ", " ████ ", "██████"],
    "L": ["██    ", "██    ", "██    ", "██    ", "██████"],
    "R": ["█████ ", "██  ██", "█████ ", "██ ██ ", "██  ██"],
    "O": ["██████", "██  ██", "██  ██", "██  ██", "██████"],
    " ": ["  ", "  ", "  ", "  ", "  "],
}


def _assert_uniform_font(font: dict[str, list[str]]) -> None:
    """Raise if glyphs in a font dict don't all share one row count and one
    column count. Callers should pass only the "real" letters (excluding the
    variable-width space glyph) so a regression (like a mismatched-width
    letter) fails loudly instead of silently shifting kerning."""
    row_counts = {len(rows) for rows in font.values()}
    assert len(row_counts) == 1, f"font glyphs have mismatched row counts: {row_counts}"
    col_counts = {len(row) for rows in font.values() for row in rows}
    assert len(col_counts) == 1, f"font glyphs have mismatched col counts: {col_counts}"


_assert_uniform_font({ch: g for ch, g in _WM_FONT_3.items() if ch != " "})
_assert_uniform_font({ch: g for ch, g in _WM_FONT_4.items() if ch != " "})

# Cell roles after extrusion composite (not painted as flat mid-tone fills).
_ROLE_FACE = "F"
_ROLE_HIGHLIGHT = "H"
_ROLE_EXTRUDE = "X"
_ROLE_EMPTY = " "

# Kept for signature compatibility; UI no longer prints faces.
STAGE_GREMLINS: dict[str, str] = {
    "TRAIN": "",
    "VAL": "",
    "TEST": "",
    "GATE": "",
    "HUB": "",
}
FOOTER_GREMLIN = ""


def stage_gremlin(stage: str) -> str:
    return STAGE_GREMLINS.get(stage, "")


def render_owl_pixels(grid: list[str], *, on: bool, blink: bool = False) -> list[str]:
    """Colorize one snowy owl: snow body, amber eyes/beak."""
    del blink  # grid already open/blink; signature kept
    out: list[str] = []
    eye_chars = set("oO-")
    beak_chars = set("Vv")  # feet ^^ stay snow; only beak is amber
    for line in grid:
        if not on:
            out.append(line)
            continue
        parts: list[str] = []
        for ch in line:
            if ch in eye_chars or ch in beak_chars or ch == ",":
                parts.append(_c(_rgb(AMBER_FACE), ch, on=True))
            elif ch == " ":
                parts.append(ch)
            else:
                parts.append(_c(_rgb(SNOW), ch, on=True))
        out.append("".join(parts))
    return out


def render_owl(*, mini: bool = False, on: bool, blink: bool = False) -> list[str]:
    if mini:
        grid = OWL_MINI_BLINK if blink else OWL_MINI_OPEN
    else:
        grid = OWL_FULL_BLINK if blink else OWL_FULL_OPEN
    return render_owl_pixels(grid, on=on, blink=blink)


def owl_width(*, mini: bool = False) -> int:
    grid = OWL_MINI if mini else OWL_FULL
    return max((len(r) for r in grid), default=0)


def owl_emoticon(*, blink: bool = False, on: bool = False) -> str:
    """One consistent small monitor face: open (o,o) or blink (-,-)."""
    face = "(-,-)" if blink else "(o,o)"
    if not on:
        return face
    parts: list[str] = []
    eye_chars = set("oO-")
    for ch in face:
        if ch in eye_chars or ch == ",":
            parts.append(_c(_rgb(AMBER_FACE), ch, on=True))
        else:
            parts.append(_c(_rgb(SNOW), ch, on=True))
    return "".join(parts)


def _loading_status_line(
    micro_done: int | None,
    micro_planned: int | None,
    *,
    blink: bool = False,
    on: bool = False,
) -> str:
    """Short tray status under the micro bar — one owl, no parade."""
    face = owl_emoticon(blink=blink, on=on)
    in_progress = (
        micro_done is not None
        and micro_planned is not None
        and int(micro_planned) > 0
        and int(micro_done) < int(micro_planned)
    )
    if (
        micro_done is not None
        and micro_planned is not None
        and int(micro_planned) > 0
        and int(micro_done) >= int(micro_planned)
    ):
        msg = "held-out test eval…"
    elif in_progress:
        msg = "sorting…"
    else:
        msg = "waiting on the next tray"
    if on:
        p = palette(on)
        return face + " " + p["dim"](msg)
    return f"{face} {msg}"


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


def _normalize_font(font: dict[str, list[str]]) -> dict[str, list[str]]:
    """Pad every glyph row so a letter is a strict rectangle."""
    out: dict[str, list[str]] = {}
    for ch, rows in font.items():
        width = max(len(r) for r in rows)
        out[ch] = [r.ljust(width) for r in rows]
    return out


def _face_bitmap(text: str, font: dict[str, list[str]]) -> list[str]:
    """Join face-only glyphs into equal-width rows (█ = solid, space = empty)."""
    font = _normalize_font(font)
    height = len(next(iter(font.values())))
    fallback = font.get(" ", [" "] * height)
    rows: list[list[str]] = [[] for _ in range(height)]
    prev_letter = False
    for ch in text:
        glyph = font.get(ch, fallback)
        if len(glyph) != height:
            gw = max(len(r) for r in glyph)
            glyph = [r.ljust(gw) for r in glyph]
            glyph = (glyph + [" " * gw] * height)[:height]
        if prev_letter and ch != " ":
            for i in range(height):
                rows[i].append(" ")
        for i in range(height):
            rows[i].append(glyph[i])
        prev_letter = ch != " "
    joined = ["".join(parts) for parts in rows]
    width = max((len(r) for r in joined), default=0)
    return [r.ljust(width) for r in joined]


def _extrude_roles(face_rows: list[str]) -> list[str]:
    """Arcade cabinet extrusion: bronze drop at +1,+1, face on top, top-edge bevel."""
    if not face_rows:
        return []
    fh = len(face_rows)
    fw = len(face_rows[0])
    ch, cw = fh + 1, fw + 1
    roles = [[_ROLE_EMPTY] * cw for _ in range(ch)]

    def solid(r: int, c: int) -> bool:
        return 0 <= r < fh and 0 <= c < fw and face_rows[r][c] not in (" ", "")

    # Shadow layer: full letter shape shifted down-right (dark bronze depth).
    for r in range(fh):
        for c in range(fw):
            if solid(r, c):
                roles[r + 1][c + 1] = _ROLE_EXTRUDE

    # Face layer overwrites; top/left stroke edges get the light bevel.
    for r in range(fh):
        for c in range(fw):
            if not solid(r, c):
                continue
            if not solid(r - 1, c) or not solid(r, c - 1):
                roles[r][c] = _ROLE_HIGHLIGHT
            else:
                roles[r][c] = _ROLE_FACE
    return ["".join(row) for row in roles]


def _paint_extrude_cell(role: str, *, on: bool) -> str:
    """Paint one extruded marquee cell. Plain keeps ▓ depth without color."""
    if role == _ROLE_EMPTY or role == " ":
        return " "
    if not on:
        # Depth without color: solid face, darker shade for the drop.
        if role == _ROLE_EXTRUDE:
            return "▓"
        return "█"
    if role == _ROLE_EXTRUDE:
        # Dark bronze drop — must not equal face amber (#FFB000).
        return _c(_rgb(AMBER_EXTRUDE), "█", on=True)
    if role == _ROLE_HIGHLIGHT:
        # Top bevel: light gold ▀ over amber face (truecolor fg/bg).
        return f"\033[{_rgb(AMBER_HI)};{_rgb_bg(AMBER_FACE)}m▀\033[0m"
    return _c(_rgb(AMBER_FACE), "█", on=True)


def _compose_wordmark(text: str, font: dict[str, list[str]], *, on: bool) -> list[str]:
    """Arcade marquee: face + highlight bevel over +1,+1 bronze extrusion."""
    face = _face_bitmap(text, font)
    roles = _extrude_roles(face)
    painted = [
        "".join(_paint_extrude_cell(ch, on=on) for ch in row)
        for row in roles
    ]
    width = max((_visible_len(r) for r in painted), default=0)
    return [_pad_visible(r, width) for r in painted]


def _stack_centered(blocks: list[list[str]]) -> list[str]:
    width = max((_visible_len(line) for block in blocks for line in block), default=0)
    out: list[str] = []
    for block in blocks:
        for line in block:
            w = _visible_len(line)
            left = max(0, (width - w) // 2)
            out.append(_pad_visible(" " * left + line, width))
    return out


def _wordmark_variants(*, on: bool) -> tuple[list[str], list[str], list[str]]:
    """Return (shaded wide row, solid row, THE/MAILROOM stack)."""
    wide = _compose_wordmark("THE MAILROOM", _WM_FONT_4, on=on)
    row = _compose_wordmark("THE MAILROOM", _WM_FONT_3, on=on)
    stacked = _stack_centered(
        [
            _compose_wordmark("THE", _WM_FONT_3, on=on),
            _compose_wordmark("MAILROOM", _WM_FONT_3, on=on),
        ]
    )
    return wide, row, stacked


def _pick_mark(budget: int, *, on: bool, prefer_stack: bool) -> list[str]:
    wide, row, stacked = _wordmark_variants(on=on)
    if prefer_stack:
        if all(_visible_len(x) <= budget for x in stacked):
            return stacked
        if all(_visible_len(x) <= budget for x in row):
            return row
        return stacked
    if all(_visible_len(x) <= budget for x in wide):
        return wide
    if all(_visible_len(x) <= budget for x in row):
        return row
    if all(_visible_len(x) <= budget for x in stacked):
        return stacked
    return row


def _join_owl_wordmark(
    owl_lines: list[str],
    mark_lines: list[str],
    *,
    budget: int,
    stacked: bool,
) -> list[str]:
    """Owl left of wordmark when room; owl above wordmark on narrow."""
    if stacked:
        return _stack_centered([owl_lines, mark_lines])
    gap = 2
    owl_w = max((_visible_len(x) for x in owl_lines), default=0)
    mark_w = max((_visible_len(x) for x in mark_lines), default=0)
    if owl_w + gap + mark_w > budget:
        return _stack_centered([owl_lines, mark_lines])
    height = max(len(owl_lines), len(mark_lines))
    owl_pad_top = max(0, (height - len(owl_lines)) // 2)
    mark_pad_top = max(0, (height - len(mark_lines)) // 2)
    out: list[str] = []
    for i in range(height):
        oi = i - owl_pad_top
        mi = i - mark_pad_top
        left = owl_lines[oi] if 0 <= oi < len(owl_lines) else " " * owl_w
        right = mark_lines[mi] if 0 <= mi < len(mark_lines) else " " * mark_w
        left = _pad_visible(left, owl_w)
        right = _pad_visible(right, mark_w)
        out.append(left + " " * gap + right)
    width = max((_visible_len(r) for r in out), default=0)
    return [_pad_visible(r, width) for r in out]


def _wordmark_lines(*, inner: int, on: bool, compact: bool, blink: bool = False) -> list[str]:
    """Header hero: one owl + amber THE MAILROOM (owl above under ~70 cols)."""
    budget = max(inner - 2, 20)
    # Narrow / compact: mini owl above stacked THE / MAILROOM.
    if compact or budget < 70:
        owl = render_owl(mini=True, on=on, blink=blink)
        mark = _pick_mark(budget, on=on, prefer_stack=True)
        return _join_owl_wordmark(owl, mark, budget=budget, stacked=True)
    # Wide: full owl to the left of the largest mark that fits beside it.
    owl = render_owl(mini=False, on=on, blink=blink)
    gap = 2
    mark_budget = max(budget - owl_width(mini=False) - gap, 20)
    mark = _pick_mark(mark_budget, on=on, prefer_stack=False)
    joined = _join_owl_wordmark(owl, mark, budget=budget, stacked=False)
    if all(_visible_len(x) <= budget for x in joined):
        return joined
    # Fallback if side-by-side still overflows.
    mark = _pick_mark(budget, on=on, prefer_stack=True)
    return _join_owl_wordmark(owl, mark, budget=budget, stacked=True)


def _header_subtitle_lines(
    *,
    run_tag: str | None,
    inner: int,
    on: bool,
) -> list[str]:
    p = palette(on)
    tag = ""
    if run_tag:
        tag = run_tag if len(run_tag) <= 28 else run_tag[:27] + "…"
    brand = "DIGITAL MAILROOM"
    path = "ModernBERT fast-path"
    route = mailroom_route_banner()
    budget = inner - 2
    if tag and _visible_len(f"{brand}  ·  {path}  ·  {tag}") <= budget:
        if on:
            line = p["brand"](brand) + p["dim"]("  ·  ") + p["snow"](path) + p["dim"]("  ·  ") + p["gold"](tag)
        else:
            line = f"{brand}  ·  {path}  ·  {tag}"
        return [line, route if not on else p["teal"](route)]
    if on:
        top = p["brand"](brand) + p["dim"]("  ·  ") + p["snow"](path if _visible_len(f"{brand}  ·  {path}") <= budget else "ModernBERT")
        extra = [p["gold"](tag)] if tag else []
        return [top, *extra, p["teal"](route)]
    top = f"{brand}  ·  {path}"
    if _visible_len(top) > budget:
        top = brand
    extra = [tag] if tag else []
    return [top, *extra, route]


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
    mark = _wordmark_lines(inner=inner, on=on, compact=compact, blink=blink)
    subs = _header_subtitle_lines(run_tag=run_tag, inner=inner, on=on)

    top = DTL + DH * inner + DTR
    bot = DBL + DH * inner + DBR
    rows: list[str] = [p["frame"](top) if on else top]
    for ol in mark:
        rows.append(_centered_row(ol, width=w, on=on))
    for sub in subs:
        rows.append(_panel_row(sub, on=False, sides=True, width=w))
    rows.append(p["frame"](bot) if on else bot)
    return rows


def render_status_bar(
    *,
    timestamp: str,
    stage: str,
    on: bool,
    width: int = PANEL_W,
    blink: bool = False,
) -> str:
    p = palette(on)
    face = owl_emoticon(blink=blink, on=False)
    left = f"watch @ {timestamp}"
    right = f"{face}  stage ▸{stage}◂"
    gap = width - 4 - len(left) - len(right)
    if gap < 1:
        gap = 1
    plain = left + " " * gap + right
    if not on:
        return _panel_row(plain, on=on, width=width)
    face_c = owl_emoticon(blink=blink, on=True)
    styled_right = face_c + "  " + p["dim"]("stage ") + p["stamp"](f"▸{stage}◂")
    gap2 = width - 4 - _visible_len(left) - _visible_len(styled_right)
    if gap2 < 1:
        gap2 = 1
    styled = p["dim"](left) + " " * gap2 + styled_right
    return _panel_row(styled, on=False, sides=True, width=width)


def _box(title: str, lines: list[str], *, width: int = _INNER_BOX_W, on: bool) -> str:
    p = palette(on)
    inner_w = max(width - 2, 20)
    title_stripped = title.strip()
    title_plain = _ANSI_RE.sub("", title_stripped)
    if len(title_plain) > inner_w - 2:
        title_plain = title_plain[: inner_w - 3] + "…"
        title_stripped = title_plain
    title_vis = _visible_len(title_stripped)
    pad = inner_w - title_vis - 2
    if on and "\033[" not in title_stripped:
        mid = " " + p["title"](title_stripped) + " "
    else:
        mid = " " + title_stripped + " "
    top = TL + mid + H * max(0, pad) + TR
    body: list[str] = []
    for line in lines:
        trunc = _truncate_visible(line, inner_w - 1)
        padded = _pad_visible(trunc, inner_w - 1)
        body.append(V + " " + padded + V)
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
    """Quiet milestone mark — no faces."""
    if step <= 0:
        return ""
    if step == 1 or step % 500 == 0 or step % 100 == 0:
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
    show_owl: bool = False,
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
    face = owl_emoticon(blink=blink, on=on)
    title = f"{face} step {step}{spark} → SORTER · doc_type slot"
    if not ep_label:
        ep_label = "routing windows → plurality vote"

    inner_w = max(width - 2, 20)
    bar_w = 12 if width < 52 else (30 if width > 70 else 22)
    bar = progress_bar(micro_done, micro_planned, width=bar_w, tick=tick, on=bool(on))
    load_ln = _loading_status_line(micro_done, micro_planned, blink=blink, on=bool(on))
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
            lines_plain = [*top]
            if bar:
                lines_plain.append(bar)
            lines_plain.append(load_ln)
            return _box(title, lines_plain, on=on, width=width)
        lines_plain = [*right_lines]
        if bar:
            lines_plain.append(bar)
        lines_plain.append(load_ln)
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
        top.append(load_ln)
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
    lines_col.append(load_ln)
    return _box(title, lines_col, on=on, width=width)


def render_epoch_event(
    row: dict,
    *,
    on: bool | None = None,
    width: int = _INNER_BOX_W,
    blink: bool = False,
    show_owl: bool = False,
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
    title = f"{owl_emoticon(blink=blink, on=on)} epoch {epoch} → TRAY · subclass labels on parcels"
    inner_w = max(width - 2, 20)
    content_w = inner_w - 1

    def row2(label: str, value: str) -> str:
        return _metric(label, value, label_w=12, total_w=content_w, on=bool(on))

    head_f1 = row.get("per_head_macro_f1_observed") or {}
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


def render_test_step_event(
    row: dict,
    *,
    on: bool | None = None,
    width: int = _INNER_BOX_W,
    prev_row: dict | None = None,
    tick: int = 0,
    blink: bool = False,
) -> str:
    on = use_color() if on is None else on
    p = palette(on)
    done = int(row.get("doc_done") or 0)
    planned = int(row.get("doc_planned") or 0)
    dt_acc = row.get("doc_type_acc")
    sc_acc = row.get("subclass_acc_conditional")
    wall_s = row.get("wall_s")
    face = owl_emoticon(blink=blink, on=on)
    title = f"{face} test doc {done}/{planned} → GATE · held-out @ 8192"
    bar_w = 12 if width < 52 else (30 if width > 70 else 22)
    bar = progress_bar(done, planned, width=bar_w, tick=tick, on=bool(on))
    prev_dt = (prev_row or {}).get("doc_type_acc")
    delta = ""
    if prev_dt is not None and dt_acc is not None:
        try:
            d = float(dt_acc) - float(prev_dt)
            if abs(d) >= 1e-6:
                delta = f" (Δ {d:+.4f})"
        except (TypeError, ValueError):
            pass
    lines = [
        f"held-out docs [{done}/{planned}]",
        f"doc_type acc {fmt_f1(dt_acc)}{delta}",
    ]
    if sc_acc is not None:
        lines.append(f"subclass acc (cond) {fmt_f1(sc_acc)}")
    if row.get("doc_type_correct") is not None and planned:
        lines.append(
            f"correct {row.get('doc_type_correct')}/{planned}"
            f"  ·  subclass {row.get('subclass_correct', '—')}/"
            f"{row.get('subclass_scorable', '—')}"
        )
    if wall_s is not None:
        lines.append(f"test wall {_fmt_duration(float(wall_s))}")
    if bar:
        lines.append(bar)
    load = f"{face} scoring held-out docs…"
    if on:
        styled = [
            p["accent"](lines[0]),
            p["mint"](lines[1]),
            *([p["cyan"](lines[2])] if len(lines) > 2 and sc_acc is not None else []),
        ]
        rest = lines[3:] if sc_acc is not None else lines[2:]
        styled.extend(p["dim"](x) for x in rest)
        styled.append(p["dim"](load))
        return _box(title, styled, on=on, width=width)
    lines.append(load)
    return _box(title, lines, on=on, width=width)


def render_test_metrics_event(
    row: dict,
    *,
    on: bool | None = None,
    width: int = _INNER_BOX_W,
    blink: bool = False,
) -> str:
    on = use_color() if on is None else on
    p = palette(on)
    face = owl_emoticon(blink=blink, on=on)
    n = row.get("n_docs") or row.get("doc_planned")
    title = f"{face} TEST complete → GATE · {n or '—'} held-out docs"
    lines = [
        f"doc_type acc {fmt_f1(row.get('doc_type_acc'))}",
        f"subclass acc (cond) {fmt_f1(row.get('subclass_acc_conditional'))}",
    ]
    if row.get("doc_type_correct") is not None and n:
        lines.append(
            f"doc_type {row.get('doc_type_correct')}/{n}"
            f"  ·  subclass {row.get('subclass_correct', '—')}/"
            f"{row.get('subclass_scorable', '—')}"
        )
    if row.get("subclass_unscorable"):
        lines.append(f"unscorable subclass rows: {row.get('subclass_unscorable')}")
    if row.get("wall_s") is not None:
        lines.append(f"test wall {_fmt_duration(float(row['wall_s']))}")
    if on:
        styled = [p["gold"](lines[0]), p["mint"](lines[1]), *[p["dim"](x) for x in lines[2:]]]
        return _box(title, styled, on=on, width=width)
    return _box(title, lines, on=on, width=width)


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


def parse_log_test_step_line(line: str) -> dict | None:
    m = re.search(
        r"test\s+doc\s+(\d+)/(\d+)\s+doc_type_acc\s+([\d.]+)"
        r"(?:\s+subclass_cond\s+([\d.]+))?\s*$",
        line.strip(),
    )
    if not m:
        return None
    row: dict[str, Any] = {
        "event": "test_step",
        "doc_done": int(m.group(1)),
        "doc_planned": int(m.group(2)),
        "doc_type_acc": float(m.group(3)),
    }
    if m.group(4) is not None:
        row["subclass_acc_conditional"] = float(m.group(4))
    return row


def parse_log_test_metrics_line(line: str) -> dict | None:
    m = re.search(r"test metrics:\s*(\{.*\})\s*$", line.strip())
    if not m:
        return None
    try:
        payload = ast.literal_eval(m.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return {"event": "test_complete", **payload}


def test_steps_path_for_watch(
    run_tag: str | None,
    jsonl_path: Path | None,
    log_path: Path | None,
) -> Path | None:
    if jsonl_path and jsonl_path.is_file():
        if jsonl_path.name == "train_steps.jsonl" and jsonl_path.parent.name == "latest":
            candidate = jsonl_path.parent / "test_steps.jsonl"
            if candidate.is_file():
                return candidate
    tag = run_tag
    if not tag and log_path and log_path.suffix == ".log":
        tag = log_path.stem
    if not tag:
        return None
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        candidate = base / "data/modernbert_training/runs" / tag / "latest/test_steps.jsonl"
        if candidate.is_file():
            return candidate
    return None


def _resolve_test_row(
    test_jsonl_row: dict | None,
    log_tail: list[str],
    summary: dict[str, Any] | None,
) -> dict | None:
    if test_jsonl_row:
        return test_jsonl_row
    for line in reversed(log_tail):
        parsed = parse_log_test_step_line(line)
        if parsed:
            return parsed
        parsed = parse_log_test_metrics_line(line)
        if parsed:
            return parsed
    if summary:
        tm = summary.get("test_metrics")
        if isinstance(tm, dict) and tm.get("n_docs") is not None:
            return {"event": "test_complete", **tm}
    return None


def _load_summary_dict(summary_path: Path | None) -> dict[str, Any] | None:
    if not summary_path or not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _eval_test_enabled(summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    hp = summary.get("hyperparameters")
    if isinstance(hp, dict) and hp.get("eval_test"):
        return True
    return bool(summary.get("eval_test"))


def _test_eval_pending(summary: dict[str, Any] | None) -> bool:
    if not _eval_test_enabled(summary):
        return False
    tm = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
    return tm.get("n_docs") is None


def _training_epochs_done(summary: dict[str, Any] | None, jsonl_row: dict | None) -> bool:
    if not summary:
        return False
    planned = _planned_epoch_count(summary, jsonl_row)
    er = int(summary.get("epochs_run") or 0)
    return bool(planned and er >= planned)


def _active_stage(
    jsonl_row: dict | None,
    epoch_jsonl_row: dict | None,
    *,
    summary: dict[str, Any] | None = None,
    trainer_lines: list[str] | None = None,
) -> str:
    if trainer_lines and summary and _test_eval_pending(summary) and _training_epochs_done(summary, jsonl_row):
        return "TEST"
    if jsonl_row and jsonl_row.get("event") == "epoch":
        return "VAL"
    if epoch_jsonl_row and epoch_jsonl_row.get("event") == "epoch":
        return "VAL"
    return "TRAIN"


def _filter_trainers_for_run(trainer_lines: list[str], run_tag: str | None) -> list[str]:
    if not run_tag:
        return trainer_lines
    needle = f"runs/{run_tag}/"
    return [ln for ln in trainer_lines if needle in ln or f"/{run_tag}/" in ln]


def summary_path_for_watch(
    run_tag: str | None,
    jsonl_path: Path | None,
    log_path: Path | None,
) -> Path | None:
    if jsonl_path and jsonl_path.is_file():
        if jsonl_path.name == "train_steps.jsonl" and jsonl_path.parent.name == "latest":
            candidate = jsonl_path.parent / "summary.json"
            if candidate.is_file():
                return candidate
    tag = run_tag
    if not tag and log_path and log_path.suffix == ".log":
        tag = log_path.stem
    if not tag:
        return None
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        candidate = base / "data/modernbert_training/runs" / tag / "latest/summary.json"
        if candidate.is_file():
            return candidate
    return None


def eval_report_path_for_watch(run_tag: str | None) -> Path | None:
    if not run_tag:
        return None
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        candidate = base / "reports" / f"eval_{run_tag}.json"
        if candidate.is_file():
            return candidate
    return None


def _planned_epoch_count(summary: dict[str, Any], jsonl_row: dict | None) -> int | None:
    for key in ("epochs_requested", "num_epochs", "max_epochs", "n_epochs"):
        val = summary.get(key)
        if val is not None:
            try:
                n = int(val)
                return n if n > 0 else None
            except (TypeError, ValueError):
                pass
    hp = summary.get("hyperparameters")
    if isinstance(hp, dict) and hp.get("epochs") is not None:
        try:
            n = int(hp["epochs"])
            return n if n > 0 else None
        except (TypeError, ValueError):
            pass
    epochs = summary.get("epochs")
    if isinstance(epochs, list):
        if epochs:
            return len(epochs)
        # Fall through — some summaries use epochs=[] until first boundary.
    elif isinstance(epochs, int) and epochs > 0:
        return epochs
    if jsonl_row:
        try:
            n = int(jsonl_row.get("epochs") or 0)
            return n if n > 0 else None
        except (TypeError, ValueError):
            pass
    return None


def job_completed_summary(
    *,
    run_tag: str | None,
    trainer_lines: list[str],
    jsonl_row: dict | None,
    log_tail: list[str],
    summary_path: Path | None,
) -> dict[str, Any] | None:
    """True when this run's trainer has exited and the full train+test pipeline finished."""
    if trainer_lines:
        return None
    summary = _load_summary_dict(summary_path)
    log_done = bool(log_tail and any("run summary:" in ln for ln in log_tail))

    def _ready(summary_obj: dict[str, Any] | None) -> bool:
        if not summary_obj:
            return log_done
        if _test_eval_pending(summary_obj):
            return log_done
        if _training_epochs_done(summary_obj, jsonl_row):
            return True
        return log_done

    if summary and _ready(summary):
        return summary
    if log_done:
        return summary or {}
    return None


def _last_epoch_record(summary: dict[str, Any]) -> dict[str, Any]:
    epochs = summary.get("epochs")
    if isinstance(epochs, list) and epochs:
        last = epochs[-1]
        return last if isinstance(last, dict) else {}
    return {}


def _training_wall_s(summary: dict[str, Any]) -> float | None:
    for key in ("training_wall_s", "wall_s"):
        val = summary.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _load_eval_report(run_tag: str | None) -> dict[str, Any] | None:
    path = eval_report_path_for_watch(run_tag)
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def job_completed_banner_lines(
    summary: dict[str, Any],
    *,
    run_tag: str | None,
) -> list[str]:
    """Plain-text lines for the completion panel (summary.json + optional eval report)."""
    tag = run_tag or summary.get("run_id") or "run"
    hp = summary.get("hyperparameters") if isinstance(summary.get("hyperparameters"), dict) else {}
    lam = hp.get("loss_lambda_dt")
    er = int(summary.get("epochs_run") or 0)
    planned = _planned_epoch_count(summary, None) or er or None

    lines: list[str] = ["JOB COMPLETED", f"run: {tag}"]
    config_bits: list[str] = []
    if lam is not None:
        try:
            config_bits.append(f"λ_dt={float(lam):g}")
        except (TypeError, ValueError):
            pass
    if planned:
        config_bits.append(f"{er}/{planned} epochs")
    if config_bits:
        lines.append(" · ".join(config_bits))

    wall = _training_wall_s(summary)
    if wall is not None:
        hrs = wall / 3600.0
        lines.append(f"train wall: {wall:.0f}s ({hrs:.2f}h)")

    ep = _last_epoch_record(summary)
    if ep:
        f1 = ep.get("doc_type_macro_f1_observed", ep.get("doc_type_macro_f1"))
        ece = ep.get("doc_type_ece_calibrated")
        gate = ep.get("gate_met")
        sub_obj = ep.get("subclass_objective")
        val_bits: list[str] = []
        if f1 is not None:
            try:
                val_bits.append(f"val doc_type F1 {float(f1):.4f}")
            except (TypeError, ValueError):
                pass
        if ece is not None:
            try:
                val_bits.append(f"cal ECE {float(ece):.4f}")
            except (TypeError, ValueError):
                pass
        if gate is not None:
            val_bits.append("selection gate PASS" if gate else "selection gate FAIL")
        if val_bits:
            lines.append(" · ".join(val_bits))
        if sub_obj is not None:
            try:
                lines.append(f"subclass objective: {float(sub_obj):.3f}")
            except (TypeError, ValueError):
                pass
        head_f1: list[str] = []
        for head in (
            "insurance_claim",
            "contract",
            "correspondence",
            "corporate_record",
            "merger_agreement",
        ):
            key = f"{head}_macro_f1_observed"
            if key not in ep and f"{head}_macro_f1" in ep:
                key = f"{head}_macro_f1"
            if key in ep:
                try:
                    head_f1.append(f"{head} F1 {float(ep[key]):.3f}")
                except (TypeError, ValueError):
                    continue
        if head_f1:
            lines.append("heads: " + " · ".join(head_f1[:3]))
            if len(head_f1) > 3:
                lines.append("       " + " · ".join(head_f1[3:]))

    tm = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
    dt = tm.get("doc_type_acc")
    sc = tm.get("subclass_acc_conditional")
    n = tm.get("n_docs")
    test_bits: list[str] = []
    if dt is not None:
        try:
            test_bits.append(f"test doc_type acc {float(dt):.4f}")
        except (TypeError, ValueError):
            pass
    if sc is not None:
        try:
            test_bits.append(f"subclass (cond) {float(sc):.4f}")
        except (TypeError, ValueError):
            pass
    if n is not None:
        test_bits.append(f"n={n}")
    if test_bits:
        lines.append(" · ".join(test_bits))

    eval_report = _load_eval_report(run_tag or tag)
    if eval_report:
        ev_bits: list[str] = []
        acc = eval_report.get("doc_type_accuracy")
        if acc is not None:
            try:
                ev_bits.append(f"eval harness acc {float(acc):.4f}")
            except (TypeError, ValueError):
                pass
        fpr = eval_report.get("fast_path_rate")
        if fpr is not None:
            try:
                ev_bits.append(f"fast_path {100.0 * float(fpr):.1f}%")
            except (TypeError, ValueError):
                pass
        ood = eval_report.get("ood")
        if isinstance(ood, dict) and ood.get("rate") is not None:
            try:
                ev_bits.append(f"ood {100.0 * float(ood['rate']):.1f}%")
            except (TypeError, ValueError):
                pass
        if ev_bits:
            lines.append(" · ".join(ev_bits))

    sel = summary.get("checkpoint_selection")
    gate_fail = ep.get("gate_met") is False or (
        isinstance(sel, dict) and sel.get("gate_met") is False
    )
    if gate_fail:
        lines.append("artifact: UNCALIBRATED (no epoch met doc_type ECE ≤ 0.05)")

    if eval_report:
        post = "complete_run done · eval on disk"
    else:
        post = "complete_run.sh"
    lines.append(f"trainer stopped · post-train: {post}")
    return lines


def render_test_eval_pending_panel(
    summary: dict[str, Any] | None,
    *,
    on: bool,
    width: int,
    blink: bool = False,
) -> str:
    p = palette(on)
    face = owl_emoticon(blink=blink, on=on)
    n = 323
    tm = summary.get("test_metrics") if summary and isinstance(summary.get("test_metrics"), dict) else {}
    if tm.get("n_docs") is not None:
        try:
            n = int(tm["n_docs"])
        except (TypeError, ValueError):
            pass
    lines = [
        "training epochs finished",
        f"held-out test eval running ({n} docs @ 8192 tok)",
        "progress bar may sit at 99% until test completes",
    ]
    title = f"{face} TEST → GATE"
    if on:
        styled = [p["mint"](lines[0]), p["cream"](lines[1]), p["dim"](lines[2])]
        return _box(title, styled, on=on, width=width)
    return _box(title, lines, on=on, width=width)


def _step_row_for_display(
    row: dict,
    *,
    summary: dict[str, Any] | None,
    trainer_lines: list[str],
) -> dict:
    if not trainer_lines or not summary:
        return row
    if not _test_eval_pending(summary) or not _training_epochs_done(summary, row):
        return row
    patched = dict(row)
    mp = patched.get("micro_planned")
    if mp is not None:
        patched["micro_done"] = mp
    patched["eta"] = "test eval"
    patched["eta_s"] = None
    return patched


def discover_trainer_lines(run_tag: str | None) -> list[str]:
    """Read-only pgrep for the trainer matching run_tag (watch / --once)."""
    if not run_tag:
        return []
    out: list[str] = []
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
                    out.append(f"{pid} {args}")
            except Exception:
                continue
    except Exception:
        return []
    return _filter_trainers_for_run(out, run_tag)


def render_job_completed_banner(
    summary: dict[str, Any],
    *,
    run_tag: str | None,
    on: bool,
    width: int,
) -> str:
    p = palette(on)
    lines = job_completed_banner_lines(summary, run_tag=run_tag)
    title = p["gold"]("✓ COMPLETE") if on else "✓ COMPLETE"
    styled: list[str] = []
    for i, ln in enumerate(lines):
        if i == 0:
            styled.append(p["gold"](ln) if on else ln)
        elif "FAIL" in ln or "UNCALIBRATED" in ln:
            styled.append(p["warn"](ln) if on else ln)
        elif "PASS" in ln:
            styled.append(p["mint"](ln) if on else ln)
        elif on:
            styled.append(p["cream"](ln))
        else:
            styled.append(ln)
    return _box(title, styled, on=on, width=width)


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
    blink: bool = False,
) -> str:
    p = palette(on)
    face = owl_emoticon(blink=blink, on=on)
    if compact:
        if not resources or not resources.get("available"):
            line = f"{owl_emoticon(blink=blink, on=False)} HW gpu: unavailable · trainer: idle"
            if on:
                return _panel_row(face + " " + p["dim"]("HW gpu: unavailable · trainer: idle"), on=False, sides=True, width=width)
            return _panel_row(line, on=on, width=width)
        gpu = f"gpu {resources.get('gpu_util', '—')}% {_fmt_mem(resources.get('gpu_mem_used'))}/{_fmt_mem(resources.get('gpu_mem_total'))}"
        cpu = f"cpu {resources.get('cpu_pct', '—')}%"
        ram = "ram —"
        if resources.get("host_mem_used") and resources.get("host_mem_total"):
            ram = f"ram {_fmt_mem(resources['host_mem_used'])}/{_fmt_mem(resources['host_mem_total'])}"
        plain = f"{owl_emoticon(blink=blink, on=False)} HW {gpu} · {cpu} · {ram}"
        if on:
            return _panel_row(face + " " + p["cyan"](f"HW {gpu} · {cpu} · {ram}"), on=False, sides=True, width=width)
        return _panel_row(plain, on=on, width=width)
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
    hw_title = f"{face} HARDWARE"
    if on:
        styled = [p["cyan"](lines[0])] + [p["dim"](x) for x in lines[1:]]
        return _box(hw_title, styled, on=on, width=width)
    return _box(hw_title, lines, on=on, width=width)


def _side_by_side(left: str, right: str, *, gap: int = 2) -> str:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    n = max(len(left_lines), len(right_lines))
    left_w = max((_visible_len(x) for x in left_lines), default=0)
    out: list[str] = []
    for i in range(n):
        left_ln = left_lines[i] if i < len(left_lines) else " " * left_w
        r = right_lines[i] if i < len(right_lines) else ""
        left_ln = _pad_visible(left_ln, left_w)
        out.append(left_ln + " " * gap + r)
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
    test_jsonl_row: dict | None = None,
    prev_test_jsonl_row: dict | None = None,
    blink: bool = False,
    tick: int = 0,
    summary_path: Path | None = None,
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

    summary_obj = _load_summary_dict(summary_path)
    completed = job_completed_summary(
        run_tag=run_tag,
        trainer_lines=trainer_lines,
        jsonl_row=jsonl_row,
        log_tail=log_tail,
        summary_path=summary_path,
    )
    test_row = _resolve_test_row(test_jsonl_row, log_tail, summary_obj)
    stage = (
        "DONE"
        if completed is not None
        else _active_stage(
            jsonl_row,
            epoch_jsonl_row,
            summary=summary_obj,
            trainer_lines=trainer_lines,
        )
    )
    if test_row and test_row.get("event") == "test_step" and stage != "DONE":
        stage = "TEST"
    chunks.append(render_status_bar(timestamp=timestamp, stage=stage, on=on, width=panel_w, blink=blink))

    if stage == "TEST" and not test_row:
        chunks.append(
            render_test_eval_pending_panel(summary_obj, on=on, width=panel_w, blink=blink)
        )
        chunks.append("")

    if completed is not None:
        chunks.append(
            render_job_completed_banner(
                completed,
                run_tag=run_tag,
                on=on,
                width=panel_w,
            )
        )
        chunks.append("")

    meta: list[str] = []
    if run_tag:
        meta.append(f"watch: {run_tag}")
    if lock_line:
        lock_tag = lock_line.strip().split()[1] if len(lock_line.strip().split()) > 1 else ""
        if run_tag and lock_tag and lock_tag != run_tag:
            meta.append(f"lock (canonical GPU0): {lock_line.strip()}")
        else:
            meta.append(f"lock: {lock_line.strip()}")
    elif not run_tag:
        meta.append("lock: (none)")
    if trainer_lines:
        meta.append("trainer: running")
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
    test_text = ""
    if test_row:
        if test_row.get("event") == "test_complete" or test_row.get("n_docs"):
            test_text = render_test_metrics_event(test_row, on=on, width=box_w, blink=blink)
        else:
            test_text = render_test_step_event(
                test_row,
                on=on,
                width=box_w,
                prev_row=prev_test_jsonl_row,
                tick=tick,
                blink=blink,
            )
    if test_text:
        metric_text = test_text
    elif jsonl_row:
        ev = jsonl_row.get("event", "step")
        if ev == "epoch":
            metric_text = render_epoch_event(jsonl_row, on=on, width=box_w, blink=blink)
        else:
            display_row = _step_row_for_display(
                jsonl_row,
                summary=summary_obj,
                trainer_lines=trainer_lines,
            )
            metric_text = render_step_event(
                display_row,
                on=on,
                width=box_w,
                prev_row=prev_step_row,
                tick=tick,
                blink=blink,
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
        show_epoch = not jsonl_row or jsonl_row.get("event") != "epoch"
        if test_text:
            show_epoch = True
        if show_epoch:
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
        hw_text = render_hardware_panel(resources, on=on, width=hw_width, compact=False, blink=blink)
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
            chunks.append(render_hardware_panel(resources, on=on, width=panel_w, compact=True, blink=blink))
        else:
            chunks.append(render_hardware_panel(resources, on=on, width=box_w, compact=False, blink=blink))

    face = owl_emoticon(blink=blink, on=on)
    pipe_stage = "HUB" if stage == "DONE" else stage
    footer_plain = f"{owl_emoticon(blink=blink, on=False)} {mailroom_route_banner()}  ·  {mailroom_pipeline_hint(pipe_stage)}"
    if on:
        footer = face + " " + p["dim"](f"{mailroom_route_banner()}  ·  {mailroom_pipeline_hint(pipe_stage)}")
        chunks.append(_panel_row(footer, on=False, sides=True, width=panel_w))
    else:
        chunks.append(_panel_row(footer_plain, on=on, sides=True, width=panel_w))

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
                elif ev in ("test_step", "test_complete"):
                    if ev == "test_complete":
                        print(render_test_metrics_event(row, on=on), flush=True)
                    else:
                        print(render_test_step_event(row, on=on), flush=True)
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
    test_path: Path | None,
    run_tag: str | None,
    lock_path: Path | None,
    on: bool,
    interval: float = 2.0,
) -> int:
    tick = 0
    jsonl_pos = jsonl_path.stat().st_size if jsonl_path and jsonl_path.is_file() else 0
    epoch_pos = epoch_path.stat().st_size if epoch_path and epoch_path.is_file() else 0
    test_pos = test_path.stat().st_size if test_path and test_path.is_file() else 0
    log_pos = log_path.stat().st_size if log_path and log_path.is_file() else 0
    last_sig = (
        _file_sig(log_path),
        _file_sig(jsonl_path),
        _file_sig(epoch_path),
        _file_sig(test_path),
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
            if test_path and test_path.is_file():
                try:
                    size = test_path.stat().st_size
                    if size < test_pos:
                        test_pos = 0
                    else:
                        test_pos = size
                except OSError:
                    pass
            sig = (
                _file_sig(log_path),
                _file_sig(jsonl_path),
                _file_sig(epoch_path),
                _file_sig(test_path),
            )
            if sig != last_sig:
                pass  # signature bump — follow loop refreshes panels
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
            eff_tag = run_tag
            if not eff_tag and log_path and log_path.suffix == ".log":
                eff_tag = log_path.stem
            trainer_lines = _filter_trainers_for_run(trainer_lines, eff_tag)
            summary_p = summary_path_for_watch(eff_tag, jsonl_path, log_path)
            log_tail: list[str] = []
            if log_path and log_path.is_file():
                try:
                    log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                except OSError:
                    log_tail = []
            prev_row, row = (None, None)
            if jsonl_path:
                prev_row, row = read_last_two_jsonl(jsonl_path)
                if row is None and prev_row is not None:
                    row, prev_row = prev_row, None
            prev_test, test_row = (None, None)
            if test_path:
                prev_test, test_row = read_last_two_jsonl(test_path)
                if test_row is None and prev_test is not None:
                    test_row, prev_test = prev_test, None
            epoch_row = read_last_jsonl(epoch_path) if epoch_path else None
            resources = sample_resources(lock_line, trainer_lines)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            blink = (tick % 6 == 5)
            snap_tag = run_tag or eff_tag or (
                lock_line.split()[1] if lock_line and len(lock_line.split()) > 1 else None
            )
            text = render_watch_snapshot(
                timestamp=ts,
                lock_line=lock_line,
                trainer_lines=trainer_lines,
                log_tail=log_tail,
                jsonl_row=row,
                epoch_jsonl_row=epoch_row,
                run_tag=snap_tag,
                on=on,
                resources=resources,
                prev_step_row=prev_row,
                test_jsonl_row=test_row,
                prev_test_jsonl_row=prev_test,
                blink=blink,
                tick=tick,
                summary_path=summary_p,
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
            while waited < interval and not stop:
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
    ap = argparse.ArgumentParser(description="LLM Mailroom retro digital TUI for M9a logs (read-only).")
    ap.add_argument("--jsonl", type=Path, help="train_steps.jsonl path")
    ap.add_argument("--epoch-jsonl", type=Path, help="epoch_metrics.jsonl path")
    ap.add_argument("--test-jsonl", type=Path, help="test_steps.jsonl path")
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
        rt = args.run_tag or None
        if not rt and args.log and args.log.suffix == ".log":
            rt = args.log.stem
        test_path = args.test_jsonl
        if test_path is None:
            test_path = test_steps_path_for_watch(rt, args.jsonl, args.log)
        return follow_live(
            log_path=args.log,
            jsonl_path=args.jsonl,
            epoch_path=args.epoch_jsonl,
            test_path=test_path,
            run_tag=rt,
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
        log_tail = args.log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]

    watch_mode = bool(args.timestamp or args.lock_line or args.trainer_line or args.run_tag)
    if watch_mode or row or log_tail or epoch_row:
        ts = args.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resources: dict[str, Any] | None = None
        try:
            # Read-only sampler; safe in --plain (still no ANSI/alt-screen).
            resources = sample_resources(args.lock_line or None, list(args.trainer_line))
        except Exception:
            resources = None
        rt = args.run_tag or None
        if not rt and args.log and args.log.suffix == ".log":
            rt = args.log.stem
        trainers = _filter_trainers_for_run(list(args.trainer_line), rt)
        if rt and not trainers:
            trainers = discover_trainer_lines(rt)
        summ = summary_path_for_watch(rt, args.jsonl, args.log)
        test_path = args.test_jsonl or test_steps_path_for_watch(rt, args.jsonl, args.log)
        prev_test, test_row = (None, None)
        if test_path:
            prev_test, test_row = read_last_two_jsonl(test_path)
            if test_row is None and prev_test is not None:
                test_row, prev_test = prev_test, None
        text = render_watch_snapshot(
            timestamp=ts,
            lock_line=args.lock_line or None,
            trainer_lines=trainers,
            log_tail=log_tail,
            jsonl_row=row,
            epoch_jsonl_row=epoch_row,
            run_tag=rt,
            on=on,
            width=args.width,
            height=args.height,
            resources=resources,
            prev_step_row=prev_row,
            test_jsonl_row=test_row,
            prev_test_jsonl_row=prev_test,
            summary_path=summ,
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
