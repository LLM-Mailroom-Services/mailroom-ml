"""Static SVG charts for markdown reports (stdlib only).

The same chart forms as the Mailroom reports hub (grouped bars, a
row-normalised confusion matrix, threshold curves and ALE curves with a
bootstrap band), rendered as standalone SVG files so a markdown report on
GitHub can embed them with ``![](charts/name.svg)``. Output is deterministic:
the same input always gives byte-identical SVG, so re-rendering a report only
shows a diff when its numbers change.
"""

from __future__ import annotations

import math
from html import escape

FONT = "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"
INK, INK2, MUTED, GRID, AXIS, BG = "#0b0b0b", "#3d3d3a", "#6f6e69", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, GREEN, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b0", "#d03b3b"
PALETTE = [BLUE, ORANGE, GREEN, GREY, "#8a5cd6", "#c7457d"]


def _n(v: float) -> str:
    """Compact, stable number formatting for coordinates."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


class _Svg:
    def __init__(self, w: int, h: int, title: str, subtitle: str = ""):
        self.w, self.h = w, h
        self.parts: list[str] = []
        self.title, self.subtitle = title, subtitle

    def add(self, tag: str, text: str | None = None, **attrs) -> None:
        a = " ".join(f'{k.rstrip("_").replace("_", "-")}="{escape(str(_n(v) if isinstance(v, float) else v))}"'
                     for k, v in attrs.items() if v is not None)
        self.parts.append(f"<{tag} {a}>{escape(text)}</{tag}>" if text is not None else f"<{tag} {a}/>")

    def text(self, x, y, s, size=11, fill=MUTED, anchor="start", weight=None, rotate=None):
        self.add("text", str(s), x=float(x), y=float(y), font_size=size, fill=fill, text_anchor=anchor,
                 font_weight=weight, transform=f"rotate({rotate} {_n(x)} {_n(y)})" if rotate else None)

    def render(self) -> str:
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="{escape(FONT)}" role="img">',
                f"<title>{escape(self.title)}</title>",
                f'<rect width="{self.w}" height="{self.h}" fill="{BG}"/>',
                f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{escape(self.title)}</text>']
        if self.subtitle:
            head.append(f'<text x="16" y="42" font-size="11.5" fill="{MUTED}">{escape(self.subtitle)}</text>')
        return "\n".join(head + self.parts + ["</svg>"]) + "\n"


def nice_ticks(top: float, n: int = 5) -> list[float]:
    if top <= 0:
        return [0.0, 1.0]
    raw = top / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    k = math.ceil(top / step - 1e-9)
    return [round(i * step, 10) for i in range(k + 1)]


def _legend(s: _Svg, items: list[tuple[str, str]], x: float, y: float, line: bool = False) -> None:
    for name, color in items:
        if line:
            s.add("line", x1=float(x), x2=float(x + 14), y1=float(y - 4), y2=float(y - 4), stroke=color, stroke_width=2.5)
        else:
            s.add("rect", x=float(x), y=float(y - 9), width=10, height=10, rx=2, fill=color)
        s.text(x + 18, y, name, fill=INK2)
        x += 18 + 6.6 * len(name) + 18


def grouped_bars(title: str, cats: list[str], series: list[tuple[str, str, list]], *, subtitle: str = "",
                 ymax: float | None = None, fmt=lambda v: f"{v:.2f}", tick_fmt=None, ref: tuple[float, str] | None = None,
                 labels: bool = True, width: int = 760, height: int = 360) -> str:
    """Vertical grouped bars; ``series`` is ``[(name, color, values)]`` aligned to ``cats`` (None = no bar)."""
    s = _Svg(width, height, title, subtitle)
    L, R, T, B = 52, 16, 76, 58
    vals = [v for _, _, vs in series for v in vs if v is not None]
    ticks = nice_ticks(ymax if ymax is not None else max(vals + [ref[0] if ref else 0, 1e-9]))
    top = ticks[-1]
    tick_fmt = tick_fmt or fmt
    y = lambda v: T + (1 - v / top) * (height - T - B)  # noqa: E731
    for t in ticks:
        s.add("line", x1=L, x2=width - R, y1=y(t), y2=y(t), stroke=GRID)
        s.text(L - 6, y(t) + 4, tick_fmt(t), anchor="end")
    _legend(s, [(n, c) for n, c, _ in series], L, 62)
    band = (width - L - R) / max(len(cats), 1)
    bw = min(34.0, band * 0.8 / max(len(series), 1))
    for i, cat in enumerate(cats):
        x0 = L + i * band + (band - bw * len(series)) / 2
        for j, (_, color, vs) in enumerate(series):
            v = vs[i]
            if v is None:
                continue
            s.add("rect", x=x0 + j * bw + 1, y=y(v), width=bw - 2, height=max(y(0) - y(v), 0.0), fill=color, rx=2)
            if labels:
                s.text(x0 + j * bw + bw / 2, y(v) - 4, fmt(v), size=10, fill=INK2, anchor="middle")
        for k, part in enumerate(_wrap(cat, max(8, int(band / 6.5)))):
            s.text(L + i * band + band / 2, height - B + 16 + 13 * k, part, fill=INK2, anchor="middle")
    s.add("line", x1=L, x2=width - R, y1=y(0), y2=y(0), stroke=AXIS)
    if ref:
        s.add("line", x1=L, x2=width - R, y1=y(ref[0]), y2=y(ref[0]), stroke=INK2, stroke_dasharray="5 4")
        s.text(L + 4, y(ref[0]) - 5, ref[1], size=10.5, fill=INK2)
    return s.render()


def hbars(title: str, labels: list[str], series: list[tuple[str, str, list]], *, subtitle: str = "",
          fmt=lambda v: str(v), width: int = 760, label_w: int = 190) -> str:
    """Horizontal grouped bars, for long category names (e.g. true vs predicted subclass counts)."""
    row = 12 * len(series) + 10
    height = 80 + row * len(labels) + 34
    s = _Svg(width, height, title, subtitle)
    L, R, T = label_w, 44, 76
    vals = [v for _, _, vs in series for v in vs if v is not None]
    ticks = nice_ticks(max(vals + [1e-9]), 4)
    top = ticks[-1]
    x = lambda v: L + v / top * (width - L - R)  # noqa: E731
    for t in ticks:
        s.add("line", x1=x(t), x2=x(t), y1=T - 4, y2=height - 30, stroke=GRID)
        s.text(x(t), height - 14, fmt(t), anchor="middle")
    _legend(s, [(n, c) for n, c, _ in series], 16, 62)
    for i, lab in enumerate(labels):
        y0 = T + i * row
        s.text(L - 8, y0 + row / 2 + 2, lab, fill=INK2, anchor="end")
        for j, (_, color, vs) in enumerate(series):
            v = vs[i] or 0
            s.add("rect", x=float(L), y=float(y0 + 5 + j * 12), width=max(x(v) - L, 0.0), height=10, fill=color, rx=2)
            s.text(x(v) + 4, y0 + 14 + j * 12, fmt(v), size=10, fill=INK2)
    return s.render()


def heatmap(title: str, rows: list[str], cols: list[str], counts: list[list[int]], *, subtitle: str = "",
            width: int = 760, label_w: int = 150) -> str:
    """Confusion matrix: rows = true, cols = predicted; shade = share of the row, text = count."""
    cell = min(92.0, (width - label_w - 20) / max(len(cols), 1))
    height = int(106 + cell * len(rows) + 30)
    s = _Svg(width, height, title, subtitle)
    L, T = label_w, 106
    s.text(L + cell * len(cols) / 2, T - 36, "predicted", fill=INK2, anchor="middle", weight=600)
    for j, c in enumerate(cols):
        parts = _wrap(c, max(int(cell / 6.2), 6))[:2]
        for k, part in enumerate(parts):
            s.text(L + j * cell + cell / 2, T - 8 - 12 * (len(parts) - 1 - k), part, size=10.5, fill=INK2, anchor="middle")
    for i, r in enumerate(rows):
        tot = sum(counts[i]) or 1
        s.text(L - 8, T + i * cell + cell / 2 + 4, r, fill=INK2, anchor="end")
        for j in range(len(cols)):
            v = counts[i][j]
            share = v / tot
            fill = _mix(BG, BLUE if i == j else RED, 0.08 + 0.92 * share if v else 0.0)
            s.add("rect", x=L + j * cell + 1, y=T + i * cell + 1, width=cell - 2, height=cell - 2, fill=fill, rx=3)
            ink = BG if share > 0.55 else INK2
            s.text(L + j * cell + cell / 2, T + i * cell + cell / 2 + 1, str(v), size=12.5, fill=ink, anchor="middle", weight=600)
            if v:
                s.text(L + j * cell + cell / 2, T + i * cell + cell / 2 + 15, f"{share:.0%}", size=9.5, fill=ink, anchor="middle")
    s.text(16, height - 10, "true class (rows)", size=10.5)
    return s.render()


def lines(title: str, series: list[dict], *, subtitle: str = "", x_label: str = "", y_label: str = "",
          y_range: tuple[float, float] | None = None, x_fmt=lambda v: _n(v), y_fmt=lambda v: _n(v),
          hlines: list[tuple[float, str]] = (), vlines: list[tuple[float, str]] = (), rug: list[float] = (),
          width: int = 760, height: int = 340) -> str:
    """Line chart. Each series: ``{"name", "color", "x", "y", optional "lo"/"hi" band, optional "dash"}``."""
    s = _Svg(width, height, title, subtitle)
    L, R, T, B = 58, 18, 76, 52
    xs = [v for se in series for v in se["x"]]
    ys = [v for se in series for k in ("y", "lo", "hi") for v in se.get(k, []) if v is not None]
    ys += [h for h, _ in hlines]
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x1 = x0 + 1
    if y_range:
        y0, y1 = y_range
    else:
        y0, y1 = min(ys + [0.0]), max(ys + [0.0])
        pad = (y1 - y0) * 0.08 or 0.1
        y0, y1 = y0 - (pad if y0 < 0 else 0), y1 + pad
    X = lambda v: L + (v - x0) / (x1 - x0) * (width - L - R)  # noqa: E731
    Y = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (height - T - B)  # noqa: E731
    for t in _span_ticks(y0, y1):
        s.add("line", x1=L, x2=width - R, y1=Y(t), y2=Y(t), stroke=AXIS if abs(t) < 1e-12 else GRID)
        s.text(L - 6, Y(t) + 4, y_fmt(t), anchor="end")
    for t in _span_ticks(x0, x1, 6):
        s.text(X(t), height - B + 16, x_fmt(t), anchor="middle")
    if x_label:
        s.text((L + width - R) / 2, height - 12, x_label, fill=INK2, anchor="middle")
    if y_label:
        s.text(14, (T + height - B) / 2, y_label, fill=INK2, anchor="middle", rotate=-90)
    for v in rug:
        s.add("line", x1=X(v), x2=X(v), y1=float(height - B), y2=float(height - B - 6), stroke=INK2, stroke_opacity=0.5)
    for h, lab in hlines:
        s.add("line", x1=L, x2=width - R, y1=Y(h), y2=Y(h), stroke=INK2, stroke_dasharray="5 4")
        s.text(width - R, Y(h) - 5, lab, size=10.5, fill=INK2, anchor="end")
    for v, lab in vlines:
        s.add("line", x1=X(v), x2=X(v), y1=float(T), y2=float(height - B), stroke=INK2, stroke_dasharray="3 3")
        s.text(X(v) + 4, T + 28, lab, size=10.5, fill=INK2)
    for se in series:
        if se.get("lo") and se.get("hi"):
            pts = [f"{_n(X(a))},{_n(Y(b))}" for a, b in zip(se["x"], se["hi"], strict=False)]
            pts += [f"{_n(X(a))},{_n(Y(b))}" for a, b in reversed(list(zip(se["x"], se["lo"], strict=False)))]
            s.add("polygon", points=" ".join(pts), fill=se["color"], fill_opacity=0.16)
    for se in series:
        pts = " ".join(f"{_n(X(a))},{_n(Y(b))}" for a, b in zip(se["x"], se["y"], strict=False) if b is not None)
        s.add("polyline", points=pts, fill="none", stroke=se["color"], stroke_width=2.2,
              stroke_dasharray=se.get("dash"), stroke_linejoin="round")
        for a, b in zip(se["x"], se["y"], strict=False):
            if b is not None:
                s.add("circle", cx=X(a), cy=Y(b), r=2.6, fill=se["color"])
    _legend(s, [(se["name"], se["color"]) for se in series], L, 62, line=True)
    return s.render()


def _span_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    out, t = [], start
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return out + [cur] if cur else out


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=False))
