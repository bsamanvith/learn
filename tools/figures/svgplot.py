"""Tiny SVG plotting for lesson figures that follow the site theme.

Figures are written as plain SVG that draws with the site's CSS variables (--fg, --accent,
--event-a, ...), so one file looks right in light, paper and dark themes when it is inlined
into a page. Lessons pull them in with <!-- figure: name --> (see site/src/subjects.ts).

    from tools.figures.svgplot import Plot
    p = Plot(560, 300, xlim=(-3, 3), ylim=(0, 1))
    p.axes(xlabel="z", ylabel="σ(z)")
    p.line(xs, ys, color="accent")
    p.save("subjects/machine-learning/figures/sigmoid.svg", "The sigmoid function")
"""
from __future__ import annotations

import html
from pathlib import Path

COLORS = {
    "fg": "var(--fg)", "muted": "var(--muted)", "rule": "var(--rule)", "bg": "var(--bg)",
    "card": "var(--card)", "panel": "var(--panel)",
    "accent": "var(--accent)", "data": "var(--event-a)", "c": "var(--event-c)", "both": "var(--event-both)",
    "good": "var(--good)", "bad": "var(--bad)",
}
MONO = "var(--f-mono)"
SERIF = "var(--f-display)"


def c(name: str) -> str:
    return COLORS.get(name, name)


def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


class Plot:
    """A figure with one or more panels. Coordinates are data units inside the current panel."""

    def __init__(self, width: int, height: int, xlim=(0, 1), ylim=(0, 1), margin=(16, 16, 34, 44)):
        self.W, self.H = width, height
        self.items: list[str] = []
        self.panel(0, 0, width, height, xlim, ylim, margin)

    # ---------- panels ----------
    def panel(self, x0, y0, w, h, xlim, ylim, margin=(16, 16, 34, 44)):
        """Set the drawing box. margin = (top, right, bottom, left) in pixels."""
        t, r, b, l = margin
        self.box = (x0 + l, y0 + t, w - l - r, h - t - b)
        self.xlim, self.ylim = xlim, ylim
        return self

    def X(self, x):
        bx, _, bw, _ = self.box
        return bx + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * bw

    def Y(self, y):
        _, by, _, bh = self.box
        return by + (1 - (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])) * bh

    # ---------- primitives (pixel or data) ----------
    def raw(self, s: str):
        self.items.append(s)
        return self

    def line(self, xs, ys, color="accent", width=2.2, dash=None, opacity=1.0):
        pts = " ".join(f"{_f(self.X(x))},{_f(self.Y(y))}" for x, y in zip(xs, ys))
        d = f";stroke-dasharray:{dash}" if dash else ""
        self.items.append(f'<polyline points="{pts}" style="fill:none;stroke:{c(color)};stroke-width:{width};stroke-linejoin:round;stroke-linecap:round;opacity:{opacity}{d}"/>')
        return self

    def area(self, xs, ys_top, ys_bot, color="accent", opacity=0.15):
        top = [f"{_f(self.X(x))},{_f(self.Y(y))}" for x, y in zip(xs, ys_top)]
        bot = [f"{_f(self.X(x))},{_f(self.Y(y))}" for x, y in zip(reversed(list(xs)), reversed(list(ys_bot)))]
        self.items.append(f'<polygon points="{" ".join(top + bot)}" style="fill:{c(color)};opacity:{opacity};stroke:none"/>')
        return self

    def seg(self, x1, y1, x2, y2, color="muted", width=1.2, dash=None, opacity=1.0):
        d = f";stroke-dasharray:{dash}" if dash else ""
        self.items.append(f'<line x1="{_f(self.X(x1))}" y1="{_f(self.Y(y1))}" x2="{_f(self.X(x2))}" y2="{_f(self.Y(y2))}" style="stroke:{c(color)};stroke-width:{width};opacity:{opacity}{d}"/>')
        return self

    def arrow(self, x1, y1, x2, y2, color="fg", width=1.5, head=7, pixels=False):
        import math
        if not pixels:
            x1, y1, x2, y2 = self.X(x1), self.Y(y1), self.X(x2), self.Y(y2)
        a = math.atan2(y2 - y1, x2 - x1)
        hx1, hy1 = x2 - head * math.cos(a - 0.4), y2 - head * math.sin(a - 0.4)
        hx2, hy2 = x2 - head * math.cos(a + 0.4), y2 - head * math.sin(a + 0.4)
        self.items.append(f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" style="stroke:{c(color)};stroke-width:{width}"/>'
                          f'<polygon points="{_f(x2)},{_f(y2)} {_f(hx1)},{_f(hy1)} {_f(hx2)},{_f(hy2)}" style="fill:{c(color)}"/>')
        return self

    def dots(self, xs, ys, color="data", r=3.6, hollow=False, opacity=1.0):
        st = f"fill:{c('card')};stroke:{c(color)};stroke-width:1.6" if hollow else f"fill:{c(color)}"
        for x, y in zip(xs, ys):
            self.items.append(f'<circle cx="{_f(self.X(x))}" cy="{_f(self.Y(y))}" r="{r}" style="{st};opacity:{opacity}"/>')
        return self

    def rect(self, x, y, w, h, color="accent", opacity=0.12, stroke=None, px=False, rx=0):
        if not px:
            x0, y0 = self.X(x), self.Y(y + h)
            w, h = self.X(x + w) - x0, self.Y(y) - y0
            x, y = x0, y0
        s = f";stroke:{c(stroke)};stroke-width:1.2" if stroke else ";stroke:none"
        self.items.append(f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" rx="{rx}" style="fill:{c(color)};fill-opacity:{opacity}{s}"/>')
        return self

    def text(self, x, y, s, color="muted", size=12, anchor="start", font=MONO, px=False, weight=400, italic=False):
        if not px:
            x, y = self.X(x), self.Y(y)
        it = ";font-style:italic" if italic else ""
        self.items.append(f'<text x="{_f(x)}" y="{_f(y)}" style="fill:{c(color)};font-family:{font};font-size:{size}px;font-weight:{weight}{it}" text-anchor="{anchor}">{html.escape(s)}</text>')
        return self

    def math(self, x, y, s, color="fg", size=15, anchor="start", px=False):
        """Italic serif label for symbols (not typeset maths, just a nice italic)."""
        return self.text(x, y, s, color=color, size=size, anchor=anchor, font=SERIF, px=px, italic=True)

    # ---------- axes ----------
    def axes(self, xlabel="", ylabel="", xticks=None, yticks=None, grid=True, zero=False, fmt=None):
        bx, by, bw, bh = self.box
        fmt = fmt or (lambda v: _f(v))
        if grid:
            for v in yticks or []:
                self.seg(self.xlim[0], v, self.xlim[1], v, color="rule", width=1)
            for v in xticks or []:
                self.seg(v, self.ylim[0], v, self.ylim[1], color="rule", width=1)
        self.items.append(f'<line x1="{_f(bx)}" y1="{_f(by + bh)}" x2="{_f(bx + bw)}" y2="{_f(by + bh)}" style="stroke:{c("muted")};stroke-width:1"/>')
        self.items.append(f'<line x1="{_f(bx)}" y1="{_f(by)}" x2="{_f(bx)}" y2="{_f(by + bh)}" style="stroke:{c("muted")};stroke-width:1"/>')
        if zero and self.ylim[0] < 0 < self.ylim[1]:
            self.seg(self.xlim[0], 0, self.xlim[1], 0, color="muted", width=1, dash="3 3")
        for v in xticks or []:
            self.text(self.X(v), by + bh + 15, fmt(v), size=11, anchor="middle", px=True)
        for v in yticks or []:
            self.text(bx - 6, self.Y(v) + 4, fmt(v), size=11, anchor="end", px=True)
        if xlabel:
            self.text(bx + bw, by + bh + 29, xlabel, size=11.5, anchor="end", px=True)
        if ylabel:
            self.text(bx + 4, by - 4, ylabel, size=11.5, anchor="start", px=True)
        return self

    # ---------- output ----------
    def svg(self, label: str) -> str:
        body = "\n  ".join(self.items)
        return (f'<svg class="fig" viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{html.escape(label)}" '
                f'style="width:100%;height:auto;display:block;overflow:visible">\n  {body}\n</svg>\n')

    def save(self, path: str | Path, label: str):
        Path(path).write_text(self.svg(label))
        return self
