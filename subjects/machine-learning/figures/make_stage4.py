"""Figures for stage 4 (lessons 12-14). Run from the repo root:
    python3 subjects/machine-learning/figures/make_stage4.py
Figures that illustrate a lesson's code run that code (code/<lesson>.py), so they show
exactly what the lesson prints."""
import contextlib
import io
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.figures.svgplot import Plot, c  # noqa: E402

OUT = Path(__file__).resolve().parent
CODE = OUT.parent / "code"


def run(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(CODE / f"{name}.py"))


def regions(p, predict, lim, res=90, colors=("data", "accent"), opacity=0.16):
    """Shade the plane by predicted class, one rectangle per run of equal predictions in a row."""
    xs = np.linspace(-lim, lim, res + 1); step = xs[1] - xs[0]
    cx = (xs[:-1] + xs[1:]) / 2
    G = np.array([[x, y] for y in cx for x in cx])
    k = predict(G).reshape(res, res)
    for r in range(res):
        start = 0
        for j in range(1, res + 1):
            if j == res or k[r, j] != k[r, start]:
                p.rect(xs[start], xs[r], (j - start) * step, step, color=colors[int(k[r, start])], opacity=opacity)
                start = j


# ---------- 12: decision regions on the spirals ----------
ns = run("12-neurons-and-mlps")
Xtr, ytr, Xte, yte = ns["Xtr"], ns["ytr"], ns["Xte"], ns["yte"]
p = Plot(900, 310)
for i, (sizes, name) in enumerate([([2, 1], "logistic regression"), ([2, 8, 1], "8 hidden ReLUs"), ([2, 64, 1], "64 hidden ReLUs")]):
    params = ns["train"](sizes, Xtr, ytr)
    acc = ns["acc"](params, Xte, yte)
    p.panel(i * 300, 0, 300, 310, (-1.25, 1.25), (-1.25, 1.25), margin=(30, 10, 10, 10))
    regions(p, lambda G: (ns["forward"](params, G)[0] > 0).astype(int), 1.25)
    for k, col in [(0, "data"), (1, "accent")]:
        m = ytr == k
        p.dots(Xtr[m, 0], Xtr[m, 1], color=col, r=2.1, opacity=0.9)
    p.text(p.box[0], 18, f"{name}: test acc {acc:.2f}", color="fg", size=12, px=True)
    print("12", name, round(acc, 3))
p.save(OUT / "mlp-spirals.svg", "Decision regions on two interleaved spirals: logistic regression cuts the plane with a line, 8 hidden ReLUs make a polygon, 64 hidden ReLUs follow the spiral arms")

# ---------- 12: universal approximation construction; tent-map compositions ----------
relu = lambda z: np.maximum(z, 0)
f = lambda x: np.sin(2 * np.pi * x) * np.exp(-x) + 0.5 * x
knots = np.linspace(0, 1, 9)
s = np.diff(f(knots)) / np.diff(knots)
s_prev = np.r_[0, s[:-1]]
x = np.linspace(0, 1, 500)
F = f(knots[0]) + sum((sk - sp) * relu(x - xk) for sk, sp, xk in zip(s, s_prev, knots[:-1]))
p = Plot(900, 300)
p.panel(0, 0, 430, 300, (0, 1), (-0.6, 1.2), margin=(36, 14, 34, 40))
p.axes(xlabel="x", xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[-0.5, 0, 0.5, 1])
p.line(x, f(x), color="muted", width=1.6, dash="5 4")
p.line(x, F, color="accent", width=2.2)
p.dots(knots, f(knots), color="accent", r=3.4)
p.text(0.98, 1.08, "8 ReLUs (solid) vs f (dashed)", color="fg", size=11, anchor="end")
p.text(p.box[0], 16, "one hidden layer: interpolate between knots", color="fg", size=12, px=True)
print("12 max interpolation error", round(float(np.max(np.abs(F - f(x)))), 3))
t = lambda z: 2 * relu(z) - 4 * relu(z - 0.5)
p.panel(450, 0, 450, 300, (0, 1), (-0.05, 1.25), margin=(36, 14, 34, 40))
p.axes(xlabel="x", xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.5, 1])
x = np.linspace(0, 1, 2049)
z = x.copy()
for L, col in [(1, "muted"), (2, "data"), (3, "accent")]:
    z = t(z)
    p.line(x, z, color=col, width=2.4 if L == 3 else 1.8, opacity=1 if L == 3 else 0.85)
p.text(0.02, 1.17, "L = 1: 2 pieces", color="muted", size=11)
p.text(0.36, 1.17, "L = 2: 4", color="data", size=11)
p.text(0.6, 1.17, "L = 3: 8 pieces", color="accent", size=11)
p.text(p.box[0], 16, "depth: compose the tent map L times", color="fg", size=12, px=True)
p.save(OUT / "relu-pieces.svg", "Left: a one-hidden-layer ReLU network with 8 units interpolates a smooth function with straight segments between knots. Right: the tent map composed with itself 1, 2 and 3 times gives 2, 4 and 8 linear pieces")


# ---------- 13: a computational graph, forward values and backward gradients ----------
import math  # noqa: E402
X0, Y0 = 2.0, -1.0
vals = {"x": X0, "y": Y0, "a": X0 * Y0, "b": math.exp(X0)}
vals["c"] = vals["a"] + vals["b"]; vals["f"] = vals["c"] * X0
grads = {"f": 1.0, "c": X0, "a": X0, "b": X0, "y": X0 * X0}
via = {"f": vals["c"], "b": X0 * math.exp(X0), "a": X0 * Y0}         # x's three contributions
grads["x"] = sum(via.values())
pos = {"x": (90, 160), "y": (90, 262), "b": (300, 160), "a": (300, 262), "c": (510, 211), "f": (720, 160)}
label = {"x": "x", "y": "y", "a": "a = x·y", "b": "b = exp(x)", "c": "c = a + b", "f": "f = c·x"}
W2, H2 = 62, 25
p = Plot(900, 320)


def edge(u, v, pts=None, lab=None, lab_at=None):
    (x1, y1), (x2, y2) = pos[u], pos[v]
    pts = pts or []
    path = [(x1 + W2 if not pts else x1, y1 if not pts else y1 - H2)] + pts + [None]
    tx, ty = (x2 - W2, y2) if not pts else (x2, y2 - H2)
    path[-1] = (tx, ty)
    if len(path) == 2:
        sx, sy = path[0]
        if abs(y2 - y1) > 1:                                         # diagonal: leave from the right edge
            sx, sy = x1 + W2, y1 + (H2 * 0.5 if y2 > y1 else -H2 * 0.5)
            tx, ty = x2 - W2, y2 - (H2 * 0.5 if y2 > y1 else -H2 * 0.5)
        p.arrow(sx, sy, tx, ty, color="muted", width=1.3, pixels=True)
    else:
        for (ax, ay), (bx, by) in zip(path[:-2], path[1:-1]):
            p.items.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" style="stroke:{c("muted")};stroke-width:1.3"/>')
        p.arrow(*path[-2], *path[-1], color="muted", width=1.3, pixels=True)
    if lab:
        p.text(*lab_at, lab, color="data", size=11, anchor="middle", px=True)


edge("x", "b", lab=f"+{via['b']:.3f}", lab_at=(195, 152))
edge("y", "a", lab=f"{grads['y']:.3f}", lab_at=(195, 254))
edge("x", "a", lab=f"{via['a']:.3f}", lab_at=(178, 222))
edge("b", "c"); edge("a", "c")
edge("x", "f", pts=[(90, 62), (720, 62)], lab=f"+{via['f']:.3f}", lab_at=(405, 54))
edge("c", "f")
p.arrow(720 + W2, 160, 860, 160, color="muted", width=1.3, pixels=True)
for k, (cx, cy) in pos.items():
    p.rect(cx - W2, cy - H2, 2 * W2, 2 * H2, color="card", opacity=1, stroke="rule", px=True, rx=6)
    p.text(cx, cy - 6, label[k], color="fg", size=12.5, anchor="middle", px=True)
    p.text(cx - 6, cy + 13, f"{vals[k]:.3f}", color="accent", size=11, anchor="end", px=True)
    p.text(cx + 6, cy + 13, f"{grads[k]:.3f}", color="data", size=11, anchor="start", px=True)
p.text(20, 22, "forward value", color="accent", size=11.5, px=True)
p.text(150, 22, "gradient ∂f/∂(node), computed backwards", color="data", size=11.5, px=True)
p.text(860, 300, f"∂f/∂x = {via['f']:.3f} + {via['b']:.3f} + ({via['a']:.3f}) = {grads['x']:.3f}", color="fg", size=11.5, anchor="end", px=True)
print("13 graph", {k: round(v, 3) for k, v in vals.items()}, {k: round(v, 3) for k, v in grads.items()})
p.save(OUT / "backprop-graph.svg", "The computational graph of f = (x y + e^x) x at x = 2, y = -1. Each node shows its forward value and its gradient. x feeds three nodes, so its gradient is the sum of three contributions: 5.389 + 14.778 - 2 = 18.167")


# ---------- 14: signal propagation at init; training a 20-layer network ----------
ns = run("14-training-deep-networks")
X = np.random.default_rng(0).standard_normal((1000, 256))
p = Plot(900, 310)
p.panel(0, 0, 420, 310, (1, 50), (-12, 1), margin=(36, 14, 34, 44))
p.axes(xlabel="layer", ylabel="std of pre-activations", yticks=[-12, -9, -6, -3, 0],
       fmt=lambda v: "1" if v == 0 else f"1e{int(v)}")
for v in [1, 10, 20, 30, 40, 50]:
    p.seg(v, -12, v, 1, color="rule", width=1)
    p.text(p.X(v), p.box[1] + p.box[3] + 15, str(v), size=11, anchor="middle", px=True)
cols = {"N(0, 0.01²)": "muted", "Xavier 1/fan_in": "c", "He 2/fan_in": "accent"}
for name, scale in ns["schemes"].items():
    params = ns["init"]([256] * 51 + [1], scale, seed=1)
    _, cache = ns["forward"](params, X)
    stds = np.array([np.std(cc[1]) for cc in cache[:-1]])
    L = np.arange(1, 51); ok = np.log10(stds) > -12
    p.line(L[ok], np.log10(stds[ok]), color=cols[name], width=2.2)
lab = {"N(0, 0.01²)": (5.5, -9.5, "N(0, 0.01²)"), "Xavier 1/fan_in": (50, -8.4, "Xavier: halves the variance per layer"), "He 2/fan_in": (49, -1.1, "He: keeps it constant")}
for name, (x, y, s) in lab.items():
    p.text(x, y, s, color=cols[name], size=11, anchor="start" if name == "N(0, 0.01²)" else "end")
p.text(p.box[0], 16, "50 ReLU layers at initialisation", color="fg", size=12, px=True)
p.panel(440, 0, 460, 310, (0, 1000), (0, 0.8), margin=(36, 14, 34, 44))
p.axes(xlabel="training step", ylabel="training log-loss", xticks=[0, 250, 500, 750, 1000], yticks=[0, 0.2, 0.4, 0.6, 0.8])
for name, scale, norm, col, y_lab in [("N(0, 0.01²)", "N(0, 0.01²)", False, "muted", 0.725), ("Xavier", "Xavier 1/fan_in", False, "c", 0.655),
                                       ("He", "He 2/fan_in", False, "accent", 0.12), ("Xavier + layer norm", "Xavier 1/fan_in", True, "data", 0.33)]:
    hist, acc, _ = ns["train"](ns["schemes"][scale], norm)
    p.line(np.arange(1000), np.minimum(hist, 0.8), color=col, width=1.6)
    p.text(990, y_lab, f"{name} (test {acc:.2f})", color=col, size=11, anchor="end")
    print("14", name, round(hist[249], 3), round(hist[-1], 3), acc)
p.text(p.box[0], 16, "training a 20-layer network on the spirals", color="fg", size=12, px=True)
p.save(OUT / "signal-propagation.svg", "Left: the standard deviation of pre-activations through 50 ReLU layers, on a log scale: with small weights it collapses at once, with Xavier initialisation it halves in variance each layer, with He initialisation it stays constant. Right: training loss of a 20-layer network: the first two initialisations never train, He initialisation and Xavier with layer normalisation both reach zero loss")
