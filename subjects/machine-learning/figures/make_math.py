"""Figures for the math toolkit (M1-M6). Run from the repo root:
    python3 subjects/machine-learning/figures/make_math.py
Every number drawn here is computed, not sketched."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.figures.svgplot import Plot, c  # noqa: E402

OUT = Path(__file__).resolve().parent
ang = np.linspace(0, 2 * np.pi, 361)

# ---------- M1: the SVD maps a circle to an ellipse; PCA ----------
A = np.array([[1.6, 0.9], [0.3, 1.1]])
U, s, Vt = np.linalg.svd(A)
for k in range(2):                                 # pick signs so the pictures point up and right
    if Vt[k, k] < 0:
        Vt[k] *= -1; U[:, k] *= -1
circ = np.stack([np.cos(ang), np.sin(ang)])
ell = A @ circ
p = Plot(900, 310)
for i, (pts, vecs, title, labs) in enumerate([
        (circ, [Vt[0], Vt[1]], "unit circle; v₁, v₂ orthonormal", ["v₁", "v₂"]),
        (ell, [s[0] * U[:, 0], s[1] * U[:, 1]], "after A: ellipse with axes σ₁u₁, σ₂u₂", ["σ₁u₁", "σ₂u₂"])]):
    p.panel(i * 300, 0, 300, 310, (-2.2, 2.2), (-2.0, 2.0), margin=(40, 12, 12, 12))
    p.seg(-2.2, 0, 2.2, 0, color="rule", width=1); p.seg(0, -2, 0, 2, color="rule", width=1)
    p.area(pts[0], pts[1], pts[1], opacity=0)
    p.line(pts[0], pts[1], color="data", width=2)
    for v, lab, col in zip(vecs, labs, ["accent", "c"]):
        p.arrow(0, 0, v[0], v[1], color=col, width=2)
        p.text(v[0] * 1.08 + 0.05, v[1] * 1.08 + 0.05, lab, color=col, size=12)
    p.text(p.box[0], 18, title, color="fg", size=11.5, px=True)
print("M1 singular values", np.round(s, 3))
# PCA panel
rng = np.random.default_rng(4)
C = np.array([[2.0, 1.2], [1.2, 1.0]])
X = rng.multivariate_normal([0, 0], C, 300)
Xc = X - X.mean(0)
ev, V = np.linalg.eigh(Xc.T @ Xc / len(X))
p.panel(600, 0, 300, 310, (-4.4, 4.4), (-4.0, 4.0), margin=(40, 12, 12, 12))
p.dots(Xc[:, 0], Xc[:, 1], color="data", r=1.9, opacity=0.55)
for k, col, lab in [(1, "accent", "PC1"), (0, "c", "PC2")]:
    v = V[:, k] * 2 * np.sqrt(ev[k]) * np.sign(V[0, k] or 1)
    p.arrow(0, 0, v[0], v[1], color=col, width=2.2)
    p.text(v[0] + 0.15, v[1] + 0.1, lab, color=col, size=12)
p.text(p.box[0], 18, f"PCA: PC1 holds {ev[1] / ev.sum():.0%} of the variance", color="fg", size=11.5, px=True)
p.save(OUT / "m1-svd.svg", "Left: the unit circle with two orthogonal right singular vectors. Middle: its image under A is an ellipse whose axes are the left singular vectors scaled by the singular values. Right: correlated 2D data with its principal axes")


# ---------- M3: Bayesian updating and the central limit theorem ----------
from math import lgamma  # noqa: E402


def beta_pdf(x, a, b):
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - (lgamma(a) + lgamma(b) - lgamma(a + b)))


rng = np.random.default_rng(10)
flips = rng.random(100) < 0.3
xs = np.linspace(0.001, 0.999, 400)
p = Plot(900, 300)
p.panel(0, 0, 450, 300, (0, 1), (0, 9.5), margin=(40, 16, 34, 40))
p.axes(xlabel="θ = probability of heads", ylabel="posterior density", xticks=[0, 0.3, 0.5, 1], yticks=[0, 3, 6, 9])
p.seg(0.3, 0, 0.3, 9.5, color="muted", dash="4 4")
LY = 70
for nflip, col, op in [(0, "muted", 1), (3, "c", 1), (10, "data", 1), (100, "accent", 1)]:
    h = int(flips[:nflip].sum())
    yv = beta_pdf(xs, 2 + h, 2 + nflip - h)
    p.line(xs, np.minimum(yv, 9.5), color=col, width=2)
    p.raw(f'<line x1="300" y1="{LY - 4}" x2="318" y2="{LY - 4}" style="stroke:{c(col)};stroke-width:2.2"/>')
    p.text(324, LY, "prior" if nflip == 0 else f"{nflip} flips, {h} head{'' if h == 1 else 's'}", size=11, px=True, color="fg"); LY += 16
p.text(p.box[0], 16, "Beta(2, 2) prior updated by coin flips (true θ = 0.3)", color="fg", size=12, px=True)
p.panel(450, 0, 450, 300, (-3.5, 4.5), (0, 1.0), margin=(40, 10, 34, 40))
p.axes(xlabel="standardised mean of n exponential draws", ylabel="density", xticks=[-3, 0, 3], yticks=[0, 0.25, 0.5, 0.75, 1])
zs = np.linspace(-3.5, 4.5, 400)
p.line(zs, np.exp(-zs**2 / 2) / np.sqrt(2 * np.pi), color="fg", width=1.4, dash="5 4")
for nn, col in [(1, "c"), (3, "data"), (30, "accent")]:
    m = rng.exponential(1.0, size=(400_000, nn)).mean(1)
    zz = (m - 1) * np.sqrt(nn)
    hist, edges = np.histogram(zz, bins=np.linspace(-3.5, 4.5, 81), density=True)
    mids = (edges[1:] + edges[:-1]) / 2
    p.line(mids, np.minimum(hist, 1.0), color=col, width=2)
p.text(-0.6, 0.93, "n = 1", color="c", size=11); p.text(0.9, 0.45, "n = 3", color="data", size=11)
p.text(-2.9, 0.3, "n = 30", color="accent", size=11); p.text(2.2, 0.1, "N(0, 1)", color="fg", size=11)
p.text(p.box[0], 16, "central limit theorem: skewed draws, Gaussian mean", color="fg", size=12, px=True)
p.save(OUT / "m3-prob.svg", "Left: a Beta posterior over a coin's bias narrows around the true value 0.3 as flips accumulate. Right: the distribution of the standardised mean of n exponential draws approaches the standard normal as n grows")


# ---------- M4: binary entropy; forward vs reverse KL ----------
pp = np.linspace(0.0005, 0.9995, 400)
Hb = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp))
p = Plot(900, 300)
p.panel(0, 0, 330, 300, (0, 1), (0, 1.1), margin=(40, 14, 34, 40))
p.axes(xlabel="p = P(heads)", ylabel="H(p) in bits", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
p.line(pp, Hb, color="accent", width=2.2)
p.dots([0.9], [-(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))], color="fg", r=3.5)
p.text(0.86, 0.5, "0.9 → 0.47 bits", color="fg", size=11, anchor="end")
p.text(p.box[0], 16, "entropy of a coin", color="fg", size=12, px=True)
xg = np.linspace(-8, 8, 4001); dx = xg[1] - xg[0]
gauss = lambda m, s: np.exp(-(xg - m) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)
target = 0.5 * gauss(-2.5, 0.8) + 0.5 * gauss(2.5, 0.8)
kl = lambda a, b: np.sum(a * np.log((a + 1e-300) / (b + 1e-300))) * dx
grid = [(m, s) for m in np.linspace(-4, 4, 161) for s in np.linspace(0.3, 4, 186)]
fwd = min(grid, key=lambda ms: kl(target, gauss(*ms)))
rev = min(grid, key=lambda ms: kl(gauss(*ms), target))
p.panel(330, 0, 570, 300, (-7, 7), (0, 0.52), margin=(40, 14, 34, 40))
p.axes(xlabel="x", ylabel="density", xticks=[-5, -2.5, 0, 2.5, 5], yticks=[0, 0.25, 0.5])
p.area(xg[::10], target[::10], np.zeros(len(xg[::10])), color="data", opacity=0.18)
p.line(xg[::10], target[::10], color="data", width=1.6)
p.line(xg[::10], gauss(*fwd)[::10], color="accent", width=2.2)
p.line(xg[::10], gauss(*rev)[::10], color="c", width=2.2)
p.text(-2.5, 0.505, "reverse KL(q‖p): picks one mode", color="c", size=11.5, anchor="middle")
p.text(6.9, 0.17, "forward KL(p‖q): covers both", color="accent", size=11.5, anchor="end")
p.text(4.6, 0.3, "target p", color="data", size=11.5)
p.text(p.box[0], 16, "fitting one Gaussian q to a two-bump p", color="fg", size=12, px=True)
print("M4 fits", fwd, rev)
p.save(OUT / "m4-info.svg", "Left: the entropy of a coin, maximal at one bit for a fair coin. Right: a Gaussian fitted to a two-bump distribution by forward KL spreads over both bumps, while reverse KL locks onto one")


# ---------- M5: convexity; Lagrange multipliers ----------
p = Plot(900, 320)
xs = np.linspace(-2, 2, 400)
fc = lambda x: 0.5 * x**2 + 0.3 * x + 0.2
fn = lambda x: 0.25 * x**4 - 0.9 * x**2 + 0.25 * x + 1.2
for i, (fun, title) in enumerate([(fc, "convex: chords lie above"), (fn, "non-convex: a chord dips below")]):
    p.panel(i * 280, 0, 280, 320, (-2, 2), (-0.2, 2.6), margin=(40, 12, 24, 20))
    p.axes(grid=False)
    p.line(xs, fun(xs), color="accent", width=2.2)
    a, b = (-1.6, 1.4) if i == 0 else (-1.5, 1.5)
    p.seg(a, fun(a), b, fun(b), color="data", width=1.6)
    p.dots([a, b], [fun(a), fun(b)], color="data", r=3.5)
    if i == 0:
        x0 = 0.6; g = x0 + 0.3
        p.seg(-0.3, fun(x0) + g * (-0.3 - x0), 1.9, fun(x0) + g * (1.9 - x0), color="c", dash="5 4", width=1.4)
        p.dots([x0], [fun(x0)], color="c", r=3.5)
        p.text(1.95, 0.05, "tangent stays below", color="c", size=11, anchor="end")
    else:
        mins = [m for m in np.roots([1, 0, -1.8, 0.25]) if abs(m.imag) < 1e-9 and 3 * m.real**2 - 1.8 > 0]
        for m in mins:
            p.dots([m.real], [fn(m.real)], color="fg", r=3.5)
        p.text(-1.9, 2.35, "two local minima (dots)", color="fg", size=11)
    p.text(p.box[0], 16, title, color="fg", size=11.5, px=True)
# Lagrange
F = lambda x, y: (x - 2) ** 2 + 2 * (y - 1.5) ** 2
th = np.linspace(0, 2 * np.pi, 20001)
k = np.argmin(F(np.cos(th), np.sin(th)))
xo, yo = np.cos(th[k]), np.sin(th[k])
p.panel(560, 0, 340, 320, (-1.4, 3.6), (-1.4, 3.0), margin=(40, 8, 12, 8))
for lev in [0.4, 1.5, F(xo, yo)]:
    r = np.sqrt(lev); e = np.stack([2 + r * np.cos(ang), 1.5 + r / np.sqrt(2) * np.sin(ang)], 1)
    p.line(e[:, 0], e[:, 1], color="accent", width=2 if abs(lev - F(xo, yo)) < 1e-9 else 1.1, opacity=0.9)
p.line(np.cos(ang), np.sin(ang), color="data", width=2.2)
p.text(-1.2, -1.25, "constraint g(x) = ‖x‖² − 1 = 0", color="data", size=11)
gf = np.array([2 * (xo - 2), 4 * (yo - 1.5)]); gg = np.array([2 * xo, 2 * yo])
lam = -(gf @ gg) / (gg @ gg)
p.arrow(xo, yo, xo - 0.32 * gf[0], yo - 0.32 * gf[1], color="accent", width=1.8)
p.arrow(xo, yo, xo + 0.3 * gg[0], yo + 0.3 * gg[1], color="data", width=2.6)
p.dots([xo], [yo], color="fg", r=4)
p.dots([2], [1.5], color="accent", r=3)
p.text(xo - 0.32 * gf[0] + 0.08, yo - 0.32 * gf[1] + 0.05, "−∇f", color="accent", size=12)
p.text(xo + 0.3 * gg[0] + 0.12, yo + 0.3 * gg[1] - 0.12, "∇g", color="data", size=12)
p.text(p.box[0], 16, "at the optimum ∇f = −λ∇g", color="fg", size=11.5, px=True)
print("M5 lagrange optimum", round(xo, 4), round(yo, 4), "lambda", round(lam, 4), "cos angle", round(-(gf @ gg) / np.linalg.norm(gf) / np.linalg.norm(gg), 6))
p.save(OUT / "m5-opt.svg", "Left: a convex function lies below its chords and above its tangents. Middle: a non-convex function has a chord that dips below it and two local minima. Right: minimising f on a circle, the optimum is where a level set of f touches the circle and the two gradients are parallel")


# ---------- M6: Monte Carlo error against n; Metropolis-Hastings traces ----------
import contextlib, io, runpy  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    ns6 = runpy.run_path(str(OUT.parent / "code" / "M6-sampling-and-monte-carlo.py"))
p = Plot(900, 330)
p.panel(0, 0, 400, 330, (1.8, 5.2), (-3.5, 0), margin=(40, 16, 44, 52))
p.axes("log₁₀ n", "log₁₀ RMS error", xticks=[2, 3, 4, 5], yticks=[-3, -2, -1, 0])
lx = np.log10(ns6["ns"])
for d, col in [(1, "data"), (10, "accent")]:
    ly = np.log10(ns6["mc_err"][d])
    p.line(lx, ly, color=col, width=1.8); p.dots(lx, ly, color=col)
p.text(2.9, -0.75, "d = 10", color="accent", size=10.5)
p.text(2.9, -2.05, "d = 1", color="data", size=10.5)
p.text(3.6, -3.2, "both slopes −1/2", size=10)
p.text(p.box[0], 22, "Monte Carlo error does not depend on d", color="fg", size=12, px=True)
p.panel(420, 0, 480, 330, (0, 5000), (-38, 9), margin=(40, 16, 44, 46))
p.axes("step", "", xticks=[0, 1000, 2000, 3000, 4000, 5000], yticks=[], grid=False)
for k, (w, col) in enumerate([(0.1, "data"), (2.5, "accent"), (50.0, "fg")]):
    off = -15 * k
    ch = ns6["chains"][w][:5000]
    p.line([0, 5000], [off + 3, off + 3], color="muted", width=0.8, dash="3 3")
    p.line([0, 5000], [off - 3, off - 3], color="muted", width=0.8, dash="3 3")
    p.line(np.arange(5000)[::5], np.clip(ch[::5], -6.5, 6.5) + off, color=col, width=1.0)
    p.text(4950, off + 7.2, f"proposal width {w:g}", color=col, size=10, anchor="end")
p.text(p.box[0], 22, "Metropolis–Hastings on two modes at ±3", color="fg", size=12, px=True)
p.save(OUT / "m6-sampling.svg", "Left: RMS error of plain Monte Carlo estimates of an integral against the number of samples on log-log axes, in 1 and in 10 dimensions; both lines have slope minus one half, the 10-dimensional one higher because its integrand has larger variance. Right: the first 5,000 steps of Metropolis-Hastings chains on a target with modes at plus and minus 3, for random-walk proposals of width 0.1, 2.5 and 50; the narrow chain never leaves its first mode, the middle one moves between the modes often, and the wide one rarely moves at all")


# ---------- 42: the landmark papers by year and stage ----------
import re  # noqa: E402
html42 = (OUT.parent / "lessons" / "42-landmark-papers.html").read_text()
stages42 = []
for head, body in re.findall(r"<h3>(.*?)</h3>\s*<div class=\"tablewrap\"><table>(.*?)</table>", html42, re.S):
    years = [int(y) for cell in re.findall(r"<tr><td>(.*?)</td>", body) for y in re.findall(r"\((?:[^()]*?)(19\d\d|20\d\d)\)", cell)]
    stages42.append((head, years))
short42 = ["1–2 foundations", "3 trees", "4 neural nets", "5 architectures", "6 generative", "7 language models", "8 RL"]
stages42 = [(s, y) for s, (_, y) in zip(short42, stages42)]
p = Plot(900, 300)
p.panel(0, 0, 900, 300, (1948, 2027), (len(stages42) - 0.4, -0.6), margin=(30, 16, 40, 118))
p.axes("year", "", xticks=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], yticks=[], grid=False)
for k, (name, years) in enumerate(stages42):
    p.line([1948, 2027], [k, k], color="muted", width=0.6, dash="2 4")
    p.dots(years, [k] * len(years), color="accent" if k % 2 else "data", r=3.6, opacity=0.8)
    p.text(1945.5, k + 0.12, name, color="fg", size=10.5, anchor="end")
allyears = sorted(y for _, ys in stages42 for y in ys)
print("42 timeline:", len(allyears), "papers;", sum(y >= 2010 for y in allyears), "from 2010 on;", [(n, len(y)) for n, y in stages42])
p.save(OUT / "paper-timeline.svg", "Each dot is a paper from the tables in this lesson, placed by year along a row for its stage. The theory and optimisation rows start in the 1950s; the rows for deep architectures, generative models and language models are almost entirely after 2010")
