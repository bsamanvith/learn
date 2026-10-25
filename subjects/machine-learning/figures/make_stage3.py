"""Figures for stage 3 (lessons 09-11). Run from the repo root:
    python3 subjects/machine-learning/figures/make_stage3.py
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


# ---------- 09: leaf boxes for three depths ----------
ns = run("09-decision-trees")
Xtr, ytr, grow, predict = ns["Xtr"], ns["ytr"], ns["grow"], ns["predict"]


def boxes(node, lo, hi, out):
    if "feature" not in node:
        out.append((lo.copy(), hi.copy(), node["predict"])); return out
    j, t = node["feature"], node["threshold"]
    h2 = hi.copy(); h2[j] = t; boxes(node["left"], lo, h2, out)
    l2 = lo.copy(); l2[j] = t; boxes(node["right"], l2, hi, out)
    return out


p = Plot(900, 310)
ang = np.linspace(0, 2 * np.pi, 200)
for i, depth in enumerate([2, 6, 20]):
    p.panel(i * 300, 0, 300, 310, (-1, 1), (-1, 1), margin=(30, 10, 10, 10))
    tree = grow(Xtr, ytr, depth)
    bx = boxes(tree, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), [])
    for lo, hi, k in bx:
        p.rect(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1], color="accent" if k else "data", opacity=0.16, stroke="rule")
    p.line(np.sqrt(0.5) * np.cos(ang), np.sqrt(0.5) * np.sin(ang), color="fg", width=1.2, dash="4 4")
    for k, col in [(0, "data"), (1, "accent")]:
        m = ytr == k
        p.dots(Xtr[m, 0], Xtr[m, 1], color=col, r=2.1, opacity=0.9)
    p.text(p.box[0], 18, f"depth {depth}: {len(bx)} leaves", color="fg", size=12, px=True)
p.save(OUT / "tree-regions.svg", "Decision regions of trees of depth 2, 6 and 20 on noisy data with a circular boundary: coarse boxes, a staircase approximation of the circle, and many tiny boxes around mislabelled points")

# ---------- 09: impurity curves; accuracy vs depth ----------
q = np.linspace(0.0005, 0.9995, 400)
mis = 1 - np.maximum(q, 1 - q)
gin = 2 * q * (1 - q)
ent = -(q * np.log2(q) + (1 - q) * np.log2(1 - q)) / 2
p = Plot(900, 300)
p.panel(0, 0, 420, 300, (0, 1), (0, 0.56), margin=(36, 14, 34, 40))
p.axes(xlabel="p = proportion of class 1", ylabel="impurity", xticks=[0, 0.5, 1], yticks=[0, 0.25, 0.5])
p.line(q, mis, color="muted", width=2); p.line(q, gin, color="accent", width=2.2); p.line(q, ent, color="data", width=2.2, dash="6 4")
p.text(0.5, 0.53, "entropy / 2", color="data", size=11, anchor="middle")
p.text(0.2, 0.37, "Gini", color="accent", size=11, anchor="end"); p.text(0.17, 0.08, "misclassification", color="muted", size=11)
p.text(p.box[0], 16, "three impurity measures (two classes)", color="fg", size=12, px=True)
Xte, yte = ns["Xte"], ns["yte"]
depths = np.arange(1, 21)
tr, te = [], []
for dd in depths:
    t = grow(Xtr, ytr, int(dd))
    tr.append(np.mean(predict(t, Xtr) == ytr)); te.append(np.mean(predict(t, Xte) == yte))
p.panel(440, 0, 460, 300, (1, 20), (0.5, 1.02), margin=(36, 14, 34, 40))
p.axes(xlabel="maximum depth", ylabel="accuracy", xticks=[1, 5, 10, 15, 20], yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
p.seg(1, 0.9, 20, 0.9, color="muted", dash="4 4")
p.line(depths, tr, color="data"); p.line(depths, te, color="accent")
p.dots(depths, tr, color="data", r=2.6); p.dots(depths, te, color="accent", r=2.6)
p.text(19.5, 0.975, "train", color="data", size=11, anchor="end"); p.text(19.5, 0.74, "test", color="accent", size=11, anchor="end")
p.text(19.5, 0.91, "best possible (10% noise)", color="muted", size=10.5, anchor="end")
p.text(p.box[0], 16, "train and test accuracy against depth", color="fg", size=12, px=True)
print("09 test acc by depth", np.round(te, 3))
p.save(OUT / "tree-impurity.svg", "Left: misclassification, Gini and half the entropy as functions of the class proportion. Right: training accuracy rises to 1 with depth while test accuracy peaks around depth 6")


# ---------- 10: variance floor; accuracy vs number of trees ----------
ns = run("10-random-forests")
p = Plot(900, 300)
p.panel(0, 0, 400, 300, (1, 100), (0, 1.02), margin=(36, 14, 34, 44))
p.axes(xlabel="number of models B", ylabel="Var(average) / σ²", xticks=[1, 25, 50, 75, 100], yticks=[0, 0.25, 0.5, 0.75, 1])
Bs = np.arange(1, 101)
for rho, col in [(0.0, "data"), (0.2, "accent"), (0.36, "c"), (0.6, "muted")]:
    p.line(Bs, rho + (1 - rho) / Bs, color=col, width=2)
    p.text(98, rho + (1 - rho) / 100 + 0.025, f"ρ = {rho:g}", color=col, size=11, anchor="end")
p.text(p.box[0], 16, "averaging B correlated predictors", color="fg", size=12, px=True)
Xtr, ytr, Xte, yte, pte = ns["Xtr"], ns["ytr"], ns["Xte"], ns["yte"], ns["pte"]
p.panel(420, 0, 480, 300, (1, 100), (0.65, 0.9), margin=(36, 14, 34, 44))
p.axes(xlabel="number of trees", ylabel="test accuracy", xticks=[1, 25, 50, 75, 100], yticks=[0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
best = np.mean(np.maximum(pte, 1 - pte))
p.seg(1, best, 100, best, color="muted", dash="4 4"); p.text(99, best - 0.012, f"best possible {best:.3f}", color="muted", size=10.5, anchor="end")
for m, col, lab in [(10, "data", "bagging"), (3, "accent", "random forest")]:
    trees, _ = ns["forest"](Xtr, ytr, 100, m)
    V = ns["votes"](trees, Xte)
    cum = np.cumsum(V, 0) / Bs[:, None]
    acc = [np.mean(np.where(cum[b] == 0.5, 0, cum[b] > 0.5) == yte) for b in range(100)]
    p.line(Bs, acc, color=col, width=2)
    p.text(99, acc[-1] + (0.008 if m == 3 else -0.015), f"{lab} {acc[-1]:.3f}", color=col, size=11, anchor="end")
    print("10", lab, np.round(np.array(acc)[[0, 9, 99]], 3))
p.text(p.box[0], 16, "test accuracy as trees are added", color="fg", size=12, px=True)
p.save(OUT / "forest.svg", "Left: the variance of an average of correlated predictors falls toward a floor equal to the correlation. Right: test accuracy of bagging and of a random forest as trees are added; the forest ends higher")


# ---------- 11: stumps on a sine curve; test loss against rounds ----------
ns = run("11-gradient-boosting")
build, tree_predict = ns["build"], ns["tree_predict"]
r = np.random.default_rng(3)
xs = np.sort(r.uniform(0, 1, 80)); ys = np.sin(2 * np.pi * xs) + r.normal(0, 0.3, 80)
X1 = xs[:, None]; grid = np.linspace(0, 1, 400); G1 = grid[:, None]
p = Plot(900, 310)
p.panel(0, 0, 420, 310, (0, 1), (-1.9, 1.9), margin=(36, 14, 34, 40))
p.axes(xlabel="x", xticks=[0, 0.5, 1], yticks=[-1.5, 0, 1.5])
p.line(grid, np.sin(2 * np.pi * grid), color="muted", width=1.2, dash="4 4")
p.dots(xs, ys, color="fg", r=2.2, opacity=0.55)
F, Fg = np.full(80, ys.mean()), np.full(400, ys.mean())
cols = {1: "c", 10: "data", 100: "accent"}
for m in range(1, 101):
    t = build(X1, F - ys, np.ones(80), 1, 0.0, 0.0)      # squared loss: g = F - y, h = 1, so leaves are mean residuals
    F += 0.3 * tree_predict(t, X1); Fg += 0.3 * tree_predict(t, G1)
    if m in cols:
        p.line(grid, Fg, color=cols[m], width=2)
for i, (m, col) in enumerate(cols.items()):
    p.text(0.98, 1.75 - 0.28 * i, f"M = {m}", color=col, size=11, anchor="end")
p.text(p.box[0], 16, "stumps, squared loss, ν = 0.3", color="fg", size=12, px=True)

Xtr, ytr, Xte, yte = ns["Xtr"], ns["ytr"], ns["Xte"], ns["yte"]
sig, ll = ns["sigmoid"], ns["logloss"]
R = 1500
p.panel(440, 0, 460, 310, (0, np.log10(R)), (0, 0.9), margin=(36, 14, 34, 44))
p.axes(xlabel="number of trees (log scale)", ylabel="log-loss", yticks=[0, 0.2, 0.4, 0.6, 0.8])
for v in [1, 10, 100, 1000]:
    p.seg(np.log10(v), 0, np.log10(v), 0.9, color="rule", width=1)
    p.text(p.X(np.log10(v)), p.box[1] + p.box[3] + 15, f"{v:,}", size=11, anchor="middle", px=True)
for lr, col in [(1.0, "c"), (0.1, "accent")]:
    F0 = np.log(ytr.mean() / (1 - ytr.mean()))
    Ftr, Fte = np.full(len(ytr), F0), np.full(len(yte), F0)
    tr, te = [], []
    for _ in range(R):
        pp = sig(Ftr)
        t = build(Xtr, pp - ytr, pp * (1 - pp), 3, 1.0, 0.0)
        Ftr += lr * tree_predict(t, Xtr); Fte += lr * tree_predict(t, Xte)
        tr.append(ll(ytr, Ftr)); te.append(ll(yte, Fte))
    k = np.log10(np.arange(1, R + 1)); te = np.array(te)
    p.line(k, np.minimum(tr, 0.9), color=col, width=1.4, dash="5 4"); p.line(k, np.minimum(te, 0.9), color=col, width=2.2)
    b = int(np.argmin(te)); p.dots([k[b]], [te[b]], color=col, r=3.6)
    print(f"11 lr={lr}: min test {te[b]:.3f} at {b + 1} trees, final {te[-1]:.3f}")
    p.text(k[-1], min(te[-1], 0.9) + (0.03 if lr == 1 else -0.035), f"ν = {lr:g} test", color=col, size=11, anchor="end")
p.text(p.X(0.05), p.Y(0.06), "dashed: training loss", color="muted", size=10.5, px=True)
p.text(p.box[0], 16, "test log-loss as trees are added", color="fg", size=12, px=True)
p.save(OUT / "boosting.svg", "Left: boosting stumps on noisy samples of a sine curve; after 1, 10 and 100 rounds the fit goes from a single step to the curve's shape to chasing noise. Right: training and test log-loss against the number of trees for learning rates 1 and 0.1; the larger rate bottoms out early and overfits sharply, the smaller one reaches a lower minimum later")
