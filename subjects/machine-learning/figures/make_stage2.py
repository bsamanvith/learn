"""Figures for stage 2 (lessons 04-08). Run from the repo root:
    python3 subjects/machine-learning/figures/make_stage2.py
Every number drawn here is computed, not sketched."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.figures.svgplot import Plot, c  # noqa: E402

OUT = Path(__file__).resolve().parent
f = lambda x: np.sin(2 * np.pi * x)
SIG = 0.3


def polyfit(x, y, deg, lam=0.0):
    X = np.vander(2 * x - 1, deg + 1, increasing=True)
    A = X.T @ X + lam * np.eye(deg + 1)
    return np.linalg.solve(A, X.T @ y) if lam else np.linalg.lstsq(X, y, rcond=None)[0]


def polyval(w, x):
    return np.vander(2 * x - 1, len(w), increasing=True) @ w


# ---------- 04: three fits ----------
# Fixed design: 15 evenly spaced inputs; each "dataset" redraws only the noise.
rng = np.random.default_rng(3)
n = 15
x = (np.arange(n) + 0.5) / n
y = f(x) + SIG * rng.standard_normal(n)
xs = np.linspace(0, 1, 400)
p = Plot(900, 270)
for i, (deg, name) in enumerate([(1, "degree 1: underfits"), (3, "degree 3: about right"), (12, "degree 12: overfits")]):
    p.panel(i * 300, 0, 300, 270, (0, 1), (-1.8, 1.8), margin=(26, 10, 24, 30))
    p.axes(xticks=[0, 1], yticks=[-1, 0, 1], grid=True)
    p.line(xs, f(xs), color="muted", width=1.4, dash="4 4")
    yy = np.clip(polyval(polyfit(x, y, deg), xs), -1.8, 1.8)
    p.line(xs, yy, color="accent", width=2.4)
    p.dots(x, y)
    p.text(p.box[0], 16, name, color="fg", size=12, px=True)
p.save(OUT / "gen-fits.svg", "Polynomial fits of degree 1, 3 and 12 to the same 15 noisy points from a sine curve")

# ---------- 04: train vs test, bias^2 / variance ----------
degs = np.arange(0, 13)
R = 4000
xt = np.linspace(0.02, 0.98, 300)
bias2, var, train = [], [], []
for d in degs:
    P, tr = [], []
    for r in range(R):
        yr = f(x) + SIG * rng.standard_normal(n)
        w = polyfit(x, yr, d)
        P.append(polyval(w, xt)); tr.append(np.mean((polyval(w, x) - yr) ** 2))
    P = np.array(P)
    bias2.append(((P.mean(0) - f(xt)) ** 2).mean()); var.append(P.var(0).mean()); train.append(np.mean(tr))
bias2, var, train = map(np.array, (bias2, var, train))
test = bias2 + var + SIG**2
print("degree  train  theory  bias2  var  test")
for i, d in enumerate(degs):
    print(d, round(train[i], 4), round(SIG**2 * (1 - (d + 1) / n), 4), round(bias2[i], 4), round(var[i], 4), round(test[i], 4))

TOP = 0.7
cl = lambda a: np.minimum(a, TOP)
p = Plot(900, 300)
p.panel(0, 0, 440, 300, (0, 12), (0, TOP), margin=(26, 14, 34, 40))
p.axes(xlabel="polynomial degree", ylabel="mean squared error", xticks=[0, 3, 6, 9, 12], yticks=[0, 0.1, 0.3, 0.5, 0.7])
p.line(degs, cl(test), color="accent")
p.line(degs, train, color="data")
p.dots(degs, cl(test), color="accent", r=3)
p.dots(degs, train, color="data", r=3)
p.seg(0, SIG**2, 12, SIG**2, color="muted", dash="4 4")
p.text(0.2, SIG**2 - 0.02, "noise σ² = 0.09", size=11)
p.text(9.6, 0.25, "test", color="accent", size=12)
p.text(6.0, 0.012, "train", color="data", size=12)
p.panel(460, 0, 440, 300, (0, 12), (0, TOP), margin=(26, 14, 34, 40))
p.axes(xlabel="polynomial degree", ylabel="expected test error, decomposed", xticks=[0, 3, 6, 9, 12], yticks=[0, 0.1, 0.3, 0.5, 0.7])
p.area(degs, np.full(len(degs), SIG**2), np.zeros(len(degs)), color="muted", opacity=0.12)
p.line(degs, cl(bias2), color="data")
p.line(degs, cl(var), color="c")
p.line(degs, cl(test), color="accent", width=2.6)
p.text(0.4, 0.5, "bias²", color="data", size=12)
p.text(11.6, 0.24, "variance", color="c", size=12, anchor="end")
p.text(3.6, 0.17, "bias² + variance + σ²", color="accent", size=12)
p.text(6, 0.04, "irreducible noise σ²", color="muted", size=11, anchor="middle")
p.save(OUT / "gen-curves.svg", "Left: training error falls with degree while test error falls then rises. Right: bias squared falls, variance rises, and their sum plus noise is U-shaped with its minimum at degree 3")


# ---------- 05: ridge on the degree-12 fit ----------
def ridge_poly(x, y, deg, lam):
    X = np.vander(2 * x - 1, deg + 1, increasing=True)
    D = np.eye(deg + 1); D[0, 0] = 0          # don't penalise the intercept
    return np.linalg.solve(X.T @ X + lam * D, X.T @ y)


p = Plot(900, 270)
for i, (lam, name) in enumerate([(0.0, "λ = 0: overfits"), (1e-2, "λ = 0.01: about right"), (1.0, "λ = 1: too smooth")]):
    p.panel(i * 300, 0, 300, 270, (0, 1), (-1.8, 1.8), margin=(26, 10, 24, 30))
    p.axes(xticks=[0, 1], yticks=[-1, 0, 1], grid=True)
    p.line(xs, f(xs), color="muted", width=1.4, dash="4 4")
    w = polyfit(x, y, 12) if lam == 0 else ridge_poly(x, y, 12, lam)
    p.line(xs, np.clip(polyval(w, xs), -1.8, 1.8), color="accent", width=2.4)
    p.dots(x, y)
    p.text(p.box[0], 16, name, color="fg", size=12, px=True)
    print("ridge", lam, "||w|| =", round(float(np.linalg.norm(w[1:])), 2))
p.save(OUT / "reg-fits.svg", "Degree-12 polynomial fitted to the same 15 points with ridge penalties 0, 0.001 and 1")

# ---------- 05: L2 ball vs L1 diamond ----------
H = np.array([[1.0, 0.6], [0.6, 0.8]])
wstar = np.array([1.6, 0.55])
loss = lambda w: 0.5 * (w - wstar) @ H @ (w - wstar)
t = 1.0
ang = np.linspace(0, 2 * np.pi, 4001)
circle = np.stack([t * np.cos(ang), t * np.sin(ang)], 1)
diamond = np.stack([np.cos(ang), np.sin(ang)], 1)
diamond = t * diamond / np.abs(diamond).sum(1, keepdims=True)
p = Plot(900, 340)
for i, (B, name) in enumerate([(circle, "ridge: ‖w‖₂ ≤ t"), (diamond, "lasso: ‖w‖₁ ≤ t")]):
    p.panel(i * 450, 0, 450, 340, (-1.6, 2.6), (-1.4, 1.9), margin=(26, 20, 20, 20))
    p.seg(-1.6, 0, 2.6, 0, color="rule", width=1); p.seg(0, -1.4, 0, 1.9, color="rule", width=1)
    p.text(2.55, -0.12, "w₁", anchor="end", size=12); p.text(0.06, 1.8, "w₂", size=12)
    vals = np.array([loss(b) for b in B])
    wb = B[np.argmin(vals)]
    p.area(B[:, 0], B[:, 1], B[:, 1], color="data", opacity=0.0)
    p.raw('<polygon points="' + " ".join(f"{p.X(a):.1f},{p.Y(b):.1f}" for a, b in B[::20]) +
          '" style="fill:var(--event-a);fill-opacity:.14;stroke:var(--event-a);stroke-width:1.6"/>')
    # loss contours (ellipses) through the constrained optimum and two more levels
    L, V = np.linalg.eigh(H)
    for lev in [loss(wb) * k for k in (0.4, 1.0, 1.6)]:
        r = np.sqrt(2 * lev / L)
        e = wstar + (np.stack([r[0] * np.cos(ang), r[1] * np.sin(ang)], 1) @ V.T)
        p.line(e[::10, 0], e[::10, 1], color="accent", width=1.3 if lev != loss(wb) else 2.0, opacity=0.9)
    p.dots([wstar[0]], [wstar[1]], color="accent", r=4)
    p.text(wstar[0] + 0.08, wstar[1] + 0.06, "least squares ŵ", color="accent", size=11)
    p.dots([wb[0]], [wb[1]], color="fg", r=4.5)
    lab = f"({wb[0]:.2f}, {wb[1]:.2f})"
    p.text(wb[0] - 0.08, wb[1] - 0.22, lab, color="fg", size=11, anchor="end")
    p.text(p.box[0], 16, name, color="fg", size=12, px=True)
    print("constrained optimum", name, np.round(wb, 3))
p.save(OUT / "reg-geometry.svg", "Loss contours meeting an L2 ball at a point with both coordinates nonzero, and an L1 diamond at a corner where one coordinate is exactly zero")

# ---------- 05: coefficient paths ----------
rng2 = np.random.default_rng(7)
N, P = 60, 8
Z = rng2.standard_normal((N, P))
Z[:, 1] = 0.7 * Z[:, 0] + 0.7 * Z[:, 1]           # two correlated features
Z = (Z - Z.mean(0)) / Z.std(0)
beta = np.array([3.0, 0.0, -2.0, 0.0, 1.5, 0.0, 0.0, 0.0])
yy = Z @ beta + 1.0 * rng2.standard_normal(N)
yy = yy - yy.mean()


def lasso_cd(X, y, lam, iters=500, w=None):
    """Minimise (1/2n)||y - Xw||^2 + lam ||w||_1 by coordinate descent."""
    n_, p_ = X.shape
    w = np.zeros(p_) if w is None else w.copy()
    sq = (X ** 2).sum(0) / n_
    r = y - X @ w
    for _ in range(iters):
        for j in range(p_):
            r += X[:, j] * w[j]
            rho = X[:, j] @ r / n_
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / sq[j]
            r -= X[:, j] * w[j]
    return w


lams_r = np.logspace(-2, 4, 80)
U, d, Vt = np.linalg.svd(Z, full_matrices=False)
ridge_path = np.array([Vt.T @ (d / (d**2 + l) * (U.T @ yy)) for l in lams_r])
lmax = np.max(np.abs(Z.T @ yy)) / N
lams_l = lmax * np.logspace(0, -3, 80)
w = np.zeros(P); lasso_path = []
for l in lams_l:
    w = lasso_cd(Z, yy, l, iters=300, w=w); lasso_path.append(w.copy())
lasso_path = np.array(lasso_path)
cols = ["accent", "fg", "data", "muted", "c", "muted", "muted", "muted"]
p = Plot(900, 310)
p.panel(0, 0, 440, 310, (-2, 4), (-2.6, 3.4), margin=(26, 14, 34, 40))
p.axes(xlabel="log₁₀ λ  (more shrinkage →)", ylabel="ridge coefficients", xticks=[-2, 0, 2, 4], yticks=[-2, 0, 2], zero=True)
for j in range(P):
    p.line(np.log10(lams_r), ridge_path[:, j], color=cols[j], width=2 if beta[j] or j == 1 else 1.2, opacity=1 if beta[j] or j == 1 else 0.7)
p.panel(460, 0, 440, 310, (0, 1), (-2.6, 3.4), margin=(26, 14, 34, 40))
p.axes(xlabel="‖w‖₁ / max ‖w‖₁  (less shrinkage →)", ylabel="lasso coefficients", xticks=[0, 0.5, 1], yticks=[-2, 0, 2], zero=True)
s = np.abs(lasso_path).sum(1); s = s / s.max()
for j in range(P):
    p.line(s, lasso_path[:, j], color=cols[j], width=2 if beta[j] or j == 1 else 1.2, opacity=1 if beta[j] or j == 1 else 0.7)
for j, lab in [(0, "w₁ (true 3)"), (2, "w₃ (true −2)"), (4, "w₅ (true 1.5)"), (1, "w₂ (true 0, correlated with w₁)")]:
    p.text(1.0, lasso_path[-1, j] + (0.18 if j != 1 else -0.25), lab, color=cols[j], size=10.5, anchor="end")
first = [int(np.argmax(np.abs(lasso_path[:, j]) > 1e-9)) if np.any(np.abs(lasso_path[:, j]) > 1e-9) else None for j in range(P)]
print("lasso entry order (path index):", first)
print("ridge at lam=0.01:", np.round(ridge_path[0], 2))
p.save(OUT / "reg-paths.svg", "Coefficient paths: ridge coefficients shrink smoothly toward zero together, lasso coefficients enter one at a time and the irrelevant ones stay exactly zero")


# ---------- 06: softmax decision regions (same data and training as the lesson's code) ----------
rng6 = np.random.default_rng(1)
K6 = 3
centres = np.array([[-2.0, 0.0], [2.0, 0.5], [0.0, 2.5]])
X6 = np.vstack([cc + rng6.standard_normal((100, 2)) for cc in centres])
y6 = np.repeat(np.arange(K6), 100)
Y6 = np.eye(K6)[y6]


def log_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


W6, b6 = np.zeros((K6, 2)), np.zeros(K6)
for _ in range(500):
    G = (np.exp(log_softmax(X6 @ W6.T + b6)) - Y6) / len(y6)
    W6 -= 0.5 * G.T @ X6; b6 -= 0.5 * G.sum(0)


def clip_poly(poly, a, c0):
    """Keep the part of a convex polygon where a·x + c0 >= 0 (Sutherland-Hodgman, one edge)."""
    out = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a @ P + c0, a @ Q + c0
        if fp >= 0: out.append(P)
        if (fp >= 0) != (fq >= 0):
            out.append(P + (Q - P) * fp / (fp - fq))
    return out


XL, YL = (-5.5, 5.5), (-3.5, 6)
box = [np.array(v, float) for v in [(XL[0], YL[0]), (XL[1], YL[0]), (XL[1], YL[1]), (XL[0], YL[1])]]
cls = ["accent", "data", "c"]
p = Plot(900, 360)
p.panel(0, 0, 480, 360, XL, YL, margin=(40, 14, 30, 34))
for k in range(K6):
    reg = box
    for j in range(K6):
        if j != k:
            reg = clip_poly(reg, W6[k] - W6[j], b6[k] - b6[j])
    p.raw('<polygon points="' + " ".join(f"{p.X(a):.1f},{p.Y(b):.1f}" for a, b in reg) +
          f'" style="fill:{c(cls[k])};fill-opacity:.10;stroke:var(--fg);stroke-opacity:.5;stroke-width:1.2"/>')
p.axes(xticks=[-4, -2, 0, 2, 4], yticks=[-2, 0, 2, 4], grid=False, xlabel="x₁", ylabel="x₂")
for k in range(K6):
    m = y6 == k
    p.dots(X6[m, 0], X6[m, 1], color=cls[k], r=2.6, opacity=0.85)
p.text(p.box[0], 16, "three classes, linear boundaries z_i = z_j", color="fg", size=12, px=True)
# right panel: temperature
z = np.array([2.0, 1.0, 0.2, -1.0])
p.panel(500, 0, 400, 360, (0, 3), (0, 1), margin=(40, 10, 30, 34))
p.axes(yticks=[0, 0.25, 0.5, 0.75, 1], grid=True, ylabel="softmax(z / T)")
for i, T in enumerate([0.5, 1.0, 3.0]):
    q = np.exp(z / T); q /= q.sum()
    for k in range(4):
        p.rect(i + 0.12 + k * 0.19, 0, 0.16, q[k], color="accent" if k == 0 else "data", opacity=0.75 if k == 0 else 0.45)
    p.text(i + 0.5, -0.06, f"T = {T:g}", anchor="middle", size=11.5, color="fg")
    print("T", T, np.round(q, 3))
p.text(p.box[0], 16, "logits z = (2, 1, 0.2, −1) at three temperatures", color="fg", size=12, px=True)
p.save(OUT / "softmax.svg", "Left: decision regions of a 3-class softmax classifier are convex polygons separated by straight lines. Right: the same four logits through softmax at temperatures 0.5, 1 and 3, sharpening or flattening the distribution")
print("W6", np.round(W6, 2), "b6", np.round(b6, 2))


# ---------- 07: SGD noise floor and schedules (same problem as the lesson's code) ----------
rng7 = np.random.default_rng(0)
n7, d7, s7 = 2000, 20, 0.5
X7 = rng7.standard_normal((n7, d7)) * np.linspace(0.2, 3.0, d7)
y7 = X7 @ rng7.standard_normal(d7) + s7 * rng7.standard_normal(n7)
wb7 = np.linalg.lstsq(X7, y7, rcond=None)[0]
L7 = lambda w: 0.5 * np.mean((X7 @ w - y7) ** 2)
h7 = np.linalg.eigvalsh(X7.T @ X7 / n7)
EP, B7 = 30, 32
T7 = EP * (n7 // B7)


def curve7(kind, lr, seed):
    r = np.random.default_rng(seed)
    w, m, v, t, out = np.zeros(d7), 0, 0, 0, []
    for ep in range(EP):
        for idx in np.array_split(r.permutation(n7), n7 // B7):
            g = X7[idx].T @ (X7[idx] @ w - y7[idx]) / len(idx)
            t += 1
            if kind == "sgd":
                w = w - lr * g
            elif kind == "cos":
                w = w - lr * 0.5 * (1 + np.cos(np.pi * t / T7)) * g
            else:
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                w = w - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        out.append(L7(w) - L7(wb7))
    return np.array(out)


SEEDS = range(1, 9)
runs = [("sgd", 0.1, "SGD η = 0.1", "accent"), ("sgd", 0.01, "SGD η = 0.01", "data"),
        ("cos", 0.1, "SGD, cosine decay from 0.1", "fg"), ("adam", 0.003, "Adam η = 0.003", "c")]
sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
p = Plot(900, 330)
p.panel(0, 0, 520, 330, (1, EP), (-6, 2), margin=(40, 14, 34, 46))
p.axes(xlabel="epoch", ylabel="excess loss L(w) − L(w*)  (log scale, mean of 8 runs)", xticks=[1, 10, 20, 30],
       yticks=[-6, -4, -2, 0, 2], fmt=lambda v: ("10" + str(int(v)).translate(sup)) if v == int(v) and v <= 2 and v >= -6 and v % 2 == 0 else f"{v:g}")
ep = np.arange(1, EP + 1)
for kind, lr, lab, col in runs:
    cc = np.mean([curve7(kind, lr, s) for s in SEEDS], 0)
    p.line(ep, np.log10(cc), color=col, width=2.2)
    print("07 curve", lab, np.round(np.log10(cc[[0, 9, 29]]), 2))
for lr, col in [(0.1, "accent"), (0.01, "data")]:
    fl = lr * s7**2 / (2 * B7) * np.sum(h7 / (2 - lr * h7))
    p.seg(1, np.log10(fl), EP, np.log10(fl), color=col, dash="4 4", width=1.2)
    print("floor", lr, fl)
p.text(p.box[0], 16, "constant steps stall at a noise floor; decaying steps don't", color="fg", size=12, px=True)
yl = 226
for kind, lr, lab, col in runs:
    p.raw(f'<line x1="56" y1="{yl - 4}" x2="74" y2="{yl - 4}" style="stroke:{c(col)};stroke-width:2.2"/>')
    p.text(80, yl, lab, size=11, px=True, color="fg"); yl += 15
p.text(80, yl, "dashed: predicted floor", size=11, px=True)

# right: trajectories on an ill-conditioned quadratic
A = np.diag([1.0, 25.0])
th = np.deg2rad(30); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
A = Rm @ A @ Rm.T
q = lambda w: 0.5 * w @ A @ w
Lm, Vm = np.linalg.eigh(A)
vs = Vm[:, 0] * np.sign(-Vm[0, 0]); vt = Vm[:, 1] * np.sign(Vm[1, 1])
w0 = 4.0 * vs + 0.35 * vt                          # mostly along the shallow direction
print("w0", w0)


def traj(kind, steps=60):
    w, v, m, s, out = w0.copy(), np.zeros(2), np.zeros(2), np.zeros(2), [w0.copy()]
    lmin, lmax = 1.0, 25.0
    for t in range(1, steps + 1):
        g = A @ w
        if kind == "gd":
            w = w - 2 / (lmin + lmax) * g
        elif kind == "mom":
            eta = 4 / (np.sqrt(lmin) + np.sqrt(lmax)) ** 2
            beta = ((np.sqrt(25) - 1) / (np.sqrt(25) + 1)) ** 2
            v = beta * v + g; w = w - eta * v
        else:
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g * g
            w = w - 0.3 * (m / (1 - 0.9**t)) / (np.sqrt(s / (1 - 0.999**t)) + 1e-8)
        out.append(w.copy())
    return np.array(out)


p.panel(540, 0, 360, 330, (-4.6, 1.4), (-2.6, 1.6), margin=(40, 8, 12, 8))
for lev in [0.05, 0.4, 1.5, 4, 9, 16]:
    r = np.sqrt(2 * lev / Lm)
    e = (np.stack([r[0] * np.cos(ang), r[1] * np.sin(ang)], 1) @ Vm.T)
    p.line(e[::20, 0], e[::20, 1], color="muted", width=0.9, opacity=0.45)
for kind, col, lab in [("gd", "data", "gradient descent"), ("mom", "accent", "momentum"), ("adam", "c", "Adam")]:
    tr = traj(kind)
    p.line(tr[:30, 0], tr[:30, 1], color=col, width=1.8)
    p.dots(tr[:30, 0], tr[:30, 1], color=col, r=2.2)
    k = next((i for i, w in enumerate(tr) if q(w) < 1e-3 * q(w0)), None)
    print("07 traj", kind, "steps to 1e-3:", k)
p.dots([0], [0], color="fg", r=3.5)
p.text(p.box[0], 16, "first 30 steps on a quadratic with κ = 25", color="fg", size=12, px=True)
yl = 62
for col, lab in [("data", "gradient descent"), ("accent", "heavy-ball momentum"), ("c", "Adam")]:
    p.raw(f'<line x1="736" y1="{yl - 4}" x2="754" y2="{yl - 4}" style="stroke:{c(col)};stroke-width:2.2"/>')
    p.text(760, yl, lab, size=11, px=True, color="fg"); yl += 16
p.save(OUT / "sgd.svg", "Left: SGD with a constant step stalls at a noise floor proportional to the step size, while a cosine-decayed step keeps improving. Right: on an ill-conditioned quadratic, gradient descent zigzags, momentum cuts across, and Adam rescales each coordinate")


# ---------- 08: evaluating a classifier (runs the lesson's own code) ----------
import runpy  # noqa: E402
import contextlib, io  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(OUT.parent / "code" / "08-evaluating-classifiers.py"))
yte, s_te, a8, b8 = ns["yte"], ns["s_te"], ns["a"], ns["b"]
o = np.argsort(-s_te); ys = yte[o]
tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
tpr, fpr = tp / ys.sum(), fp / (1 - ys).sum()
prec = tp / np.arange(1, len(ys) + 1)
thr = [(0.0, "p̂ > 0.5", "accent"), (np.log(0.1 / 0.9), "p̂ > 0.1", "c")]
p = Plot(900, 300)
# panel 1: score histograms
p.panel(0, 0, 320, 300, (-20, 8), (0, 1000), margin=(40, 10, 34, 44))
bins = np.linspace(-20, 8, 57)
hn, _ = np.histogram(s_te[yte == 0], bins); hp, _ = np.histogram(s_te[yte == 1], bins)
mid = np.repeat(bins, 2)[1:-1]
p.line(mid, np.repeat(hn, 2), color="data", width=1.6)
p.line(mid, np.repeat(hp, 2), color="accent", width=1.8)
p.axes(xlabel="score (logit)", ylabel="test examples per bin", xticks=[-16, -8, 0, 8], yticks=[0, 250, 500, 750, 1000])
for s0, lab, col in thr:
    p.seg(s0, 0, s0, 1000, color=col, dash="4 3", width=1.4)
p.text(-19.5, 880, "negatives", color="data", size=11); p.text(1.0, 260, "positives", color="accent", size=11)
p.text(p.box[0], 16, "scores of the two classes", color="fg", size=12, px=True)
# panel 2: ROC
p.panel(320, 0, 290, 300, (0, 1), (0, 1), margin=(40, 10, 34, 40))
p.axes(xlabel="false positive rate", ylabel="true positive rate (recall)", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
p.seg(0, 0, 1, 1, color="muted", dash="4 4")
p.area(np.r_[0, fpr[::20], 1], np.r_[0, tpr[::20], 1], np.zeros(len(fpr[::20]) + 2), color="accent", opacity=0.08)
p.line(np.r_[0, fpr[::20], 1], np.r_[0, tpr[::20], 1], color="accent", width=2)
for s0, lab, col in thr:
    k = np.sum(s_te > s0) - 1
    p.dots([fpr[k]], [tpr[k]], color=col, r=4.5); p.text(fpr[k] + 0.03, tpr[k] - 0.05, lab, color=col, size=11)
p.text(0.55, 0.25, f"AUC = {ns['auc'](yte, s_te):.3f}", color="fg", size=12)
p.text(0.62, 0.55, "random", color="muted", size=10.5)
p.text(p.box[0], 16, "ROC curve", color="fg", size=12, px=True)
# panel 3: precision-recall
p.panel(610, 0, 290, 300, (0, 1), (0, 1), margin=(40, 10, 34, 40))
p.axes(xlabel="recall", ylabel="precision", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
p.seg(0, yte.mean(), 1, yte.mean(), color="muted", dash="4 4")
st = slice(30, None, 10)
p.line(tpr[st], prec[st], color="accent", width=2)
for s0, lab, col in thr:
    k = np.sum(s_te > s0) - 1
    p.dots([tpr[k]], [prec[k]], color=col, r=4.5); p.text(tpr[k] + 0.03, prec[k] + 0.04, lab, color=col, size=11)
p.text(0.5, yte.mean() + 0.03, f"random = base rate {yte.mean():.3f}", color="muted", size=10.5, anchor="middle")
p.text(0.55, 0.8, f"AP = {ns['average_precision'](yte, s_te):.3f}", color="fg", size=12)
p.text(p.box[0], 16, "precision–recall curve", color="fg", size=12, px=True)
p.save(OUT / "eval-curves.svg", "Score histograms for rare positives and common negatives, the ROC curve with its area, and the precision-recall curve, with two thresholds marked on each")

# reliability diagram
sig8 = lambda z: 1 / (1 + np.exp(-z))
p = Plot(900, 320)
for i, (pp, lab) in enumerate([(sig8(s_te), "raw model: overconfident"), (sig8(a8 * s_te + b8), "after Platt scaling")]):
    p.panel(i * 450, 0, 450, 320, (0, 1), (0, 1), margin=(40, 20, 34, 44))
    p.axes(xlabel="predicted probability p̂", ylabel="observed fraction positive", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    p.seg(0, 0, 1, 1, color="muted", dash="4 4")
    nb = 10
    bk = np.minimum((pp * nb).astype(int), nb - 1)
    for k in range(nb):
        m = bk == k
        if m.sum() < 20:
            continue
        p.rect(k / nb + 0.01, 0, 1 / nb - 0.02, yte[m].mean(), color="accent" if i == 0 else "data", opacity=0.55)
        p.dots([pp[m].mean()], [yte[m].mean()], color="fg", r=2.6)
    p.text(0.03, 0.92, f"ECE = {ns['ece'](yte, pp):.3f}", color="fg", size=12)
    p.text(p.box[0], 16, lab, color="fg", size=12, px=True)
p.save(OUT / "eval-calibration.svg", "Reliability diagrams: the raw model's bars fall well below the diagonal at high predicted probability, and after Platt scaling they sit on the diagonal")
