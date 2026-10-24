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
