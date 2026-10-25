"""Problem book assignment — overfit on purpose, then cure it.

Twenty noisy points from y = sin(2πx) + noise. Fit polynomials of degree 0 to 19 by least
squares and watch training error fall to zero while test error explodes. Then keep the
degree-19 model and add a ridge penalty, choosing λ by leave-one-out cross-validation with
the closed-form shortcut (y_i − ŷ_i)/(1 − H_ii), no refitting.

Features are Chebyshev polynomials of 2x − 1, so the design matrix stays well-conditioned
enough for degree 19; the model class is the same as plain powers of x.

Run: .venv/bin/python subjects/machine-learning/sims/overfit.py [--plot]
"""

import sys
import numpy as np
from numpy.polynomial import chebyshev as C

rng = np.random.default_rng(1)
n, sigma = 20, 0.3
f = lambda x: np.sin(2 * np.pi * x)
x = np.sort(rng.uniform(0, 1, n))
y = f(x) + rng.normal(0, sigma, n)
xt = rng.uniform(0, 1, 2000)                    # fresh test inputs
yt = f(xt) + rng.normal(0, sigma, xt.size)
feats = lambda x, d: C.chebvander(2 * x - 1, d)
mse = lambda a, b: np.mean((a - b) ** 2)
print(f"Step 0 · {n} training points, noise σ = {sigma} (so the best possible test MSE is σ² = {sigma**2:.2f}); 2000 test points")

# Step 1 · least squares for each degree.
print("\nStep 1 · least squares by degree")
fits = {}
for d in [0, 1, 3, 5, 7, 9, 12, 15, 19]:
    w = np.linalg.lstsq(feats(x, d), y, rcond=None)[0]
    fits[d] = w
    print(f"         degree {d:>2}: train MSE = {mse(feats(x, d) @ w, y):8.4f}   test MSE = {mse(feats(xt, d) @ w, yt):12.4f}   ‖w‖ = {np.linalg.norm(w):10.1f}")

# Step 2 · keep degree 19 and add a ridge penalty on the coefficients of 1, x, …, x¹⁹,
# choosing λ by leave-one-out. Solve with an SVD: the power basis is badly conditioned.
d = 19
X, Xt = np.vander(x, d + 1, increasing=True), np.vander(xt, d + 1, increasing=True)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
lams = 10.0 ** np.arange(-14, 1.01, 1)
print("\nStep 2 · ridge on degree 19 (power basis)")
rows = []
for lam in lams:
    shrink = S / (S**2 + lam)
    w = Vt.T @ (shrink * (U.T @ y))
    h = np.sum(U**2 * (S**2 / (S**2 + lam)), axis=1)      # diagonal of the hat matrix
    loo = np.mean(((y - X @ w) / (1 - h)) ** 2)
    rows.append((lam, mse(X @ w, y), loo, mse(Xt @ w, yt), h.sum()))
for lam, tr, loo, te, df in rows[::2]:
    print(f"         λ = {lam:6.0e}: train {tr:.4f}   LOO {loo:8.4f}   test {te:8.4f}   effective parameters tr(H) = {df:5.2f}")
best = min(rows, key=lambda r: r[2])
oracle = min(rows, key=lambda r: r[3])
print(f"         LOO picks λ = {best[0]:.0e}: test MSE {best[3]:.4f} with {best[4]:.1f} effective parameters")
print(f"         (the best λ in hindsight, {oracle[0]:.0e}, gets {oracle[3]:.4f})")

# Step 3 · or let leave-one-out choose the degree instead.
loo_deg = {}
for dd in range(0, 16):
    F = feats(x, dd)
    H = F @ np.linalg.pinv(F)
    loo_deg[dd] = np.mean(((y - H @ y) / (1 - np.diag(H))) ** 2)
dbest = min(loo_deg, key=loo_deg.get)
wd = np.linalg.lstsq(feats(x, dbest), y, rcond=None)[0]
print(f"\nStep 3 · LOO over degrees 0–15 picks degree {dbest} (LOO {loo_deg[dbest]:.4f}): test MSE {mse(feats(xt, dbest) @ wd, yt):.4f}")

# Step 4 · the same with 200 points: more data tames degree 19 without any penalty.
x2 = rng.uniform(0, 1, 200)
y2 = f(x2) + rng.normal(0, sigma, 200)
w2 = np.linalg.lstsq(feats(x2, 19), y2, rcond=None)[0]
print(f"\nStep 4 · degree 19, no penalty, 200 points: train {mse(feats(x2, 19) @ w2, y2):.4f}   test {mse(feats(xt, 19) @ w2, yt):.4f}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    g = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for dd in (1, 3, 19):
        ax[0].plot(g, feats(g, dd) @ fits[dd], label=f"degree {dd}")
    ax[0].plot(x, y, "ko", ms=4)
    ax[0].plot(g, f(g), "k:", lw=1, label="truth")
    ax[0].set(ylim=(-2, 2), title="Wiggles through every point")
    ax[0].legend(fontsize=8)
    degs = sorted(fits)
    ax[1].semilogy(degs, [mse(feats(x, k) @ fits[k], y) + 1e-12 for k in degs], "o-", label="train")
    ax[1].semilogy(degs, [mse(feats(xt, k) @ fits[k], yt) for k in degs], "o-", label="test")
    ax[1].set(xlabel="degree", ylabel="MSE", title="The U that isn't a U on the training side")
    ax[1].legend()
    L = np.array(rows)
    ax[2].loglog(L[:, 0], L[:, 2], label="leave-one-out")
    ax[2].loglog(L[:, 0], L[:, 3], label="test")
    ax[2].loglog(L[:, 0], L[:, 1], label="train")
    ax[2].axvline(best[0], color="k", ls=":")
    ax[2].set(xlabel="λ", title="LOO finds the valley without a test set")
    ax[2].legend()
    plt.tight_layout()
    plt.show()
