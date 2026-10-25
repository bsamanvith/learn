"""Problem book assignment — watch logistic regression run away.

Logistic regression by gradient descent on two 2D datasets: one linearly separable, one
not. On separable data the loss has no minimiser, the weights grow like log t forever, and
their direction creeps towards the maximum-margin separator (Soudry et al., 2018). On
overlapping data the weights settle. The bias is folded in as a third feature, so the
"maximum-margin separator" is the hard-margin SVM on (x1, x2, 1) without a separate bias.

Run: .venv/bin/python subjects/machine-learning/sims/runaway_logistic.py [--plot]
"""

import sys
import numpy as np

rng = np.random.default_rng(0)
n = 40


def blobs(gap):
    """n points per class around (±gap, ±gap), labels ±1, with a column of ones for the bias."""
    pos = rng.normal(0, 1, (n, 2)) + gap
    neg = rng.normal(0, 1, (n, 2)) - gap
    X = np.r_[pos, neg]
    return np.c_[X, np.ones(2 * n)], np.r_[np.ones(n), -np.ones(n)]


def loss_grad(w, X, y):
    m = y * (X @ w)                              # margins
    loss = np.mean(np.logaddexp(0, -m))
    g = -(X * (y / (1 + np.exp(m)))[:, None]).mean(0)
    return loss, g


def max_margin(X, y, iters=200_000):
    """Hard-margin SVM through the origin of the augmented space, by coordinate ascent on the dual."""
    Z = X * y[:, None]
    K = Z @ Z.T
    a = np.zeros(len(y))
    for _ in range(iters // len(y)):
        for i in range(len(y)):
            a[i] = max(0.0, a[i] + (1 - K[i] @ a) / K[i, i])
    w = Z.T @ a
    return w, 1 / np.linalg.norm(w)              # w has margin 1; geometric margin 1/‖w‖


def angle(u, v):
    c = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def run(X, y, eta, steps, checkpoints, lam=0.0):
    w = np.zeros(X.shape[1])
    out = {}
    for t in range(1, steps + 1):
        loss, g = loss_grad(w, X, y)
        w -= eta * (g + lam * w)
        if t in checkpoints:
            out[t] = (loss, w.copy())
    return out


CHECK = [10, 100, 1_000, 10_000, 100_000]

# Step 1 · separable data.
X, y = blobs(gap=2.0)
w_svm, gamma = max_margin(X, y)
print(f"Step 1 · separable blobs: {2 * n} points; max-margin separator has margin γ = {gamma:.3f}, ‖ŵ‖ = {np.linalg.norm(w_svm):.3f}")
H = X.T @ X / len(y) / 4
print(f"         logistic smoothness bound L = λ_max(XᵀX)/(4n) = {np.linalg.eigvalsh(H).max():.3f}, so take η = 1/L ≈ 0.4")
res = run(X, y, eta=0.4, steps=max(CHECK), checkpoints=set(CHECK))
for t in CHECK:
    loss, w = res[t]
    print(f"         t = {t:>7,}: loss = {loss:.2e}   ‖w‖ = {np.linalg.norm(w):6.2f}   angle to max-margin = {angle(w, w_svm):5.2f}°")
n4, n5 = (np.linalg.norm(res[t][1]) for t in (10_000, 100_000))
print(f"         growth per factor of 10 in t, last decade: {n5 - n4:.2f}   (‖ŵ‖·ln 10 = {np.linalg.norm(w_svm) * np.log(10):.2f})")

# Step 2 · overlapping data.
Xo, yo = blobs(gap=0.6)
reso = run(Xo, yo, eta=0.4, steps=max(CHECK), checkpoints=set(CHECK))
print("\nStep 2 · overlapping blobs:")
for t in CHECK:
    loss, w = reso[t]
    print(f"         t = {t:>7,}: loss = {loss:.4f}   ‖w‖ = {np.linalg.norm(w):.4f}")

# Step 3 · break the runaway with a little weight decay.
print("\nStep 3 · separable data with weight decay λ‖w‖²/2:")
for lam in (1e-2, 1e-3, 1e-4):
    r = run(X, y, eta=0.4, steps=100_000, checkpoints={100_000}, lam=lam)[100_000][1]
    print(f"         λ = {lam:.0e}: ‖w‖ = {np.linalg.norm(r):6.2f}   angle to max-margin = {angle(r, w_svm):5.2f}°")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(*X[y > 0, :2].T, s=12, label="+1")
    ax[0].scatter(*X[y < 0, :2].T, s=12, label="−1")
    xs = np.linspace(-5, 5, 2)
    for t in CHECK:
        w = res[t][1]
        ax[0].plot(xs, -(w[0] * xs + w[2]) / w[1], lw=1, alpha=0.4 + 0.12 * CHECK.index(t), label=f"t = {t:,}")
    ax[0].plot(xs, -(w_svm[0] * xs + w_svm[2]) / w_svm[1], "k--", lw=1, label="max margin")
    ax[0].set(xlim=(-5, 5), ylim=(-5, 5), title="The boundary drifts to the max-margin line")
    ax[0].legend(fontsize=7)
    ts = np.array(CHECK)
    ax[1].semilogx(ts, [np.linalg.norm(res[t][1]) for t in CHECK], "o-", label="separable")
    ax[1].semilogx(ts, [np.linalg.norm(reso[t][1]) for t in CHECK], "o-", label="overlapping")
    ax[1].set(xlabel="step t (log scale)", ylabel="‖w‖", title="A straight line on a log axis: ‖w‖ ∝ log t")
    ax[1].legend()
    plt.tight_layout()
    plt.show()
