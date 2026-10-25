"""Problem book assignment — handwritten digits in pure NumPy.

A two-layer network (64 → 64 ReLU → 10 softmax) on scikit-learn's 8×8 digits, trained with
mini-batch SGD and backpropagation written by hand. Checks: the loss at initialisation, a
finite-difference gradient check, the final accuracy, and three ways to break it.

Run: .venv/bin/python subjects/machine-learning/sims/digits_numpy.py [--plot]
"""

import sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0                               # pixels in [0, 1]
y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print(f"Step 0 · {len(X)} images of 8×8 = 64 pixels; train {len(Xtr)}, test {len(Xte)}")


def init(rng, hidden=64, scale="he"):
    s1 = np.sqrt(2 / 64) if scale == "he" else 1.0
    return {"W1": rng.normal(0, s1, (64, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, np.sqrt(1 / hidden), (hidden, 10)), "b2": np.zeros(10)}


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    h = np.maximum(z1, 0)
    z2 = h @ p["W2"] + p["b2"]
    z2 = z2 - z2.max(1, keepdims=True)               # stable softmax
    P = np.exp(z2)
    P /= P.sum(1, keepdims=True)
    return z1, h, P


def loss_grads(p, X, y):
    z1, h, P = forward(p, X)
    n = len(y)
    loss = -np.mean(np.log(P[np.arange(n), y] + 1e-300))
    dz2 = P.copy()
    dz2[np.arange(n), y] -= 1
    dz2 /= n                                          # ∂L/∂z2 = (P − Y)/n
    g = {"W2": h.T @ dz2, "b2": dz2.sum(0)}
    dh = dz2 @ p["W2"].T
    dz1 = dh * (z1 > 0)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(0)
    return loss, g


def accuracy(p, X, y):
    return np.mean(forward(p, X)[2].argmax(1) == y)


def train(lr=0.1, epochs=30, batch=32, seed=0, **kw):
    rng = np.random.default_rng(seed)
    p = init(rng, **kw)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            _, g = loss_grads(p, Xtr[idx], ytr[idx])
            for k in p:
                p[k] -= lr * g[k]
        hist.append((loss_grads(p, Xtr, ytr)[0], accuracy(p, Xte, yte)))
    return p, hist


# Step 1 · the loss before training.
p0 = init(np.random.default_rng(0))
print(f"\nStep 1 · loss at initialisation = {loss_grads(p0, Xtr, ytr)[0]:.3f}   (ln 10 = {np.log(10):.3f})")

# Step 2 · gradient check on a few weights.
loss0, g = loss_grads(p0, Xtr[:50], ytr[:50])
eps, worst = 1e-5, 0.0
for k, idx in [("W1", (10, 3)), ("W1", (36, 40)), ("b1", (7,)), ("W2", (5, 2)), ("b2", (9,))]:
    pp = {kk: v.copy() for kk, v in p0.items()}
    pp[k][idx] += eps
    up = loss_grads(pp, Xtr[:50], ytr[:50])[0]
    pp[k][idx] -= 2 * eps
    dn = loss_grads(pp, Xtr[:50], ytr[:50])[0]
    num = (up - dn) / (2 * eps)
    worst = max(worst, abs(num - g[k][idx]) / max(1e-12, abs(num) + abs(g[k][idx])))
print(f"Step 2 · gradient check: worst relative error over 5 weights = {worst:.1e}")

# Step 3 · train.
p, hist = train()
print(f"\nStep 3 · SGD, lr 0.1, batch 32, 30 epochs:")
for ep in (0, 4, 9, 29):
    print(f"         epoch {ep + 1:>2}: train loss {hist[ep][0]:.4f}   test accuracy {hist[ep][1]:.4f}")
pred = forward(p, Xte)[2].argmax(1)
conf = np.zeros((10, 10), int)
np.add.at(conf, (yte, pred), 1)
off = [(conf[i, j], i, j) for i in range(10) for j in range(10) if i != j and conf[i, j]]
print(f"         mistakes: {len(yte) - np.trace(conf)} of {len(yte)}; most common: " + ", ".join(f"{i}→{j} ×{c}" for c, i, j in sorted(off, reverse=True)[:3]))

# Step 4 · baselines and breakages.
from sklearn.linear_model import LogisticRegression
lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
print(f"\nStep 4 · no hidden layer (scikit-learn softmax regression): test accuracy {lin:.4f}")
with np.errstate(all="ignore"):
    _, hb = train(lr=3.0, epochs=5)
print(f"         lr = 3.0:   train loss after each epoch " + ", ".join(f"{l:.3g}" for l, _ in hb) + f";  test accuracy {hb[-1][1]:.3f}")
pw = init(np.random.default_rng(0), scale="big")
print(f"         first-layer weights N(0, 1) instead of N(0, 2/64): loss at initialisation {loss_grads(pw, Xtr, ytr)[0]:.2f}")
_, hw = train(scale="big")
print(f"         ... after 30 epochs: train loss {hw[-1][0]:.4f}, test accuracy {hw[-1][1]:.4f}")
_, hs = train(lr=0.1, epochs=30, hidden=4)
print(f"         only 4 hidden units: test accuracy {hs[-1][1]:.4f}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([l for l, _ in hist], label="train loss")
    ax[0].plot([1 - a for _, a in hist], label="test error")
    ax[0].set(yscale="log", xlabel="epoch", title="Learning curves")
    ax[0].legend()
    wrong = np.flatnonzero(pred != yte)[:8]
    tiles = np.hstack([Xte[i].reshape(8, 8) for i in wrong])
    ax[1].imshow(tiles, cmap="gray_r")
    ax[1].set(title="Mistakes: " + "  ".join(f"{yte[i]}→{pred[i]}" for i in wrong), xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
