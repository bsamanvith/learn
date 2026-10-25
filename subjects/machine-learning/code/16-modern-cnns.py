import numpy as np

rng = np.random.default_rng(0)

def spirals(n, noise=0.08):                        # lesson 12's data
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    y = rng.integers(0, 2, n); r = t / (3 * np.pi)
    return np.c_[r * np.cos(t + np.pi * y), r * np.sin(t + np.pi * y)] + noise * rng.standard_normal((n, 2)), y

def init(depth, width, residual, seed):
    """An input layer, `depth` blocks of two ReLU layers each, and an output layer.
    Plain blocks: h -> relu(W2 relu(W1 h)).  Residual blocks: h -> h + W2 relu(W1 h)."""
    r = np.random.default_rng(seed); he = lambda m, k: r.standard_normal((m, k)) * np.sqrt(2 / k)
    blocks = []
    for _ in range(depth):
        W2 = he(width, width) * (0.0 if residual else 1.0)   # residual branches start at 0: each block is the identity
        blocks.append([he(width, width), np.zeros(width), W2, np.zeros(width)])
    return {"inp": [he(width, 2), np.zeros(width)], "blocks": blocks, "out": [he(1, width), np.zeros(1)], "res": residual}

def forward(P, X):
    h = np.maximum(X @ P["inp"][0].T + P["inp"][1], 0)
    cache = [h]
    for W1, b1, W2, b2 in P["blocks"]:
        a = np.maximum(h @ W1.T + b1, 0)
        u = a @ W2.T + b2
        h = h + u if P["res"] else np.maximum(u, 0)
        cache.append((a, u, h))
    return (h @ P["out"][0].T + P["out"][1])[:, 0], cache

def backward(P, X, y):
    F, cache = forward(P, X)
    loss = np.mean(np.logaddexp(0, F) - y * F)
    d = ((1 / (1 + np.exp(-F))) - y)[:, None] / len(y)
    h_last = cache[-1][2] if P["blocks"] else cache[0]
    G = {"out": [d.T @ h_last, d.sum(0)], "blocks": [None] * len(P["blocks"])}
    dh = d @ P["out"][0]
    for k in reversed(range(len(P["blocks"]))):
        W1, b1, W2, b2 = P["blocks"][k]
        a, u, h = cache[k + 1]
        h_prev = cache[k] if k == 0 else cache[k][2]
        du = dh if P["res"] else dh * (u > 0)
        da = (du @ W2) * (a > 0)
        G["blocks"][k] = [da.T @ h_prev, da.sum(0), du.T @ a, du.sum(0)]
        dh = (dh if P["res"] else 0) + da @ W1           # the identity path adds dh unchanged
    dz = dh * (cache[0] > 0)
    G["inp"] = [dz.T @ X, dz.sum(0)]
    return loss, G

def flat(P):
    return [P["inp"], *P["blocks"], P["out"]]

def train(depth, residual, steps=1500, lr=0.01, seed=0):
    """Full-batch SGD with momentum 0.9, which (unlike Adam) doesn't rescale small gradients."""
    P = init(depth, 32, residual, seed)
    vel = [[np.zeros_like(x) for x in grp] for grp in flat(P)]
    hist = []
    for t in range(steps):
        loss, G = backward(P, Xtr, ytr)
        if t == 0:
            g_first = np.linalg.norm(G["inp"][0])         # gradient reaching the first layer at init
        hist.append(loss)
        for grp, ggrp, vg in zip(flat(P), flat(G), vel):
            for i in range(len(grp)):
                vg[i] = 0.9 * vg[i] - lr * ggrp[i]; grp[i] += vg[i]
    return hist, np.mean((forward(P, Xte)[0] > 0) == yte), g_first

Xtr, ytr = spirals(400); Xte, yte = spirals(4000)
for depth in [4, 12, 24]:
    for residual in [False, True]:
        hist, acc, g = train(depth, residual)
        print(f"{2 * depth + 2:2d} layers  {'residual' if residual else 'plain   '}  |grad W_in| at init {g:.1e}"
              f"   train loss at 0/300/1500 steps {hist[0]:.3f} {hist[299]:.3f} {hist[-1]:.3f}   test acc {acc:.3f}")

# 10 layers  plain     |grad W_in| at init 3.4e-02   train loss at 0/300/1500 steps 0.670 0.148 0.001   test acc 0.954
# 10 layers  residual  |grad W_in| at init 1.1e-01   train loss at 0/300/1500 steps 0.826 0.510 0.004   test acc 0.946
# 26 layers  plain     |grad W_in| at init 1.5e-02   train loss at 0/300/1500 steps 0.702 0.676 0.016   test acc 0.930
# 26 layers  residual  |grad W_in| at init 1.2e-01   train loss at 0/300/1500 steps 0.770 0.434 0.000   test acc 0.943
# 50 layers  plain     |grad W_in| at init 8.6e-03   train loss at 0/300/1500 steps 0.692 0.691 0.693   test acc 0.494
# 50 layers  residual  |grad W_in| at init 6.7e-02   train loss at 0/300/1500 steps 0.666 0.077 0.000   test acc 0.951
