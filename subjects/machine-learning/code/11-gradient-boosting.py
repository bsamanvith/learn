import numpy as np

rng = np.random.default_rng(0)
d = 10

def make(n):                                       # the same problem as lesson 10
    X = rng.uniform(-1, 1, (n, d))
    score = X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] * X[:, 3] - 0.6
    p = 1 / (1 + np.exp(-8 * score))
    return X, (rng.random(n) < p).astype(int), p

def build(X, g, h, depth, lam, gamma):
    """A regression tree fitted to gradients g and Hessians h with the second-order gain.
    Leaves store the Newton step w* = -G / (H + lam)."""
    G, H = g.sum(), h.sum()
    leaf = -G / (H + lam)
    if depth == 0 or len(g) < 2:
        return leaf
    best_gain, best = 0.0, None
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j]); xs = X[o, j]
        GL, HL = np.cumsum(g[o])[:-1], np.cumsum(h[o])[:-1]
        gain = 0.5 * (GL**2 / (HL + lam) + (G - GL)**2 / (H - HL + lam) - G**2 / (H + lam)) - gamma
        gain[xs[1:] == xs[:-1]] = -np.inf
        i = np.argmax(gain)
        if gain[i] > best_gain:
            best_gain, best = gain[i], (j, (xs[i] + xs[i + 1]) / 2)
    if best is None:
        return leaf
    j, t = best; m = X[:, j] <= t
    return (j, t, build(X[m], g[m], h[m], depth - 1, lam, gamma), build(X[~m], g[~m], h[~m], depth - 1, lam, gamma))

def tree_predict(tree, X):
    if not isinstance(tree, tuple):
        return np.full(len(X), tree)
    j, t, l, r = tree; m = X[:, j] <= t
    out = np.empty(len(X)); out[m], out[~m] = tree_predict(l, X[m]), tree_predict(r, X[~m])
    return out

sigmoid = lambda z: 1 / (1 + np.exp(-z))
logloss = lambda y, F: np.mean(np.logaddexp(0, F) - y * F)

def boost(X, y, rounds, lr=0.1, depth=3, lam=1.0, gamma=0.0):
    F0 = np.log(y.mean() / (1 - y.mean()))        # start from the best constant: the log-odds
    F = np.full(len(y), F0)
    trees = []
    for _ in range(rounds):
        p = sigmoid(F)
        g, h = p - y, p * (1 - p)                  # gradient and Hessian of the log-loss in F
        tree = build(X, g, h, depth, lam, gamma)
        F += lr * tree_predict(tree, X)
        trees.append(tree)
    return F0, trees

def boosted_F(F0, trees, X, lr=0.1, upto=None):
    return F0 + lr * sum(tree_predict(t, X) for t in trees[:upto])

Xtr, ytr, _ = make(500); Xte, yte, pte = make(4000)
F0, trees = boost(Xtr, ytr, rounds=400)
for M in [1, 10, 50, 100, 200, 400]:
    Ftr, Fte = boosted_F(F0, trees, Xtr, upto=M), boosted_F(F0, trees, Xte, upto=M)
    print(f"{M:3d} trees: train log-loss {logloss(ytr, Ftr):.3f}  test log-loss {logloss(yte, Fte):.3f}"
          f"  test acc {np.mean((Fte > 0) == yte):.3f}")
print(f"best possible: test log-loss {logloss(yte, np.log(pte / (1 - pte))):.3f}  test acc {np.mean(np.maximum(pte, 1 - pte)):.3f}")

#   1 trees: train log-loss 0.664  test log-loss 0.670  test acc 0.711
#  10 trees: train log-loss 0.526  test log-loss 0.572  test acc 0.746
#  50 trees: train log-loss 0.311  test log-loss 0.472  test acc 0.778
# 100 trees: train log-loss 0.234  test log-loss 0.464  test acc 0.781
# 200 trees: train log-loss 0.121  test log-loss 0.436  test acc 0.795
# 400 trees: train log-loss 0.040  test log-loss 0.427  test acc 0.815
# best possible: test log-loss 0.265  test acc 0.883
