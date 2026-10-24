import numpy as np

rng = np.random.default_rng(0)

def make(n):                                       # two classes separated by a circle, with 10% label noise
    X = rng.uniform(-1, 1, (n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5).astype(int)
    flip = rng.random(n) < 0.1
    return X, np.where(flip, 1 - y, y)

def gini(counts):                                  # 1 - sum_k p_k^2, from class counts
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    return 1 - (p ** 2).sum(-1)

def best_split(X, y, K=2):
    """Try every feature and every threshold between consecutive sorted values.
    Returns (feature, threshold, weighted child impurity)."""
    n, d = X.shape
    best = (None, None, gini(np.bincount(y, minlength=K)[None])[0])
    for j in range(d):
        order = np.argsort(X[:, j]); xs, ys = X[order, j], y[order]
        left = np.cumsum(np.eye(K)[ys], axis=0)[:-1]       # class counts left of each cut
        right = np.bincount(ys, minlength=K) - left
        nl = np.arange(1, n)
        imp = (nl * gini(left) + (n - nl) * gini(right)) / n
        imp[xs[1:] == xs[:-1]] = np.inf                    # can't cut between equal values
        i = np.argmin(imp)
        if imp[i] < best[2] - 1e-12:
            best = (j, (xs[i] + xs[i + 1]) / 2, imp[i])
    return best

def grow(X, y, depth, min_leaf=1):
    counts = np.bincount(y, minlength=2)
    node = {"predict": int(np.argmax(counts)), "n": len(y)}
    if depth == 0 or len(y) < 2 * min_leaf or counts.min() == 0:
        return node
    j, t, _ = best_split(X, y)
    if j is None:
        return node
    m = X[:, j] <= t
    if m.sum() < min_leaf or (~m).sum() < min_leaf:
        return node
    node.update(feature=j, threshold=t, left=grow(X[m], y[m], depth - 1, min_leaf), right=grow(X[~m], y[~m], depth - 1, min_leaf))
    return node

def predict(node, X):
    if "feature" not in node:
        return np.full(len(X), node["predict"])
    m = X[:, node["feature"]] <= node["threshold"]
    out = np.empty(len(X), int)
    out[m], out[~m] = predict(node["left"], X[m]), predict(node["right"], X[~m])
    return out

def leaves(node):
    return 1 if "feature" not in node else leaves(node["left"]) + leaves(node["right"])

Xtr, ytr = make(400); Xte, yte = make(5000)
for depth in [1, 2, 4, 6, 8, 12, 20]:
    tree = grow(Xtr, ytr, depth)
    print(f"depth {depth:2d}: {leaves(tree):3d} leaves   train acc {np.mean(predict(tree, Xtr) == ytr):.3f}"
          f"   test acc {np.mean(predict(tree, Xte) == yte):.3f}")
print("best achievable test accuracy with 10% label noise: about 0.900")

# depth  1:   2 leaves   train acc 0.630   test acc 0.549
# depth  2:   4 leaves   train acc 0.750   test acc 0.691
# depth  4:  11 leaves   train acc 0.860   test acc 0.816
# depth  6:  27 leaves   train acc 0.882   test acc 0.823
# depth  8:  45 leaves   train acc 0.932   test acc 0.818
# depth 12:  76 leaves   train acc 0.995   test acc 0.766
# depth 20:  80 leaves   train acc 1.000   test acc 0.764
# best achievable test accuracy with 10% label noise: about 0.900
