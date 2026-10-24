import numpy as np

rng = np.random.default_rng(0)
d = 10

def make(n):                                       # the label depends on 4 of the 10 features
    X = rng.uniform(-1, 1, (n, d))
    score = X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] * X[:, 3] - 0.6
    p = 1 / (1 + np.exp(-8 * score))               # labels are noisy: P(y = 1 | x) = p
    return X, (rng.random(n) < p).astype(int), p

def gini(counts):
    p = counts / np.maximum(counts.sum(-1, keepdims=True), 1)
    return 1 - (p ** 2).sum(-1)

def grow(X, y, max_features, rng):
    """A fully grown classification tree; each split considers a random subset of features."""
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        return int(np.argmax(counts))
    n = len(y); best = (None, None, gini(counts[None])[0] - 1e-12)
    for j in rng.choice(d, max_features, replace=False):
        o = np.argsort(X[:, j]); xs, ys = X[o, j], y[o]
        left = np.cumsum(np.eye(2)[ys], 0)[:-1]; right = counts - left; nl = np.arange(1, n)
        imp = (nl * gini(left) + (n - nl) * gini(right)) / n
        imp[xs[1:] == xs[:-1]] = np.inf
        i = np.argmin(imp)
        if imp[i] < best[2]:
            best = (j, (xs[i] + xs[i + 1]) / 2, imp[i])
    j, t, _ = best
    if j is None:
        return int(np.argmax(counts))
    m = X[:, j] <= t
    return (j, t, grow(X[m], y[m], max_features, rng), grow(X[~m], y[~m], max_features, rng))

def predict(tree, x):
    while not isinstance(tree, int):
        j, t, l, r = tree
        tree = l if x[j] <= t else r
    return tree

def forest(X, y, B, max_features, seed=0):
    r = np.random.default_rng(seed); n = len(y)
    trees, oob = [], []
    for _ in range(B):
        idx = r.integers(0, n, n)                  # bootstrap sample: n draws with replacement
        trees.append(grow(X[idx], y[idx], max_features, r))
        oob.append(np.setdiff1d(np.arange(n), idx))   # points this tree never saw
    return trees, oob

def votes(trees, X):
    return np.array([[predict(t, x) for x in X] for t in trees])   # (B, n) array of 0/1 votes

Xtr, ytr, _ = make(500); Xte, yte, pte = make(4000)
print(f"fraction of points left out of a bootstrap sample: {np.mean([len(np.setdiff1d(np.arange(500), rng.integers(0, 500, 500))) / 500 for _ in range(200)]):.3f}   (1/e = {np.exp(-1):.3f})")

print(f"best possible test accuracy (knowing p): {np.mean(np.maximum(pte, 1 - pte)):.3f}")
single = grow(Xtr, ytr, d, np.random.default_rng(1))
print(f"one deep tree               test acc {np.mean([predict(single, x) for x in Xte] == yte):.3f}")
for name, m in [("bagging (all 10 features)", d), ("random forest (3 features)", 3)]:
    trees, oob = forest(Xtr, ytr, 100, m)
    V = votes(trees, Xte)
    acc = [np.mean((V[:B].mean(0) > 0.5) == yte) for B in (1, 10, 100)]
    Vtr = votes(trees, Xtr); oob_mask = np.zeros_like(Vtr, bool)
    for b, o in enumerate(oob):
        oob_mask[b, o] = True
    oob_pred = (np.where(oob_mask, Vtr, 0).sum(0) / np.maximum(oob_mask.sum(0), 1)) > 0.5
    corr = np.corrcoef(V[:, :])[np.triu_indices(100, 1)].mean()
    print(f"{name:27s} test acc with 1/10/100 trees: {acc[0]:.3f} {acc[1]:.3f} {acc[2]:.3f}"
          f"   OOB acc {np.mean(oob_pred == ytr):.3f}   mean correlation between trees {corr:.2f}")

# fraction of points left out of a bootstrap sample: 0.367   (1/e = 0.368)
# best possible test accuracy (knowing p): 0.883
# one deep tree               test acc 0.710
# bagging (all 10 features)   test acc with 1/10/100 trees: 0.712 0.771 0.785   OOB acc 0.768   mean correlation between trees 0.36
# random forest (3 features)  test acc with 1/10/100 trees: 0.699 0.757 0.800   OOB acc 0.786   mean correlation between trees 0.20
