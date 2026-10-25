"""Assignment A3: gradient-boosted trees from scratch.

Fill in every function marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same functions

The checker tests each function on its own and tells you what is wrong, so you can work
through them in any order. The lesson each one builds on is noted in its docstring.
Conventions: X is an (n, d) array of features, y an (n,) array of targets (0/1 for the
logistic loss), F an (n,) array of raw scores (log-odds for the logistic loss). g and h
are the per-point first and second derivatives of the loss with respect to F.
"""
import numpy as np


# ---------------------------------------------------------------- Part 1: losses (lessons 03, 11)

def squared_loss(y, F):
    """Mean of (y - F)^2 / 2."""
    raise NotImplementedError  # TODO


def squared_grad_hess(y, F):
    """Return (g, h): derivatives of (y - F)^2 / 2 with respect to F, one entry per point."""
    raise NotImplementedError  # TODO


def log_loss(y, F):
    """Mean of log(1 + e^F) - y F, the log-loss written in terms of the log-odds F.
    Must not overflow for |F| up to 1000 (use np.logaddexp)."""
    raise NotImplementedError  # TODO


def log_grad_hess(y, F):
    """Return (g, h) for log(1 + e^F) - y F. Compute the sigmoid without overflow."""
    raise NotImplementedError  # TODO


def init_score(y, loss):
    """The best constant score F0 for loss "squared" (the mean) or "logistic" (the log-odds
    of the positive rate)."""
    raise NotImplementedError  # TODO


LOSSES = {"squared": (squared_loss, squared_grad_hess), "logistic": (log_loss, log_grad_hess)}


# ---------------------------------------------------------------- Part 2: one tree (lessons 09, 11)

def best_split(x, g, h, lam, gamma=0.0, min_child_weight=0.0):
    """Best threshold on ONE feature x for the second-order objective.

    Candidate thresholds are midpoints between consecutive DISTINCT sorted values of x;
    points with x <= t go left. The gain of a split is
        1/2 [ G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam) ] - gamma.
    Only splits with H_L >= min_child_weight and H_R >= min_child_weight are allowed.
    Return (threshold, gain) for the best one, or (None, -np.inf) if none is allowed.
    Do it in O(n log n): sort once, then use cumulative sums."""
    raise NotImplementedError  # TODO


def grow_tree(X, g, h, max_depth, lam=1.0, gamma=0.0, min_child_weight=0.0):
    """Grow a regression tree on gradients g and Hessians h.

    At each node, find the best split over all features with best_split. Split only if
    the depth limit allows it and the best gain is > 0; otherwise make a leaf with value
    w* = -G / (H + lam). Any representation is fine, as long as predict_tree reads it."""
    raise NotImplementedError  # TODO


def predict_tree(tree, X):
    """Value of the tree at each row of X, as an (n,) array."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 3: boosting (lesson 11)

def fit_gbdt(X, y, loss="logistic", rounds=100, lr=0.1, max_depth=3, lam=1.0, gamma=0.0,
             min_child_weight=0.0, subsample=1.0, seed=0):
    """Gradient boosting with second-order trees.

    Start from F0 = init_score(y, loss). Each round: compute (g, h) on ALL rows at the
    current F; if subsample < 1, keep each row independently with probability subsample
    (rng = np.random.default_rng(seed), one rng.random(n) draw per round); grow a tree on
    the kept rows; add lr * tree to F on all rows.
    Return a dict {"F0": float, "lr": float, "trees": list}."""
    raise NotImplementedError  # TODO


def predict_gbdt(model, X, upto=None):
    """Raw score F0 + lr * (sum of the first `upto` trees), all trees if upto is None."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 4: using the model (lessons 04, 10)

def best_rounds(model, X_val, y_val, loss="logistic"):
    """Early stopping after the fact. Return (M, curve): curve[m-1] is the validation loss
    with the first m trees (m = 1 .. number of trees), and M is the m with the smallest
    loss (the first one, if there are ties). Make it O(rounds), not O(rounds^2): add
    trees one at a time."""
    raise NotImplementedError  # TODO


def permutation_importance(predict, X, y, score, n_repeats=5, seed=0):
    """For each feature j, the mean drop in score(y, predict(X)) when column j is shuffled.

    predict maps an (n, d) array to predictions; score(y, pred) is higher-is-better. Use
    rng = np.random.default_rng(seed) and, for each feature in order, n_repeats shuffles
    of that column made with rng.permutation. Don't modify X. Return a (d,) array."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- data for the experiments

def make_data(n, d=10, noise=8.0, seed=None):
    """Lesson 10's problem: the label depends on x0^2 + x1^2 + x2 x3. Returns (X, y, p)."""
    r = np.random.default_rng(seed)
    X = r.uniform(-1, 1, (n, d))
    score = X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] * X[:, 3] - 0.6
    p = 1 / (1 + np.exp(-noise * score))
    return X, (r.random(n) < p).astype(int), p
