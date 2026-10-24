"""Autograder for assignment A3. Usage: python3 check.py [your_file.py]

Each test checks a property the correct answer must have (a derivative, an optimality
condition, a brute-force search over every split, an exact identity), so passing means
your function is right, not merely that it matches one particular implementation."""
import importlib.util
import sys
import time
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

rng = np.random.default_rng(2024)
n, d = 150, 4
X = np.round(rng.uniform(-1, 1, (n, d)), 2)          # rounding creates ties, which must be handled
X[:, 3] = rng.integers(0, 3, n)                       # a feature with only three values
yr = np.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.2 * rng.standard_normal(n)
yb = (rng.random(n) < 1 / (1 + np.exp(-3 * (X[:, 0] ** 2 + X[:, 1] - 0.4)))).astype(float)
F_rand = rng.standard_normal(n)

results = []


def test(name, hint=""):
    def wrap(fn):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                fn()
            results.append((name, True, ""))
        except NotImplementedError:
            results.append((name, None, "not implemented yet"))
        except AssertionError as e:
            results.append((name, False, (str(e) or "wrong answer") + (f"\n        hint: {hint}" if hint else "")))
        except Exception:
            results.append((name, False, "crashed:\n" + "".join(traceback.format_exc(limit=-2))))
        return fn
    return wrap


def close(a, b, tol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: shape {a.shape}, expected {b.shape}"
    err = np.max(np.abs(a - b)) if a.size else 0
    assert err <= tol, f"{what}: off by {err:.3g} (tolerance {tol:g})"


def brute_split(x, g, h, lam, gamma=0.0, mcw=0.0):
    """Every allowed threshold, scored directly: O(n^2) but obviously right."""
    G, H = g.sum(), h.sum(); best = (None, -np.inf)
    v = np.unique(x)
    for t in (v[1:] + v[:-1]) / 2:
        m = x <= t
        if h[m].sum() < mcw or h[~m].sum() < mcw:
            continue
        gain = 0.5 * (g[m].sum()**2 / (h[m].sum() + lam) + g[~m].sum()**2 / (h[~m].sum() + lam) - G**2 / (H + lam)) - gamma
        if gain > best[1]:
            best = (t, gain)
    return best


def leaves(pred):
    """Group rows by predicted value: each group is one leaf."""
    return [pred == v for v in np.unique(pred)]


def objective(pred, g, h, lam):
    """The second-order objective of a tree, sum over leaves of -G^2 / (2 (H + lam))."""
    return sum(-g[m].sum() ** 2 / (2 * (h[m].sum() + lam)) for m in leaves(pred))


def num_derivs(loss, y, F, eps=1e-4):
    f = lambda Fv: np.array([loss(y[i:i + 1], Fv[i:i + 1]) for i in range(len(y))])
    return (f(F + eps) - f(F - eps)) / (2 * eps), (f(F + eps) - 2 * f(F) + f(F - eps)) / eps**2


@test("squared_loss / squared_grad_hess", "the loss is a MEAN of (y - F)^2 / 2; g = F - y and h = 1")
def _():
    close(S.squared_loss(np.array([1.0, 3.0]), np.array([0.0, 0.0])), 2.5, 1e-12, "squared_loss([1,3], [0,0])")
    g, h = S.squared_grad_hess(yr, F_rand)
    gn, hn = num_derivs(S.squared_loss, yr, F_rand)
    close(g, gn, 1e-6, "g against finite differences"); close(h, hn, 1e-3, "h against finite differences")


@test("log_loss / log_grad_hess", "use np.logaddexp(0, F); sigmoid(F) = exp(-logaddexp(0, -F)) never overflows")
def _():
    close(S.log_loss(np.array([1.0, 0.0]), np.array([1000.0, 1000.0])), 500.0, 1e-9, "log_loss at F = 1000 for y = 1 and y = 0")
    close(S.log_loss(np.array([0.0]), np.array([0.0])), np.log(2), 1e-12, "log_loss at F = 0")
    g, h = S.log_grad_hess(yb, 3 * F_rand)
    gn, hn = num_derivs(S.log_loss, yb, 3 * F_rand)
    close(g, gn, 1e-6, "g against finite differences"); close(h, hn, 1e-3, "h against finite differences")
    g, h = S.log_grad_hess(np.array([1.0, 0.0]), np.array([-1000.0, 1000.0]))
    close(g, [-1, 1], 1e-12, "g at F = -1000 (y = 1) and F = 1000 (y = 0)"); close(h, [0, 0], 1e-12, "h at F = +-1000")


@test("init_score", "F0 is where the mean loss has zero derivative: the mean for squared loss, log(m / (1 - m)) for logistic")
def _():
    for loss, y, gh in [("squared", yr, S.squared_grad_hess), ("logistic", yb, S.log_grad_hess)]:
        F0 = S.init_score(y, loss)
        assert np.ndim(F0) == 0, f"init_score({loss!r}) should be a single number"
        close(gh(y, np.full(n, float(F0)))[0].sum(), 0, 1e-8, f"sum of gradients at F0 ({loss})")


@test("best_split", "sort once; cumulative sums of g and h give G_L and H_L at every threshold; skip ties x[i] == x[i+1]")
def _():
    for loss_y, gh in [(yr, S.squared_grad_hess), (yb, S.log_grad_hess)]:
        g, h = gh(loss_y, 0.3 * F_rand)
        for j in range(d):
            for lam, gamma, mcw in [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (5.0, 0.5, 3.0)]:
                t, gain = S.best_split(X[:, j], g, h, lam, gamma, mcw)
                tb, gb = brute_split(X[:, j], g, h, lam, gamma, mcw)
                close(gain, gb, 1e-8, f"best gain on feature {j} (lam={lam}, gamma={gamma}, min_child_weight={mcw})")
                if tb is not None:
                    v = np.unique(X[:, j])
                    assert t is not None and np.isclose(t, (v[1:] + v[:-1]) / 2).any(), f"threshold {t} on feature {j} is not a midpoint between distinct values"
                    m = X[:, j] <= t
                    gt = 0.5 * (g[m].sum()**2 / (h[m].sum() + lam) + g[~m].sum()**2 / (h[~m].sum() + lam) - g.sum()**2 / (h.sum() + lam)) - gamma
                    close(gt, gb, 1e-8, f"the gain of your threshold {t:.3f} on feature {j}")
    t, gain = S.best_split(np.ones(10), np.arange(10.0), np.ones(10), 1.0)
    assert t is None and gain == -np.inf, "a constant feature has no split: return (None, -np.inf)"
    t, gain = S.best_split(X[:, 0], np.ones(n), np.ones(n), 1.0, 0.0, min_child_weight=n)
    assert t is None and gain == -np.inf, "no split satisfies min_child_weight = n: return (None, -np.inf)"
    big = np.random.default_rng(0).standard_normal(200_000)
    t0 = time.perf_counter(); S.best_split(big, big, np.ones_like(big), 1.0)
    assert time.perf_counter() - t0 < 3, "too slow on 200,000 points: avoid a Python loop over thresholds"


@test("grow_tree / predict_tree", "split only when the best gain over all features is > 0; leaves hold -G / (H + lam)")
def _():
    g, h = S.log_grad_hess(yb, 0.3 * F_rand)
    for lam in [1.0, 0.1]:
        p0 = S.predict_tree(S.grow_tree(X, g, h, 0, lam), X)
        close(p0, np.full(n, -g.sum() / (h.sum() + lam)), 1e-10, f"a depth-0 tree is one leaf (lam={lam})")
        p1 = S.predict_tree(S.grow_tree(X, g, h, 1, lam), X)
        best = max(brute_split(X[:, j], g, h, lam)[1] for j in range(d))
        close(objective(p0, g, h, lam) - objective(p1, g, h, lam), best, 1e-8, "depth-1 tree: its gain should be the best over all features")
        objs = []
        for depth in [2, 3, 4]:
            p = S.predict_tree(S.grow_tree(X, g, h, depth, lam), X)
            L = leaves(p)
            assert len(L) <= 2 ** depth, f"depth {depth} tree has {len(L)} leaves (max {2 ** depth})"
            for m in L:
                close(p[m][0], -g[m].sum() / (h[m].sum() + lam), 1e-10, f"a leaf value in a depth-{depth} tree")
            objs.append(objective(p, g, h, lam))
        assert objs[0] >= objs[1] - 1e-12 >= objs[2] - 2e-12, "objective should not increase with depth"
    p = S.predict_tree(S.grow_tree(X, g, h, 4, 1.0, gamma=1e6), X)
    assert len(leaves(p)) == 1, "with a huge gamma no split has positive gain, so the tree is one leaf"
    p = S.predict_tree(S.grow_tree(X, g, h, 6, 1.0, min_child_weight=4.0), X)
    for m in leaves(p):
        assert h[m].sum() >= 4.0 - 1e-9, "a leaf has Hessian sum below min_child_weight"
    p = S.predict_tree(S.grow_tree(X[:40], g[:40], h[:40], 3, 1.0), X[40:])
    assert p.shape == (n - 40,), "predict_tree must work on rows the tree was not grown on"


@test("fit_gbdt / predict_gbdt", "F starts at init_score; each round: g, h at the current F, grow a tree, F += lr * tree")
def _():
    m0 = S.fit_gbdt(X, yb, "logistic", rounds=0)
    close(S.predict_gbdt(m0, X), np.full(n, S.init_score(yb, "logistic")), 1e-12, "with 0 rounds, predictions are F0")
    m = S.fit_gbdt(X, yb, "logistic", rounds=30, lr=0.3, max_depth=2)
    assert len(m["trees"]) == 30, "model should hold one tree per round"
    losses = [S.log_loss(yb, S.predict_gbdt(m, X, upto=k)) for k in range(31)]
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])), "training log-loss should fall every round with lr = 0.3"
    close(S.predict_gbdt(m, X), S.predict_gbdt(m, X, upto=30), 1e-12, "upto=None should use every tree")
    F, F0 = np.full(n, m["F0"]), m["F0"]
    for k, t in enumerate(m["trees"][:5]):           # replay the rounds: each tree is the right tree
        g, h = S.log_grad_hess(yb, F)
        p = S.predict_tree(t, X); p_ref = S.predict_tree(S.grow_tree(X, g, h, 2, 1.0), X)
        close(objective(p, g, h, 1.0), objective(p_ref, g, h, 1.0), 1e-8, f"round {k + 1}: tree objective at the current F")
        F = F + 0.3 * p
    # squared loss, lr 1, lam 0, one stump: exactly "fit the best stump to the residuals"
    ms = S.fit_gbdt(X, yr, "squared", rounds=1, lr=1.0, max_depth=1, lam=0.0)
    r = yr - yr.mean()
    best = max(brute_split(X[:, j], -r, np.ones(n), 0.0)[1] for j in range(d))
    sse = np.sum((yr - S.predict_gbdt(ms, X)) ** 2)
    close(np.sum(r ** 2) - sse, 2 * best, 1e-8, "squared loss, one stump: SSE decrease should be twice the best gain")


@test("fit_gbdt subsample", "rng = np.random.default_rng(seed); each round keep = rng.random(n) < subsample; gradients on all rows")
def _():
    a = S.fit_gbdt(X, yb, "logistic", rounds=4, lr=0.5, max_depth=2, subsample=0.5, seed=7)
    b = S.fit_gbdt(X, yb, "logistic", rounds=4, lr=0.5, max_depth=2, subsample=0.5, seed=7)
    close(S.predict_gbdt(a, X), S.predict_gbdt(b, X), 0, "same seed should give the same model")
    r = np.random.default_rng(7); F = np.full(n, a["F0"])
    for k, t in enumerate(a["trees"]):
        keep = r.random(n) < 0.5
        g, h = S.log_grad_hess(yb, F)
        p = S.predict_tree(t, X)
        for lf in leaves(p[keep]):
            idx = np.flatnonzero(keep)[lf]
            close(p[idx[0]], -g[idx].sum() / (h[idx].sum() + 1.0), 1e-10, f"round {k + 1}: leaf values should use only the kept rows")
        F = F + 0.5 * p


@test("best_rounds", "add one tree at a time to a running F; np.argmin returns the first minimum")
def _():
    m = S.fit_gbdt(X[:100], yb[:100], "logistic", rounds=60, lr=0.5, max_depth=3)
    M, curve = S.best_rounds(m, X[100:], yb[100:], "logistic")
    ref = np.array([S.log_loss(yb[100:], S.predict_gbdt(m, X[100:], upto=k)) for k in range(1, 61)])
    close(np.asarray(curve), ref, 1e-10, "validation curve")
    assert M == int(np.argmin(ref)) + 1, f"M = {M}, but the smallest validation loss is at {int(np.argmin(ref)) + 1} trees"
    ms = S.fit_gbdt(X[:100], yr[:100], "squared", rounds=20, lr=0.5, max_depth=2)
    M, curve = S.best_rounds(ms, X[100:], yr[100:], "squared")
    close(curve[-1], S.squared_loss(yr[100:], S.predict_gbdt(ms, X[100:])), 1e-10, "squared-loss curve uses the loss argument")


@test("permutation_importance", "copy X before shuffling; for j in range(d): for each repeat: rng.permutation(column)")
def _():
    w = np.array([2.0, -1.0, 0.0, 0.5])
    predict = lambda A: A @ w
    score = lambda y, p: -np.mean((y - p) ** 2)
    Xc = X.copy(); yy = X @ w
    imp = np.asarray(S.permutation_importance(predict, X, yy, score, n_repeats=4, seed=3))
    assert np.array_equal(X, Xc), "permutation_importance changed X: shuffle a copy"
    assert imp.shape == (d,), f"shape {imp.shape}, expected ({d},)"
    close(imp[2], 0, 0, "a feature the model ignores has importance exactly 0")
    assert imp[0] > imp[1] > 0 and imp[3] > 0, "features the model uses should have positive importance, larger for larger weights"
    r = np.random.default_rng(3); ref = np.zeros(d)
    for j in range(d):
        for _ in range(4):
            A = X.copy(); A[:, j] = r.permutation(A[:, j]); ref[j] += -score(yy, predict(A)) / 4
    close(imp, ref, 1e-10, "importances with the specified random draws")


done = sum(ok is True for _, ok, _m in results)
for name, ok, msg in results:
    mark = {True: "PASS", False: "FAIL", None: "todo"}[ok]
    print(f"  {mark}  {name}" + (f"\n        {msg}" if msg else ""))
print(f"\n{done} / {len(results)} passed")
sys.exit(0 if done == len(results) else 1)
