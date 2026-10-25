"""Autograder for assignment A1. Usage: python3 check.py [your_file.py]

Each test checks a property the correct answer must have (an optimality condition, a
known closed form, an invariance), so passing means your function is right, not merely
that it matches one particular implementation."""
import importlib.util
import sys
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

rng = np.random.default_rng(12345)
n, d = 120, 6
X = rng.standard_normal((n, d)) * [1, 2, 0.5, 1, 3, 1] + [0, 1, -2, 0, 0.5, 3]
w_true = np.array([1.5, -0.5, 2.0, 0.0, 0.3, 0.0])
y = X @ w_true + 4.0 + 0.5 * rng.standard_normal(n)
yb = (rng.random(n) < 1 / (1 + np.exp(-(X @ w_true * 0.5 - 1)))).astype(float)
Xc, yc = X - X.mean(0), y - y.mean()

results = []


def test(name, hint=""):
    def wrap(fn):
        try:
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


@test("fit_least_squares", "centre X and y, solve for w, then b = mean(y) - mean(X) @ w")
def _():
    w, b = S.fit_least_squares(X, y)
    r = y - X @ w - b
    close(X.T @ r, np.zeros(d), 1e-6, "normal equations X^T r = 0")
    close(r.sum(), 0, 1e-6, "residuals must sum to zero (intercept condition)")


@test("gd_least_squares", "gradient wrt w is -(1/n) X^T r and wrt b is -(1/n) sum(r), with r = y - Xw - b")
def _():
    w, b, losses = S.gd_least_squares(Xc / 3, yc, lr=0.1, steps=50)
    assert len(losses) == 50, f"expected 50 losses, got {len(losses)}"
    close(losses[0], 0.5 * np.mean(yc**2), 1e-9, "loss before the first step (weights start at zero)")
    assert np.all(np.diff(losses) <= 1e-12), "the loss went up at some step with a safe learning rate"
    ww, bb = np.zeros(d), 0.0                      # three exact steps by hand
    for _ in range(3):
        r = yc - (Xc / 3) @ ww - bb
        ww, bb = ww + 0.1 * (Xc / 3).T @ r / n, bb + 0.1 * r.mean()
    w3, b3, _l = S.gd_least_squares(Xc / 3, yc, lr=0.1, steps=3)
    close(w3, ww, 1e-9, "weights after 3 steps")


@test("max_stable_lr", "Hessian of L is A^T A / n with A = [X, 1]")
def _():
    lr = S.max_stable_lr(X)
    A = np.c_[X, np.ones(n)]
    close(lr, 2 / np.linalg.eigvalsh(A.T @ A / n)[-1], 1e-9, "2 / lambda_max")
    _w, _b, l1 = S.gd_least_squares(X, y, lr=0.95 * lr, steps=400)
    _w, _b, l2 = S.gd_least_squares(X, y, lr=1.05 * lr, steps=400)
    assert l1[-1] < l1[0] and l2[-1] > l2[0], "GD should converge just below this rate and diverge just above it"


@test("sigmoid", "for z < 0 use exp(z) / (1 + exp(z))")
def _():
    z = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        s = S.sigmoid(z)
    close(s, [0, 1 / (1 + np.exp(5)), 0.5, 1 / (1 + np.exp(-5)), 1], 1e-12, "sigmoid values")


@test("fit_logistic", "gradient of the mean log-loss is X^T (sigmoid(Xw + b) - y) / n")
def _():
    for lam in [0.0, 0.1]:
        w, b = S.fit_logistic(Xc, yb, lam=lam, lr=0.3, steps=4000)
        p = 1 / (1 + np.exp(-(Xc @ w + b)))
        close(Xc.T @ (p - yb) / n + lam * w, np.zeros(d), 2e-4, f"gradient at the solution (lam={lam})")
        close(np.mean(p - yb), 0, 2e-4, f"intercept gradient (lam={lam}); is b penalised by mistake?")


@test("fit_ridge", "solve (Xc^T Xc + lam I) w = Xc^T yc on centred data")
def _():
    for lam in [0.0, 1.0, 50.0]:
        w, b = S.fit_ridge(X, y, lam)
        r = y - X @ w - b
        close(-X.T @ r + lam * w, np.zeros(d), 1e-6, f"stationarity X^T r = lam w (lam={lam})")
        close(r.sum(), 0, 1e-6, f"intercept condition (lam={lam})")
    w1, _ = S.fit_ridge(X, y, 10.0); w2, _ = S.fit_ridge(X, y + 100.0, 10.0)
    close(w1, w2, 1e-8, "shifting y must not change w (don't penalise the intercept)")


@test("ridge_df")
def _():
    d_ = np.linalg.svd(Xc, compute_uv=False)
    for lam in [0.0, 5.0, 1e6]:
        close(S.ridge_df(X, lam), np.sum(d_**2 / (d_**2 + lam)), 1e-9, f"df at lam={lam}")


@test("soft_threshold")
def _():
    close(S.soft_threshold(np.array([-3.0, -0.5, 0.0, 0.2, 2.0]), 1.0), [-2, 0, 0, 0, 1], 1e-12, "values")


@test("fit_lasso", "KKT: |x_j^T r| / n <= lam where w_j = 0, and = lam * sign(w_j) where w_j != 0")
def _():
    lam = 0.15
    w, b = S.fit_lasso(X, y, lam, iters=2000)
    r = y - X @ w - b
    corr = Xc.T @ r / n
    nz = np.abs(w) > 1e-8
    close(corr[nz], lam * np.sign(w[nz]), 1e-6, "active coordinates must have x_j^T r / n = lam sign(w_j)")
    assert np.all(np.abs(corr[~nz]) <= lam + 1e-6), "a zero coordinate violates |x_j^T r| / n <= lam"
    assert (~nz).sum() >= 1, "with this lam at least one weight should be exactly zero"
    close(r.sum(), 0, 1e-6, "intercept condition")
    lmax = np.max(np.abs(Xc.T @ (y - y.mean()))) / n
    w0, _ = S.fit_lasso(X, y, 1.01 * lmax)
    close(w0, np.zeros(d), 0, "above lam_max every weight must be exactly 0")


@test("kfold_indices")
def _():
    folds = S.kfold_indices(103, 5, seed=1)
    assert len(folds) == 5, "expected 5 folds"
    allidx = np.sort(np.concatenate(folds))
    assert np.array_equal(allidx, np.arange(103)), "folds must partition range(n)"
    sizes = sorted(len(f) for f in folds)
    assert sizes[-1] - sizes[0] <= 1, f"fold sizes {sizes} differ by more than one"
    assert not np.array_equal(np.concatenate(folds), np.arange(103)), "indices should be shuffled"


@test("cv_mse", "fit on the other k-1 folds, predict the held-out fold, average squared errors over all n points")
def _():
    folds = S.kfold_indices(n, 4, seed=3)
    tot = 0.0
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        A = np.c_[X[tr] - X[tr].mean(0)]; wf = np.linalg.solve(A.T @ A + 2.0 * np.eye(d), A.T @ (y[tr] - y[tr].mean()))
        bf = y[tr].mean() - X[tr].mean(0) @ wf
        tot += np.sum((y[f] - X[f] @ wf - bf) ** 2)
    close(S.cv_mse(X, y, 2.0, k=4, seed=3), tot / n, 1e-8, "4-fold CV error at lam=2")


@test("choose_lambda")
def _():
    Xs = X[:, :1]; ys = y.copy()
    lams = [0.0, 1.0, 1e5]
    scores = [S.cv_mse(Xs, ys, l) for l in lams]
    assert S.choose_lambda(Xs, ys, lams) == lams[int(np.argmin(scores))], "should return the lam with the smallest cv_mse"


done = sum(ok is True for _, ok, _m in results)
for name, ok, msg in results:
    mark = {True: "PASS", False: "FAIL", None: "todo"}[ok]
    print(f"  {mark}  {name}" + (f"\n        {msg}" if msg else ""))
print(f"\n{done} / {len(results)} passed")
sys.exit(0 if done == len(results) else 1)
