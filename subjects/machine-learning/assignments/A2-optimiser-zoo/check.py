"""Autograder for assignment A2. Usage: python3 check.py [your_file.py]

Each optimiser is run for a few steps on fixed gradients and compared with the update rule
written out independently here; the loss is checked against finite differences, the schedules
and metrics on hand-computed values, and the optimisers on an ill-conditioned valley. It does
not need MNIST, and takes a few seconds."""
import importlib.util
import math
import sys
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

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


rng0 = np.random.default_rng(0)
P0 = [rng0.normal(0, 1, (3, 2)), rng0.normal(0, 1, 4)]
G = [[rng0.normal(0, 1, (3, 2)), rng0.normal(0, 1, 4)] for _ in range(4)]


def run(opt_cls, **kw):
    params = [p.copy() for p in P0]
    ids = [id(p) for p in params]
    opt = opt_cls(params, **kw)
    for g in G:
        opt.step([x.copy() for x in g])
    assert [id(p) for p in params] == ids, "update the parameter arrays in place (p -= ...), do not rebind them"
    return params


def reference(update):
    """Run a per-array update rule written out here, with a state dict per array."""
    out = []
    for k in range(2):
        p, st = P0[k].copy(), {}
        for t, g in enumerate(G, start=1):
            p = update(p, G[t - 1][k], st, t)
        out.append(p)
    return out


def compare(mine, ref, what, tol=1e-10):
    for a, b, name in zip(mine, ref, ("the matrix", "the vector")):
        close(a, b, tol, f"{what}: {name} after 4 steps")


# ---------------------------------------------------------------- Part 1

@test("softmax_cross_entropy", "subtract the row max before exponentiating; the gradient is (softmax - onehot) / B")
def _():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, (5, 4)); y = np.array([0, 3, 1, 1, 2])
    loss, g = S.softmax_cross_entropy(z, y)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    close(loss, -np.mean(np.log(p[np.arange(5), y])), 1e-12, "loss")
    num = np.zeros_like(z); h = 1e-6
    for i in np.ndindex(z.shape):
        zp, zm = z.copy(), z.copy(); zp[i] += h; zm[i] -= h
        num[i] = (S.softmax_cross_entropy(zp, y)[0] - S.softmax_cross_entropy(zm, y)[0]) / (2 * h)
    close(g, num, 1e-7, "gradient")
    big, _ = S.softmax_cross_entropy(np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([0, 0]))
    close(big, 500.0, 1e-9, "loss with logits of 1000 (needs the max-subtraction trick)")


# ---------------------------------------------------------------- Part 2

@test("SGD")
def _():
    compare(run(S.SGD, lr=0.1), reference(lambda p, g, st, t: p - 0.1 * g), "SGD")


@test("Momentum (heavy ball)", "v <- beta v + g, then p <- p - lr v")
def _():
    def upd(p, g, st, t):
        st["v"] = 0.9 * st.get("v", 0) + g
        return p - 0.1 * st["v"]
    compare(run(S.Momentum, lr=0.1, beta=0.9), reference(upd), "heavy ball")


@test("Momentum (Nesterov)", "with nesterov=True the step is lr (g + beta v), with v already updated")
def _():
    def upd(p, g, st, t):
        st["v"] = 0.9 * st.get("v", 0) + g
        return p - 0.1 * (g + 0.9 * st["v"])
    compare(run(S.Momentum, lr=0.1, beta=0.9, nesterov=True), reference(upd), "Nesterov")


@test("AdaGrad and RMSProp", "AdaGrad sums squared gradients; RMSProp keeps an exponential average of them")
def _():
    def ada(p, g, st, t):
        st["s"] = st.get("s", 0) + g * g
        return p - 0.5 * g / (np.sqrt(st["s"]) + 1e-8)
    compare(run(S.AdaGrad, lr=0.5), reference(ada), "AdaGrad")
    def rms(p, g, st, t):
        st["s"] = 0.9 * st.get("s", 0) + 0.1 * g * g
        return p - 0.01 * g / (np.sqrt(st["s"]) + 1e-8)
    compare(run(S.RMSProp, lr=0.01, rho=0.9), reference(rms), "RMSProp")


@test("Adam", "bias-correct both moments with the step count t, starting at t = 1")
def _():
    def upd(p, g, st, t):
        st["m"] = 0.9 * st.get("m", 0) + 0.1 * g; st["v"] = 0.999 * st.get("v", 0) + 0.001 * g * g
        return p - 0.01 * (st["m"] / (1 - 0.9 ** t)) / (np.sqrt(st["v"] / (1 - 0.999 ** t)) + 1e-8)
    compare(run(S.Adam, lr=0.01), reference(upd), "Adam")
    params = [np.array([1.0, 1.0])]; opt = S.Adam(params, lr=0.01)
    opt.step([np.array([1e-3, -500.0])])
    close(params[0], [0.99, 1.01], 1e-5, "Adam's first step is lr * sign(g), whatever the gradient's size")


@test("Adam with L2 and AdamW", "Adam: add wd * p to the gradient. AdamW: p -= lr * wd * p, then the Adam step on the plain gradient")
def _():
    def l2(p, g, st, t):
        g = g + 0.1 * p
        st["m"] = 0.9 * st.get("m", 0) + 0.1 * g; st["v"] = 0.999 * st.get("v", 0) + 0.001 * g * g
        return p - 0.01 * (st["m"] / (1 - 0.9 ** t)) / (np.sqrt(st["v"] / (1 - 0.999 ** t)) + 1e-8)
    compare(run(S.Adam, lr=0.01, weight_decay=0.1), reference(l2), "Adam with weight_decay (L2)")
    def w(p, g, st, t):
        p = p - 0.01 * 0.1 * p
        st["m"] = 0.9 * st.get("m", 0) + 0.1 * g; st["v"] = 0.999 * st.get("v", 0) + 0.001 * g * g
        return p - 0.01 * (st["m"] / (1 - 0.9 ** t)) / (np.sqrt(st["v"] / (1 - 0.999 ** t)) + 1e-8)
    compare(run(S.AdamW, lr=0.01, weight_decay=0.1), reference(w), "AdamW")


@test("the optimisers on an ill-conditioned valley", "if the update tests pass, this one should too")
def _():
    f, grad = S.quadratic(100.0)
    def go(make, steps=200):
        x = [np.array([1.0, 1.0])]; opt = make(x)
        for _ in range(steps):
            opt.step([grad(x[0])])
        return f(x[0])
    sgd = go(lambda x: S.SGD(x, lr=0.019))
    mom = go(lambda x: S.Momentum(x, lr=0.019, beta=0.8))
    adam = go(lambda x: S.Adam(x, lr=0.05))
    close(sgd, 0.5 * (1 - 0.019) ** 400 + 50 * 0.9 ** 400, 1e-12, "SGD at lr 0.019 after 200 steps (each axis shrinks by |1 - lr * curvature| per step)")
    assert mom < 1e-15, f"heavy-ball momentum (beta 0.8) should reach f below 1e-15 in 200 steps; got {mom:.3g}"
    assert adam < 1e-6, f"Adam (lr 0.05) should reach f below 1e-6 in 200 steps; got {adam:.3g}"
    big = go(lambda x: S.SGD(x, lr=0.021), steps=50)
    assert big > 1, f"SGD with lr above 2 / kappa = 0.02 must diverge along the steep axis; got f = {big:.3g}"


# ---------------------------------------------------------------- Part 3

@test("schedules", "warmup_cosine: (t + 1) / warmup of base_lr during warm-up, then the cosine from t = warmup")
def _():
    close([S.step_decay(t, 0.1, 100) for t in (0, 99, 100, 250)], [0.1, 0.1, 0.01, 0.001], 1e-15, "step_decay")
    close([S.warmup_cosine(t, 1.0, 10, 110) for t in (0, 4, 9, 10, 60, 110)],
          [0.1, 0.5, 1.0, 1.0, 0.5, 0.0], 1e-12, "warmup_cosine at t = 0, 4, 9, 10, 60, 110")
    close(S.warmup_cosine(60, 1.0, 10, 110, min_lr=0.2), 0.6, 1e-12, "warmup_cosine half-way with min_lr 0.2")
    close([S.inverse_sqrt(t, 1.0, 100) for t in (0, 49, 99, 399)], [0.01, 0.5, 1.0, 0.5], 1e-12, "inverse_sqrt")


# ---------------------------------------------------------------- Part 4

@test("confusion_matrix and expected_calibration_error", "ECE: bin by the top probability; weight |accuracy - confidence| by the bin's share")
def _():
    C = S.confusion_matrix(np.array([0, 0, 1, 2, 2, 2]), np.array([0, 1, 1, 2, 0, 2]), k=3)
    close(C, [[1, 1, 0], [0, 1, 0], [1, 0, 2]], 0, "confusion matrix")
    probs = np.array([[0.95, 0.05], [0.95, 0.05], [0.65, 0.35], [0.35, 0.65], [0.55, 0.45], [1.0, 0.0]])
    y = np.array([0, 1, 0, 0, 0, 0])
    # bins: 0.95, 0.95 -> bin 9 with 1.0 -> acc 2/3, conf (0.95+0.95+1)/3; 0.65, 0.65 -> bin 6, acc 1/2, conf 0.65; 0.55 -> bin 5, acc 1
    expect = 3 / 6 * abs(2 / 3 - 2.9 / 3) + 2 / 6 * abs(0.5 - 0.65) + 1 / 6 * abs(1 - 0.55)
    close(S.expected_calibration_error(probs, y), expect, 1e-12, "ECE")


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
