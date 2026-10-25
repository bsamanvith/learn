"""Autograder for assignment A8. Usage: python3 check.py [your_file.py]

Probabilities are checked by enumerating all responses, gradients against central finite
differences, the closed-form policy against its defining objective, and the training steps
by whether they reach the right answers."""
import copy
import importlib.util
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


def numgrad(f, a, idxs, h=1e-6):
    out = []
    for idx in idxs:
        old = a[idx]
        a[idx] = old + h; up = f()
        a[idx] = old - h; dn = f()
        a[idx] = old
        out.append((up - dn) / (2 * h))
    return np.array(out)


def some(shape, n, seed):
    r = np.random.default_rng(seed)
    return [tuple(r.integers(0, s) for s in shape) for _ in range(n)]


class Recorder:
    def __init__(self, layers):
        self.layers, self.calls = layers, 0
    def step(self, sign=1.0):
        self.calls += 1; self.sign = sign
        self.grads = [{k: v.copy() for k, v in l.grads.items()} for l in self.layers]


def setup(seed=0):
    rng = np.random.default_rng(seed)
    ref = S.make_reference(rng)
    R = np.array([S.true_reward(np.full(216, x), S.RESPONSES) for x in range(10)])
    return rng, ref, R


# ---------------------------------------------------------------- Part 1: LoRA

@test("LoRALinear: forward, merge and gradients", "y = x W0 + b0 + (alpha / r) (x A) B; the gradient of B is scale * (xA)^T g, of A is scale * x^T (g B^T)")
def _():
    r = np.random.default_rng(1)
    base = S.Linear(5, 4, r)
    lora = S.LoRALinear(base, 2, 4.0, r)
    x = r.standard_normal((3, 5))
    close(lora.forward(x), base.forward(x), 1e-12, "output at initialisation (B = 0)")
    assert set(lora.params) == {"A", "B"}, f"params should be exactly A and B, got {sorted(lora.params)}"
    lora.params["B"] = r.standard_normal((2, 4))
    close(lora.merged().forward(x), lora.forward(x), 1e-12, "merged layer against the adapted one")
    close(lora.forward(x) - base.forward(x), 2.0 * x @ lora.params["A"] @ lora.params["B"], 1e-12, "the update is (alpha / r) x A B")
    G = r.standard_normal((3, 4))
    loss = lambda: float(np.sum(lora.forward(x) * G))
    lora.forward(x); gx = lora.backward(G)
    for k in ("A", "B"):
        idxs = some(lora.params[k].shape, 5, 2)
        lora.forward(x); lora.backward(G)
        close([lora.grads[k][i] for i in idxs], numgrad(loss, lora.params[k], idxs), 1e-6, f"gradient of {k}")
    close(gx, np.array([numgrad(loss, x, [(i, j)])[0] for i in range(3) for j in range(5)]).reshape(3, 5), 1e-6, "gradient of the input")
    before = lora.forward(x).copy()
    base.params["W"] += 1; base.params["b"] += 1
    close(lora.forward(x), before, 0, "LoRALinear must keep its own copies of W0 and b0")


# ---------------------------------------------------------------- Part 2: probabilities

@test("seq_logprob: a proper distribution over responses", "sum, over the 3 positions, log_softmax(logits of features(x, y[:, :i]))[y_i]")
def _():
    rng, ref, R = setup()
    lp = S.all_logprobs(ref)
    close(np.exp(lp).sum(1), np.ones(10), 1e-10, "probabilities of the 216 responses for each prompt")
    x, y = np.array([3]), np.array([[2, 0, 5]])
    manual = 0.0
    for i in range(3):
        z = ref.logits(S.features(x, y[:, :i]))[0]
        manual += z[y[0, i]] - np.log(np.sum(np.exp(z)))
    close(S.seq_logprob(ref, x, y), [manual], 1e-10, "log pi('2 0 5' | 3)")


@test("seq_logprob_backward: the gradient summed over positions", "for each position, backpropagate w * (one_hot(y_i) - softmax), then add up the three positions' gradients")
def _():
    rng, ref, R = setup()
    x = rng.integers(0, 10, 7); y = rng.integers(0, 6, (7, 3)); w = rng.standard_normal(7)
    f = lambda: float(w @ S.seq_logprob(ref, x, y))
    S.seq_logprob_backward(ref, x, y, w)
    grads = [{k: v.copy() for k, v in l.grads.items()} for l in ref.trainable()]
    for l, gd in zip(ref.trainable(), grads):
        for k, p in l.params.items():
            idxs = some(p.shape, 4, 3)
            close([gd[k][i] for i in idxs], numgrad(f, p, idxs), 1e-6, f"gradient of {k} in a {type(l).__name__}")


@test("sample: responses drawn from the policy", "sample token i from softmax(logits of features(x, y[:, :i])), one position at a time")
def _():
    rng, ref, R = setup()
    p = np.exp(S.all_logprobs(ref))[6]
    y = S.sample(ref, np.full(20000, 6), np.random.default_rng(4))
    assert y.shape == (20000, 3), f"shape {y.shape}, expected (20000, 3)"
    idx = y @ np.array([36, 6, 1])
    freq = np.bincount(idx, minlength=216) / 20000
    top = np.argsort(p)[-5:]
    close(freq[top], p[top], 0.015, "frequencies of the 5 likeliest responses to prompt 6")


# ---------------------------------------------------------------- Part 3: reward models

@test("bradley_terry_loss: value, gradients, stability", "-log sigmoid(d) = logaddexp(0, -d); its derivative in d is -sigmoid(-d)")
def _():
    r = np.random.default_rng(5)
    rw, rl = r.standard_normal(6), r.standard_normal(6)
    loss, gw, gl = S.bradley_terry_loss(rw, rl)
    close(loss, np.mean(np.log(1 + np.exp(-(rw - rl)))), 1e-12, "loss")
    f = lambda: S.bradley_terry_loss(rw, rl)[0]
    close(gw, numgrad(f, rw, [(i,) for i in range(6)]), 1e-8, "gradient for the winners")
    close(gl, numgrad(f, rl, [(i,) for i in range(6)]), 1e-8, "gradient for the losers")
    big = S.bradley_terry_loss(np.array([-1000.0, 1000.0]), np.array([0.0, 0.0]))[0]
    close(big, 500.0, 1e-9, "loss with reward differences of -1000 and +1000")


@test("fit_reward_model recovers a linear reward", "gradient descent on the mean BT loss plus (l2 / 2) |w|^2, from w = 0")
def _():
    r = np.random.default_rng(6)
    w_true = r.standard_normal(5)
    a, b = r.random((20000, 5)) * 2, r.random((20000, 5)) * 2
    pa = 1 / (1 + np.exp(-(a - b) @ w_true))
    first = r.random(20000) < pa
    pw, pl = np.where(first[:, None], a, b), np.where(first[:, None], b, a)
    w = S.fit_reward_model(pw, pl, steps=3000, lr=0.5, l2=0.0)
    close(w, w_true, 0.15, "fitted weights against the true ones (20,000 comparisons)")


# ---------------------------------------------------------------- Part 4: KL-regularised optimum and RLHF

@test("kl_optimal_policy and kl_divergence", "pi_ref exp(r / beta), normalised per row; subtract the row maximum of log pi_ref + r / beta before exponentiating")
def _():
    rng, ref, R = setup()
    lp = S.all_logprobs(ref)
    for beta in (1.0, 0.1, 1e-3):
        q = S.kl_optimal_policy(lp, R, beta)
        close(q.sum(1), np.ones(10), 1e-10, f"rows sum to 1 at beta = {beta}")
        J = np.sum(q * R, 1) - beta * S.kl_divergence(q, lp)
        logZ = np.log(np.sum(np.exp(lp + R / beta - (lp + R / beta).max(1, keepdims=True)), 1)) + (lp + R / beta).max(1)
        close(J, beta * logZ, 1e-6 * max(1, 1 / beta) * beta, f"J(pi*) = beta log Z at beta = {beta}")
    p = np.array([[0.5, 0.5, 0.0]]); lr_ = np.log(np.array([[0.25, 0.25, 0.5]]))
    close(S.kl_divergence(p, lr_), [np.log(2)], 1e-12, "KL with a zero entry")


@test("rlhf_step: the gradient it follows", "R = reward - beta (log pi - log pi_ref); advantage = R minus the mean R of the same prompt's group; weights advantage / (number of samples)")
def _():
    rng, ref, R = setup(4)
    pol = copy.deepcopy(ref)
    for l in pol.trainable():
        for k in l.params:
            l.params[k] = l.params[k] + 0.3 * rng.standard_normal(l.params[k].shape)
    def fake_sample(policy, x, rng_):
        x = np.asarray(x); out = np.zeros((len(x), 3), int); seen = {}
        for n, xi in enumerate(x):
            k = seen.get(int(xi), 0); seen[int(xi)] = k + 1
            out[n] = S.RESPONSES[(37 * int(xi) + 53 * k) % 216]
        return out
    real = S.sample
    S.sample = fake_sample
    try:
        rec = Recorder(pol.trainable())
        S.rlhf_step(pol, ref, S.true_reward, rec, 0.5, 4, rng)
    finally:
        S.sample = real
    assert rec.calls == 1 and rec.sign == -1, "call opt.step(sign=-1) exactly once, to ascend"
    x = np.repeat(np.arange(10), 4); y = fake_sample(None, x, None)
    Rv = S.true_reward(x, y) - 0.5 * (S.seq_logprob(pol, x, y) - S.seq_logprob(ref, x, y))
    adv = Rv - Rv.reshape(10, 4).mean(1).repeat(4)
    f = lambda: float(adv @ S.seq_logprob(pol, x, y)) / len(x)
    for l, gd in zip(pol.trainable(), rec.grads):
        for k, p in l.params.items():
            idxs = some(p.shape, 4, 9)
            close([gd[k][i] for i in idxs], numgrad(f, p, idxs), 1e-7, f"gradient of {k}")


@test("rlhf_step approaches the KL-regularised optimum", "advantage = R - (mean of R within the prompt's group); gradient of mean(adv * log pi); opt.step(sign=-1)")
def _():
    rng, ref, R = setup(1)
    lp_ref = S.all_logprobs(ref)
    pol = copy.deepcopy(ref)
    opt = S.Adam(pol.trainable(), 3e-3)
    for _ in range(500):
        S.rlhf_step(pol, ref, S.true_reward, opt, 1.0, 16, rng)
    q = np.exp(S.all_logprobs(pol))
    q_star = S.kl_optimal_policy(lp_ref, R, 1.0)
    gap = np.mean(np.sum(q * (np.log(q) - np.log(q_star)), 1))
    p0 = np.exp(lp_ref)
    gap0 = np.mean(np.sum(p0 * (lp_ref - np.log(q_star)), 1))
    er, er_star, er_ref = np.mean(np.sum(q * R, 1)), np.mean(np.sum(q_star * R, 1)), np.mean(np.sum(np.exp(lp_ref) * R, 1))
    assert gap < 0.4 * gap0, f"KL(policy || optimum) went from {gap0:.3f} to {gap:.3f} in 500 steps at beta = 1; a working step gets to about a quarter of where it started"
    assert er > er_ref + 0.5 * (er_star - er_ref), f"expected true reward {er:.2f}: started at {er_ref:.2f}, optimum {er_star:.2f}"


# ---------------------------------------------------------------- Part 5: DPO and IPO

@test("dpo_loss and ipo_loss: values and gradients", "DPO: h = beta (margin); loss logaddexp(0, -h); dloss/dlp_w = -beta sigmoid(-h) / n. IPO: (Delta - 1/(2 tau))^2")
def _():
    r = np.random.default_rng(7)
    a, b, c, d = (r.standard_normal(5) for _ in range(4))
    for fn, coef, ref in [(S.dpo_loss, 0.3, lambda: np.mean(np.log(1 + np.exp(-0.3 * ((a - c) - (b - d)))))),
                          (S.ipo_loss, 0.3, lambda: np.mean(((a - c) - (b - d) - 1 / 0.6) ** 2))]:
        loss, gw, gl = fn(a, b, c, d, coef)
        close(loss, ref(), 1e-12, f"{fn.__name__} value")
        f = lambda: fn(a, b, c, d, coef)[0]
        close(gw, numgrad(f, a, [(i,) for i in range(5)]), 1e-8, f"{fn.__name__} gradient for the winners")
        close(gl, numgrad(f, b, [(i,) for i in range(5)]), 1e-8, f"{fn.__name__} gradient for the losers")
    close(S.dpo_loss(np.array([1000.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 1.0)[0], 0.0, 1e-12, "DPO loss at a huge margin")


@test("preference_step: one descent step on the pairwise loss", "one seq_logprob_backward call on the concatenated winners and losers, with weights (g_w, g_l); then opt.step()")
def _():
    rng, ref, R = setup(2)
    pol = copy.deepcopy(ref)
    for l in pol.trainable():
        for k in l.params:
            l.params[k] = l.params[k] + 0.2 * rng.standard_normal(l.params[k].shape)
    x = rng.integers(0, 10, 6); yw = rng.integers(0, 6, (6, 3)); yl = rng.integers(0, 6, (6, 3))
    rw, rl = S.seq_logprob(ref, x, yw), S.seq_logprob(ref, x, yl)
    rec = Recorder(pol.trainable())
    loss = S.preference_step(pol, x, yw, yl, rw, rl, rec, S.dpo_loss, 0.5)
    assert rec.calls == 1 and rec.sign == 1.0, "call opt.step() exactly once, descending"
    f = lambda: S.dpo_loss(S.seq_logprob(pol, x, yw), S.seq_logprob(pol, x, yl), rw, rl, 0.5)[0]
    close(loss, f(), 1e-12, "returned loss")
    for l, gd in zip(pol.trainable(), rec.grads):
        for k, p in l.params.items():
            idxs = some(p.shape, 4, 8)
            close([gd[k][i] for i in idxs], numgrad(f, p, idxs), 1e-7, f"gradient of {k}")


@test("end to end: DPO improves the true reward", "if the other checks pass, check the sign of the step: preference_step descends the loss")
def _():
    rng, ref, R = setup(3)
    n = 2000
    x = rng.integers(0, 10, n); y1 = S.sample(ref, x, rng); y2 = S.sample(ref, x, rng)
    r1, r2 = S.true_reward(x, y1), S.true_reward(x, y2)
    first = rng.random(n) < 1 / (1 + np.exp(-(r1 - r2)))
    yw, yl = np.where(first[:, None], y1, y2), np.where(first[:, None], y2, y1)
    rw, rl = S.seq_logprob(ref, x, yw), S.seq_logprob(ref, x, yl)
    pol = copy.deepcopy(ref); opt = S.Adam(pol.trainable(), 1e-3)
    for _ in range(15):
        perm = rng.permutation(n)
        for i in range(0, n, 200):
            b = perm[i:i + 200]
            S.preference_step(pol, x[b], yw[b], yl[b], rw[b], rl[b], opt, S.dpo_loss, 0.25)
    before = np.mean(np.sum(np.exp(S.all_logprobs(ref)) * R, 1))
    after = np.mean(np.sum(np.exp(S.all_logprobs(pol)) * R, 1))
    assert after > before + 0.4, f"expected true reward went from {before:.2f} to {after:.2f}; a working DPO gains about 0.8"


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
