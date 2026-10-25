"""Autograder for assignment A6. Usage: python3 check.py [your_file.py]

Every gradient is compared with central finite differences, in float64. Attention is compared
with a slow loop over heads and positions written here, the KV-cached generation path with the
full forward pass, and the optimiser with a reference implementation."""
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


def numgrad(f, a, h=1e-6, idxs=None):
    g = np.zeros_like(a)
    for idx in (idxs if idxs is not None else np.ndindex(a.shape)):
        old = a[idx]
        a[idx] = old + h; up = f()
        a[idx] = old - h; dn = f()
        a[idx] = old
        g[idx] = (up - dn) / (2 * h)
    return g


def to64(*modules):
    for m in modules:
        for k in m.params:
            m.params[k] = m.params[k].astype(np.float64)


def gpt64(vocab=7, T=6, d=8, heads=2, n_layer=2):
    model = S.GPT(vocab, T, d=d, heads=heads, n_layer=n_layer, seed=3)
    to64(*model.modules())
    # larger random weights than the initialisation, so that every path carries gradient
    r = np.random.default_rng(5)
    for m in model.modules():
        for k, p in m.params.items():
            m.params[k] = p + 0.3 * r.standard_normal(p.shape)
    return model


def check_layer_grads(layer, x, modules, tol, what, n_entries=None):
    """Loss = sum(out * R). Compare backward's input and parameter gradients with numgrad."""
    R = np.random.default_rng(1).standard_normal(layer.forward(x).shape)
    loss = lambda: float(np.sum(layer.forward(x) * R))
    layer.forward(x); gx = layer.backward(R)
    close(gx, numgrad(loss, x), tol, f"{what}: gradient with respect to the input")
    for m in modules:
        for k, p in m.params.items():
            idxs = None
            if n_entries:
                rr = np.random.default_rng(2)
                idxs = [tuple(rr.integers(0, s) for s in p.shape) for _ in range(n_entries)]
            layer.forward(x); layer.backward(R)
            g = m.grads[k]
            ng = numgrad(loss, p, idxs=idxs)
            if idxs is not None:
                g = np.array([g[i] for i in idxs]); ng = np.array([ng[i] for i in idxs])
            close(g, ng, tol, f"{what}: gradient of {type(m).__name__}.{k}")


# ---------------------------------------------------------------- Part 1

@test("softmax", "subtract the maximum along the axis before exponentiating")
def _():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, -1000.0]])
    p = S.softmax(z)
    close(p[0], np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum(), 1e-12, "softmax([1, 2, 3])")
    close(p[1], [0.5, 0.5, 0.0], 1e-12, "softmax([1000, 1000, -1000])")
    close(S.softmax(z, axis=0).sum(0), np.ones(3), 1e-12, "softmax along axis 0 sums to 1")


@test("cross_entropy: value and gradient", "use log-softmax; the gradient is (softmax - one_hot) / number of positions")
def _():
    r = np.random.default_rng(0)
    logits = r.standard_normal((2, 3, 5)); y = r.integers(0, 5, (2, 3))
    loss, g = S.cross_entropy(logits, y)
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    close(loss, -np.mean(np.take_along_axis(lp, y[..., None], -1)), 1e-10, "loss")
    close(g, numgrad(lambda: S.cross_entropy(logits, y)[0], logits), 1e-7, "gradient")
    big, _ = S.cross_entropy(np.array([[1000.0, 0.0]]), np.array([1]))
    close(big, 1000.0, 1e-9, "loss with a logit of 1,000")


@test("LayerNorm: forward and backward", "normalise over the last axis with the biased variance; see lesson 14 for the backward pass")
def _():
    r = np.random.default_rng(1)
    ln = S.LayerNorm(6); to64(ln)
    x = r.standard_normal((2, 3, 6)) * 3 + 1
    y = ln.forward(x)
    close(y.mean(-1), np.zeros((2, 3)), 1e-10, "mean of the output with gain 1, bias 0")
    close(y.std(-1), np.sqrt(x.var(-1) / (x.var(-1) + 1e-5)), 1e-8, "standard deviation of the output")
    ln.params["g"] = r.standard_normal(6); ln.params["b"] = r.standard_normal(6)
    check_layer_grads(ln, x, [ln], 1e-6, "LayerNorm")


@test("GELU: forward and backward", "0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))); differentiate with the product rule")
def _():
    g = S.GELU()
    x = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0])
    c = np.sqrt(2 / np.pi)
    close(g.forward(x), 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3))), 1e-12, "values")
    x = np.random.default_rng(2).standard_normal((3, 4))
    check_layer_grads(g, x, [], 1e-7, "GELU")


# ---------------------------------------------------------------- Part 2

def slow_attention(att_layer, x):
    W, b = att_layer.qkv.params["W"], att_layer.qkv.params["b"]
    Wo, bo = att_layer.proj.params["W"], att_layer.proj.params["b"]
    B, T, d = x.shape; h = att_layer.h; dh = d // h
    qkv = x @ W + b
    out = np.zeros((B, T, d)); att = np.zeros((B, h, T, T))
    for bb in range(B):
        for hh in range(h):
            q = qkv[bb, :, hh * dh:(hh + 1) * dh]
            k = qkv[bb, :, d + hh * dh:d + (hh + 1) * dh]
            v = qkv[bb, :, 2 * d + hh * dh:2 * d + (hh + 1) * dh]
            for t in range(T):
                s = k[:t + 1] @ q[t] / np.sqrt(dh)
                a = np.exp(s - s.max()); a /= a.sum()
                att[bb, hh, t, :t + 1] = a
                out[bb, t, hh * dh:(hh + 1) * dh] = a @ v[:t + 1]
    return out @ Wo + bo, att


def make_attention(d=8, heads=2):
    r = np.random.default_rng(4)
    att = S.CausalSelfAttention(d, heads, r)
    to64(att.qkv, att.proj)
    att.qkv.params["W"] = att.qkv.params["W"] * 2
    return att


@test("attention forward: matches a loop over heads and positions", "split q, k, v as [q | k | v] along the last axis, then reshape each to (B, heads, T, d_head)")
def _():
    att = make_attention()
    x = np.random.default_rng(5).standard_normal((2, 5, 8))
    out = att.forward(x)
    ref, ref_att = slow_attention(att, x)
    close(out, ref, 1e-10, "output")
    close(att.att, ref_att, 1e-10, "attention weights self.att")


@test("attention is causal", "mask scores above the diagonal with -inf before the softmax")
def _():
    att = make_attention()
    x = np.random.default_rng(6).standard_normal((1, 6, 8))
    out1 = att.forward(x).copy()
    x2 = x.copy(); x2[0, 4:] += 5
    out2 = att.forward(x2)
    close(out2[0, :4], out1[0, :4], 1e-12, "outputs at positions 0-3 after changing positions 4-5")


@test("attention backward", "lesson 19: dS = A * (dA - sum(dA * A, last axis)), then dQ = dS K and dK = dS^T Q, scaled by 1/sqrt(d_head)")
def _():
    att = make_attention()
    x = np.random.default_rng(7).standard_normal((2, 4, 8))
    check_layer_grads(att, x, [att.qkv, att.proj], 1e-6, "attention")


@test("attention step with a KV cache equals the full forward pass", "append this position's k and v to cache['k'] and cache['v'] along the time axis (axis 2)")
def _():
    att = make_attention()
    x = np.random.default_rng(8).standard_normal((2, 5, 8))
    full = att.forward(x).copy()
    cache = {}
    for t in range(5):
        out = att.step(x[:, t:t + 1], cache)
        close(out[:, 0], full[:, t], 1e-10, f"step output at position {t}")
    assert cache["k"].shape == (2, 2, 5, 4), f"cache['k'] has shape {cache['k'].shape}, expected (2, 2, 5, 4)"


# ---------------------------------------------------------------- Part 3

@test("Block backward", "the residual connections add the incoming gradient to the gradient through each branch")
def _():
    r = np.random.default_rng(9)
    blk = S.Block(8, 2, r, 0.3)
    mods = [blk.ln1, blk.attn.qkv, blk.attn.proj, blk.ln2, blk.fc1, blk.fc2]
    to64(*mods)
    for m in mods:
        for k, p in m.params.items():
            m.params[k] = p + 0.3 * r.standard_normal(p.shape)
    x = r.standard_normal((2, 4, 8))
    check_layer_grads(blk, x, mods, 1e-6, "Block", n_entries=4)


@test("GPT backward, through every parameter", "the position embedding's gradient is the token gradient summed over the batch")
def _():
    model = gpt64()
    r = np.random.default_rng(10)
    idx, y = r.integers(0, 7, (2, 6)), r.integers(0, 7, (2, 6))
    loss = lambda: S.cross_entropy(model.forward(idx), y)[0]
    _, g = S.cross_entropy(model.forward(idx), y); model.backward(g)
    grads = [{k: v.copy() for k, v in m.grads.items()} for m in model.modules()]
    for m, gd in zip(model.modules(), grads):
        for k, p in m.params.items():
            rr = np.random.default_rng(11)
            idxs = [tuple(rr.integers(0, s) for s in p.shape) for _ in range(3)]
            ng = numgrad(loss, p, idxs=idxs)
            close([gd[k][i] for i in idxs], [ng[i] for i in idxs], 1e-6, f"{type(m).__name__}.{k}")


@test("GPT step with KV caches equals the full forward pass", "embed the token plus position t, then run each Block.step with its own cache dict")
def _():
    model = gpt64()
    idx = np.random.default_rng(12).integers(0, 7, (3, 6))
    full = model.forward(idx)
    caches = [{} for _ in model.blocks]
    for t in range(6):
        close(model.step(idx[:, t], t, caches), full[:, t], 1e-9, f"logits at position {t}")


# ---------------------------------------------------------------- Part 4

@test("get_batch", "pick B random starts, then x = data[s:s+T] and y = data[s+1:s+T+1]")
def _():
    data = np.arange(1000)
    x, y = S.get_batch(data, 16, 8, np.random.default_rng(13))
    assert x.shape == (16, 8) and y.shape == (16, 8), f"shapes {x.shape}, {y.shape}; expected (16, 8) twice"
    close(y[:, :-1], x[:, 1:], 0, "y shifted against x")
    close(y[:, -1], x[:, -1] + 1, 0, "the last target follows the last input")
    close(np.diff(x, axis=1), np.ones((16, 7)), 0, "windows are contiguous")
    assert y.max() < 1000 and x.min() >= 0
    assert len(np.unique(x[:, 0])) > 8, "starts should be random"


@test("lr_schedule", "warm-up: max_lr * step / warmup; then min_lr + 0.5 (max_lr - min_lr)(1 + cos(pi * progress))")
def _():
    f = lambda s: S.lr_schedule(s, 1.0, 10, 110, 0.1)
    close([f(1), f(5), f(10)], [0.1, 0.5, 1.0], 1e-12, "warm-up")
    close([f(60), f(110), f(500)], [0.55, 0.1, 0.1], 1e-12, "cosine midpoint, end, and after the end")
    close(f(35), 0.1 + 0.45 * (1 + np.cos(np.pi / 4)), 1e-12, "a quarter of the way through the decay")
    assert f(30) > f(31) > f(32), "the decay should be monotone"


@test("clip_grad_norm", "the global norm is the square root of the sum of squares of every gradient entry")
def _():
    a, b = S.Linear(3, 2, np.random.default_rng(0)), S.LayerNorm(2)
    a.grads = {"W": np.full((3, 2), 1.0), "b": np.full(2, 1.0)}
    b.grads = {"g": np.full(2, 1.0), "b": np.full(2, 1.0)}
    n = S.clip_grad_norm([a, b], 1.0)
    close(n, np.sqrt(12), 1e-9, "returned norm")
    close(a.grads["W"], np.full((3, 2), 1 / np.sqrt(12)), 1e-6, "clipped gradient")
    n2 = S.clip_grad_norm([a, b], 10.0)
    close(n2, 1.0, 1e-6, "norm after clipping")
    close(a.grads["W"], np.full((3, 2), 1 / np.sqrt(12)), 1e-6, "a norm below max_norm must not be changed")


@test("AdamW", "bias-correct m and v; add wd * p to the update only for parameters with ndim >= 2")
def _():
    r = np.random.default_rng(14)
    lin = S.Linear(3, 2, r); to64(lin)
    W0, b0 = lin.params["W"].copy(), lin.params["b"].copy()
    opt = S.AdamW([lin], b1=0.9, b2=0.95, eps=1e-8, wd=0.1)
    gs = [(r.standard_normal((3, 2)), r.standard_normal(2)) for _ in range(3)]
    W, b, mW, vW, mb, vb = W0.copy(), b0.copy(), 0, 0, 0, 0
    for t, (gW, gb) in enumerate(gs, 1):
        lin.grads = {"W": gW.copy(), "b": gb.copy()}
        opt.step(0.01)
        mW = 0.9 * mW + 0.1 * gW; vW = 0.95 * vW + 0.05 * gW ** 2
        mb = 0.9 * mb + 0.1 * gb; vb = 0.95 * vb + 0.05 * gb ** 2
        W = W - 0.01 * ((mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.95 ** t)) + 1e-8) + 0.1 * W)
        b = b - 0.01 * ((mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.95 ** t)) + 1e-8))
    close(lin.params["W"], W, 1e-10, "weights after 3 steps")
    close(lin.params["b"], b, 1e-10, "biases after 3 steps (no weight decay on biases)")


# ---------------------------------------------------------------- Part 5

@test("generate: greedy, top-k and sampling", "run the prompt through model.step first; temperature 0 is argmax; top_k keeps the k largest logits")
def _():
    model = gpt64(vocab=5, T=8)
    prompt = np.array([1, 3])
    seq = S.generate(model, prompt, 6, np.random.default_rng(0), temperature=0)
    assert len(seq) == 8, f"length {len(seq)}, expected 8"
    close(seq[:2], prompt, 0, "the prompt is kept")
    ref = list(prompt)
    for _ in range(6):
        ref.append(int(np.argmax(model.forward(np.array([ref]))[0, -1])))
    close(seq, ref, 0, "greedy generation against argmax of the full forward pass")
    seq_k1 = S.generate(model, prompt, 6, np.random.default_rng(1), temperature=1.0, top_k=1)
    close(seq_k1, ref, 0, "top_k = 1 must be greedy")
    logits = model.forward(prompt[None])[0, -1]
    p = np.exp(logits / 0.7 - (logits / 0.7).max()); p /= p.sum()
    firsts = [S.generate(model, prompt, 1, np.random.default_rng(100 + i), temperature=0.7)[2] for i in range(3000)]
    freq = np.bincount(firsts, minlength=5) / 3000
    close(freq, p, 0.035, "frequencies of the first sampled token at temperature 0.7")


@test("end to end: a small GPT learns a repeating text", "if the gradient checks pass, check the training loop: get_batch, clip_grad_norm, AdamW, lr_schedule")
def _():
    text = "the cat sat on the mat. the dog sat on the log. " * 40
    chars = sorted(set(text)); data = np.array([chars.index(c) for c in text])
    model = S.GPT(len(chars), 32, d=32, heads=2, n_layer=2, seed=0)
    mods = model.modules(); opt = S.AdamW(mods)
    r = np.random.default_rng(15)
    for s in range(1, 301):
        x, y = S.get_batch(data, 16, 32, r)
        loss, g = S.cross_entropy(model.forward(x), y); model.backward(g)
        S.clip_grad_norm(mods, 1.0); opt.step(S.lr_schedule(s, 3e-3, 30, 300, 3e-4))
    x, y = S.get_batch(data, 64, 32, r)
    loss = S.cross_entropy(model.forward(x), y)[0]
    assert loss < 0.35, f"loss after 300 steps is {loss:.3f}; a working model gets below 0.3 (the text is almost deterministic)"


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
