import numpy as np

rng = np.random.default_rng(0)

# 1. FlashAttention's arithmetic: attention computed block by block with an online softmax,
# never forming the full n x n score matrix, gives the same answer as the direct formula.
def attention(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    P = np.exp(S - S.max(1, keepdims=True))
    return (P / P.sum(1, keepdims=True)) @ V

def tiled_attention(Q, K, V, block=64):
    n, d = Q.shape
    out = np.zeros_like(Q)
    for i in range(0, n, block):                       # one block of queries at a time
        q = Q[i:i + block] / np.sqrt(d)
        m = np.full(len(q), -np.inf)                   # running max of the scores
        l = np.zeros(len(q))                           # running sum of exp(score - m)
        acc = np.zeros((len(q), V.shape[1]))           # running sum of exp(score - m) * v
        for j in range(0, n, block):                   # stream over blocks of keys and values
            s = q @ K[j:j + block].T
            m_new = np.maximum(m, s.max(1))
            scale = np.exp(m - m_new)                  # rescale what was summed under the old max
            p = np.exp(s - m_new[:, None])
            l = l * scale + p.sum(1)
            acc = acc * scale[:, None] + p @ V[j:j + block]
            m = m_new
        out[i:i + block] = acc / l[:, None]
    return out

n, d = 1024, 64
Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))
print(f"tiled vs direct attention: max difference {np.abs(tiled_attention(Q, K, V) - attention(Q, K, V)).max():.1e};"
      f" the direct version stores {n * n:,} scores, the tiled one {64 * 64:,} at a time")

# 2. The KV cache: when generating token t, the keys and values of tokens 1..t-1 do not change
# (causal attention), so keep them instead of recomputing them.
d_model, T = 64, 200
Wq, Wk, Wv = (rng.standard_normal((d_model, d_model)) / np.sqrt(d_model) for _ in range(3))
X = rng.standard_normal((T, d_model))
def causal_full(X):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(d_model) + np.triu(np.full((len(X), len(X)), -np.inf), 1)
    P = np.exp(S - S.max(1, keepdims=True))
    return (P / P.sum(1, keepdims=True)) @ V
cache_k, cache_v, outs = [], [], []
for t in range(T):                                     # one token at a time, as in generation
    x = X[t]
    cache_k.append(x @ Wk); cache_v.append(x @ Wv)
    s = np.array(cache_k) @ (x @ Wq) / np.sqrt(d_model)
    p = np.exp(s - s.max())
    outs.append(p @ np.array(cache_v) / p.sum())
print(f"KV-cached decoding vs recomputing everything: max difference {np.abs(np.array(outs) - causal_full(X)).max():.1e}")
proj_no_cache = sum(3 * t * d_model * d_model for t in range(1, T + 1))
print(f"projection multiply-adds for {T} tokens: {proj_no_cache:,} without a cache, {3 * T * d_model * d_model:,} with one")

# 3. Quantisation: store weights as small integers times a scale. A weight matrix with a few
# outlier input channels, as in real LLMs, quantised in three ways; error of y = x W.
W = rng.standard_normal((512, 512)) * 0.02
W[:, rng.choice(512, 4, replace=False)] *= 30         # 4 outlier output columns
x = rng.standard_normal((256, 512))
def quantise(W, bits, axis=None, group=None):
    qmax = 2 ** (bits - 1) - 1
    if group:                                          # one scale per group of `group` rows in each column
        Wg = W.reshape(-1, group, W.shape[1])
        s = np.abs(Wg).max(1, keepdims=True) / qmax
        return (np.round(Wg / s) * s).reshape(W.shape)
    s = np.abs(W).max(axis, keepdims=True) / qmax
    return np.round(W / s) * s
y = x @ W
for name, Wq_ in [("int8, one scale per tensor", quantise(W, 8)),
                  ("int8, one scale per column", quantise(W, 8, axis=0)),
                  ("int4, one scale per tensor", quantise(W, 4)),
                  ("int4, one scale per column", quantise(W, 4, axis=0)),
                  ("int4, groups of 64 per column", quantise(W, 4, group=64))]:
    err = np.linalg.norm(x @ Wq_ - y) / np.linalg.norm(y)
    print(f"{name:31s} relative output error {err:.4f}")

# 4. LoRA: adapt a frozen W0 by a low-rank update B A. The "task" needs a rank-4 change to W0.
# Full fine-tuning trains all d*k weights; LoRA trains r*(d+k).
d_in, d_out, n_train = 256, 256, 2000
W0 = rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)
delta = rng.standard_normal((d_in, 4)) @ rng.standard_normal((4, d_out)) / np.sqrt(d_in) * 0.5
Xtr, Xte = rng.standard_normal((n_train, d_in)), rng.standard_normal((1000, d_in))
Ytr = Xtr @ (W0 + delta) + 0.1 * rng.standard_normal((n_train, d_out)); Yte = Xte @ (W0 + delta)
def fit(r, steps=1500, lr=0.02):
    """Adam on the mean squared error; r = None means full fine-tuning."""
    if r is None:
        params = [np.zeros((d_in, d_out))]             # a dense update dW
    else:                                              # LoRA: A random, B zero, so B A = 0 at the start
        params = [rng.standard_normal((d_in, r)) / np.sqrt(d_in), np.zeros((r, d_out))]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        dW = params[0] if r is None else params[0] @ params[1]
        G = 2 * Xtr.T @ (Xtr @ (W0 + dW) - Ytr) / (n_train * d_out)
        grads = [G] if r is None else [G @ params[1].T, params[0].T @ G]
        for p, g, mi, vi in zip(params, grads, m, v):
            mi *= 0.9; mi += 0.1 * g; vi *= 0.999; vi += 0.001 * g * g
            p -= lr * (mi / (1 - 0.9 ** t)) / (np.sqrt(vi / (1 - 0.999 ** t)) + 1e-8)
    dW = params[0] if r is None else params[0] @ params[1]
    return sum(p.size for p in params), np.mean((Xte @ (W0 + dW) - Yte) ** 2)
print(f"no fine-tuning: test error {np.mean((Xte @ W0 - Yte) ** 2):.2e}")
for r in [1, 2, 4, 8, None]:
    n_p, err = fit(r)
    print(f"{'full fine-tuning' if r is None else f'LoRA rank {r}':16s} {n_p:6,} trainable parameters, test error {err:.2e}")

# 5. Speculative decoding: a cheap draft model q proposes a token, the big model p accepts it
# with probability min(1, p/q), and on rejection samples from the normalised (p - q)+.
# The tokens that come out are distributed exactly as p, and the acceptance rate is sum min(p, q).
Vsz = 8
p = rng.dirichlet(np.ones(Vsz)); q = 0.6 * p + 0.4 * rng.dirichlet(np.ones(Vsz))
def speculative_sample(r):
    x = r.choice(Vsz, p=q)
    if r.random() < min(1, p[x] / q[x]):
        return x, True
    resid = np.maximum(p - q, 0)
    return r.choice(Vsz, p=resid / resid.sum()), False
r = np.random.default_rng(1)
draws = [speculative_sample(r) for _ in range(200_000)]
freq = np.bincount([x for x, _ in draws], minlength=Vsz) / len(draws)
print(f"speculative sampling: largest |frequency - p| = {np.abs(freq - p).max():.4f} over 200,000 draws;"
      f" accepted {np.mean([a for _, a in draws]):.4f} of drafts, predicted sum min(p, q) = {np.minimum(p, q).sum():.4f}")

# tiled vs direct attention: max difference 3.9e-16; the direct version stores 1,048,576 scores, the tiled one 4,096 at a time
# KV-cached decoding vs recomputing everything: max difference 1.8e-15
# projection multiply-adds for 200 tokens: 246,988,800 without a cache, 2,457,600 with one
# int8, one scale per tensor      relative output error 0.0825
# int8, one scale per column      relative output error 0.0068
# int4, one scale per tensor      relative output error 0.3706
# int4, one scale per column      relative output error 0.1255
# int4, groups of 64 per column   relative output error 0.1070
# no fine-tuning: test error 9.52e-01
# LoRA rank 1         512 trainable parameters, test error 6.43e-01
# LoRA rank 2       1,024 trainable parameters, test error 3.64e-01
# LoRA rank 4       2,048 trainable parameters, test error 4.15e-05
# LoRA rank 8       4,096 trainable parameters, test error 1.37e-04
# full fine-tuning 65,536 trainable parameters, test error 1.49e-03
# speculative sampling: largest |frequency - p| = 0.0018 over 200,000 draws; accepted 0.8268 of drafts, predicted sum min(p, q) = 0.8268
