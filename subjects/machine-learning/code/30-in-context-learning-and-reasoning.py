import numpy as np
from math import comb

rng = np.random.default_rng(0)

# 1. Decoding rules applied to one next-token distribution (Zipf-like over 1,000 tokens).
p = 1 / np.arange(1, 1001) ** 1.1; p /= p.sum()
H = lambda q: -np.sum(q[q > 0] * np.log2(q[q > 0]))
def top_k(p, k):
    q = np.where(np.arange(len(p)) < k, p, 0); return q / q.sum()          # p is sorted, largest first
def top_p(p, thr):
    keep = np.cumsum(p) - p < thr                                        # smallest prefix with mass >= thr
    q = np.where(keep, p, 0); return q / q.sum(), keep.sum()
def temperature(p, tau):
    q = p ** (1 / tau); return q / q.sum()
print(f"model's distribution: entropy {H(p):.2f} bits, top token {p[0]:.3f}, top 10 hold {p[:10].sum():.3f}")
print(f"temperature 0.7: entropy {H(temperature(p, 0.7)):.2f} bits;  top-k 50: {H(top_k(p, 50)):.2f} bits")
for thr in [0.5, 0.9, 0.95]:
    q, n = top_p(p, thr)
    print(f"top-p {thr}: keeps {n} tokens, entropy {H(q):.2f} bits")

# 2. In-context learning as an algorithm: one layer of linear self-attention, with suitable
# weights, makes exactly the prediction of one gradient-descent step on the in-context examples
# (von Oswald et al., 2023). Tokens are (x_i, y_i); the query token is (x_q, 0).
d, n_ctx, lr = 5, 20, 0.1
w_true = rng.standard_normal(d)
Xc = rng.standard_normal((n_ctx, d)); yc = Xc @ w_true
xq = rng.standard_normal(d)
Z = np.vstack([np.column_stack([Xc, yc]), np.append(xq, 0)])             # (n_ctx + 1) tokens of size d + 1
WK = np.zeros((d + 1, d + 1)); WK[:d, :d] = np.eye(d)                      # keys and queries read x
WQ = WK.copy()
WV = np.zeros((d + 1, d + 1)); WV[d, d] = lr / n_ctx                       # values read y, scaled
attn = (Z @ WQ) @ (Z[:-1] @ WK).T                                          # linear attention, no softmax,
lsa_pred = (attn @ (Z[:-1] @ WV))[-1, d]                                   # over the context tokens only
w_gd = lr * Xc.T @ yc / n_ctx                                              # one GD step on 1/2 mean sq. error from w = 0
print(f"linear self-attention prediction {lsa_pred:.6f}; one step of gradient descent {w_gd @ xq:.6f}")

# 3. Chain of thought as decomposition: the parity of 24 random bits. The same small MLP is
# trained either to output the parity directly, or to produce the running parity one step at
# a time, where each step sees the bits, which step it is on, and its own previous output.
n_bits, n_train = 24, 5000
def mlp_train(X, y, steps=4000, hidden=128, lr=0.01, seed=0):
    r = np.random.default_rng(seed)
    W1 = r.standard_normal((X.shape[1], hidden)) * np.sqrt(2 / X.shape[1]); b1 = np.zeros(hidden)
    w2 = r.standard_normal(hidden) * np.sqrt(1 / hidden); b2 = 0.0
    params = [W1, b1, w2]; ms = [np.zeros_like(q) for q in params]; vs = [np.zeros_like(q) for q in params]
    for t in range(1, steps + 1):
        idx = r.integers(0, len(X), 128)
        h = np.maximum(X[idx] @ W1 + b1, 0); z = h @ w2 + b2
        g = (1 / (1 + np.exp(-z)) - y[idx]) / len(idx)                     # logistic loss gradient
        gh = np.outer(g, w2) * (h > 0)
        grads = [X[idx].T @ gh, gh.sum(0), h.T @ g]; b2 -= lr * g.sum()
        for q, gq, m, v in zip(params, grads, ms, vs):
            m *= 0.9; m += 0.1 * gq; v *= 0.999; v += 0.001 * gq * gq
            q -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return lambda X: (np.maximum(X @ W1 + b1, 0) @ w2 + b2) > 0
bits = rng.integers(0, 2, (n_train, n_bits)); test = rng.integers(0, 2, (2000, n_bits))
direct = mlp_train(2.0 * bits - 1, bits.sum(1) % 2)
print(f"direct: train accuracy {np.mean(direct(2.0 * bits - 1) == bits.sum(1) % 2):.3f},"
      f" test accuracy {np.mean(direct(2.0 * test - 1) == test.sum(1) % 2):.3f}")
def step_inputs(B, i, prev):                       # the bits, a one-hot of the step, the previous output
    return np.column_stack([2.0 * B - 1, np.eye(n_bits)[np.full(len(B), i)], 2.0 * prev - 1])
running = np.cumsum(bits, 1) % 2
Xs = np.vstack([step_inputs(bits, i, running[:, i - 1] if i else np.zeros(n_train)) for i in range(n_bits)])
ys = np.concatenate([running[:, i] for i in range(n_bits)])
step = mlp_train(Xs, ys)
prev = np.zeros(len(test))
for i in range(n_bits):                            # generate the chain, feeding back its own outputs
    prev = step(step_inputs(test, i, prev)).astype(float)
run_t = np.cumsum(test, 1) % 2
per_step = np.mean([step(step_inputs(test, i, run_t[:, i - 1] if i else np.zeros(len(test)))) == run_t[:, i]
                    for i in range(n_bits)])
print(f"chain of thought: each step right {per_step:.4f} of the time on test strings;"
      f" final answer right {np.mean(prev == test.sum(1) % 2):.3f} after {n_bits} steps")

# 4. Self-consistency: sample several reasoning chains and take the majority answer. A model
# that is right 40% of the time and spreads its wrong answers over many values.
def majority_accuracy(p_right, n_wrong, N, trials=20000, seed=1):
    r = np.random.default_rng(seed)
    probs = np.append(p_right, np.full(n_wrong, (1 - p_right) / n_wrong))
    wins = 0.0
    for counts in r.multinomial(N, probs, size=trials):
        top = np.flatnonzero(counts == counts.max())
        wins += (0 in top) / len(top)                                      # ties broken at random
    return wins / trials
for N in [1, 5, 15, 41]:
    print(f"majority of {N:2d} samples: accuracy {majority_accuracy(0.4, 10, N):.3f}"
          f" (wrong answers spread over 10 values) vs {majority_accuracy(0.4, 1, N):.3f} (one wrong answer)")

# 5. pass@k, estimated without bias from n samples of which c are correct (Chen et al., 2021).
n, c = 20, 3
est = lambda k: 1 - comb(n - c, k) / comb(n, k)
print(f"pass@k from {c} correct of {n} samples: " + ", ".join(f"k={k}: {est(k):.3f}" for k in [1, 5, 10]))

# model's distribution: entropy 6.80 bits, top token 0.179, top 10 hold 0.481
# temperature 0.7: entropy 3.85 bits;  top-k 50: 4.37 bits
# top-p 0.5: keeps 12 tokens, entropy 2.98 bits
# top-p 0.9: keeps 349 tokens, entropy 6.01 bits
# top-p 0.95: keeps 582 tokens, entropy 6.40 bits
# linear self-attention prediction -0.017513; one step of gradient descent -0.017513
# direct: train accuracy 0.935, test accuracy 0.493
# chain of thought: each step right 0.9998 of the time on test strings; final answer right 0.995 after 24 steps
# majority of  1 samples: accuracy 0.405 (wrong answers spread over 10 values) vs 0.398 (one wrong answer)
# majority of  5 samples: accuracy 0.637 (wrong answers spread over 10 values) vs 0.318 (one wrong answer)
# majority of 15 samples: accuracy 0.923 (wrong answers spread over 10 values) vs 0.213 (one wrong answer)
# majority of 41 samples: accuracy 0.999 (wrong answers spread over 10 values) vs 0.099 (one wrong answer)
# pass@k from 3 correct of 20 samples: k=1: 0.150, k=5: 0.601, k=10: 0.895
