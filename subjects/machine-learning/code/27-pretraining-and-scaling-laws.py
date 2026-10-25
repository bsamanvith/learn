import os
import time
import urllib.request
import numpy as np

# 1. Data: Tiny Shakespeare (1.1 MB of plays), as characters. The last 10% is held out.
path = os.path.expanduser("~/.cache/learn-text/shakespeare.txt")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
                               "tinyshakespeare/input.txt", path)
text = open(path).read()
chars = sorted(set(text)); V = len(chars)
ids = np.array([chars.index(ch) for ch in text], dtype=np.int64)
split = int(0.9 * len(ids)); train, val = ids[:split], ids[split:]
K, E = 16, 16                                    # context of 16 characters, 16-dim embeddings
print(f"{len(text):,} characters, vocabulary {V}; {len(train):,} for training")

# 2. A family of models: Bengio-style MLP language models, embeddings -> H -> H -> softmax,
# that differ only in their width H. N counts the weights used in matrix multiplies (not the
# embedding table, which is a lookup), so training costs about 6N FLOPs per token.
def init(H, rng):
    s = lambda a: np.sqrt(1 / a)
    return {"emb": rng.standard_normal((V, E)) * 0.5,
            "W1": rng.standard_normal((K * E, H)) * s(K * E), "b1": np.zeros(H),
            "W2": rng.standard_normal((H, H)) * s(H), "b2": np.zeros(H),
            "W3": rng.standard_normal((H, V)) * s(H) * 0.1, "b3": np.zeros(V)}

def n_params(H):
    return K * E * H + H * H + H * V

def loss_and_grads(P, idx, data, grad=True):
    X = data[idx[:, None] + np.arange(K)]          # (B, K) contexts
    y = data[idx + K]                              # the next character
    e = P["emb"][X].reshape(len(idx), K * E)
    h1 = np.tanh(e @ P["W1"] + P["b1"])
    h2 = np.tanh(h1 @ P["W2"] + P["b2"])
    z = h2 @ P["W3"] + P["b3"]
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    if not grad:
        return loss
    B = len(y)
    g = np.exp(logp); g[np.arange(B), y] -= 1; g /= B
    G = {"W3": h2.T @ g, "b3": g.sum(0)}
    g = (g @ P["W3"].T) * (1 - h2 ** 2)
    G["W2"], G["b2"] = h1.T @ g, g.sum(0)
    g = (g @ P["W2"].T) * (1 - h1 ** 2)
    G["W1"], G["b1"] = e.T @ g, g.sum(0)
    ge = (g @ P["W1"].T).reshape(B, K, E)
    G["emb"] = np.zeros_like(P["emb"]); np.add.at(G["emb"], X, ge)
    return loss, G

# Gradient check on a tiny model.
r = np.random.default_rng(0)
P = init(8, r); idx = r.integers(0, len(train) - K, 4)
_, G = loss_and_grads(P, idx, train)
num = []
for k, i in [("W1", (3, 2)), ("W2", (1, 5)), ("W3", (4, 0)), ("emb", tuple(train[idx[0]:idx[0] + 1]) + (2,))]:
    P[k][i] += 1e-5; lp = loss_and_grads(P, idx, train, False)
    P[k][i] -= 2e-5; lm = loss_and_grads(P, idx, train, False); P[k][i] += 1e-5
    num.append(abs((lp - lm) / 2e-5 - G[k][i]))
print(f"gradient check: max error {max(num):.1e}")

# 3. Train each width once, single pass over the data (no example is seen twice), with Adam at a
# constant learning rate, and record the held-out loss at log-spaced numbers of tokens D.
val_idx = np.random.default_rng(1).integers(0, len(val) - K, 20000)
widths = [16, 32, 64, 128, 256, 512]
B, steps = 128, 7000
checkpoints = np.unique(np.geomspace(25, steps, 14).astype(int))
results = {}                                       # (N, D) -> held-out loss in nats
t0 = time.time()
for H in widths:
    rng = np.random.default_rng(2)
    P = init(H, rng)
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    order = rng.permutation(len(train) - K)
    lr = 3e-3 if H <= 64 else 2e-3 if H <= 256 else 1e-3
    for t in range(1, steps + 1):
        _, G = loss_and_grads(P, order[(t - 1) * B:t * B], train)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
            P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
        if t in checkpoints:
            results[(n_params(H), t * B)] = loss_and_grads(P, val_idx, val, False)
    print(f"H = {H:3d}: N = {n_params(H):7,}, held-out loss after {steps * B:,} tokens:"
          f" {results[(n_params(H), steps * B)]:.3f} nats ({results[(n_params(H), steps * B)] / np.log(2):.3f} bits) per character")
print(f"(trained in {time.time() - t0:.0f} s)")

# 4. Fit the Chinchilla form L(N, D) = E + A / N^alpha + B / D^beta to the checkpoints after
# 100,000 tokens (earlier ones are dominated by the start-up of training). For fixed exponents
# the loss is linear in (E, A, B), so grid-search the exponents and solve least squares inside.
keys = [k for k in results if k[1] >= 100_000]
Ns, Ds = np.array([k[0] for k in keys], float), np.array([k[1] for k in keys], float)
Ls = np.array([results[k] for k in keys])
best = None
for a in np.arange(0.05, 1.5, 0.01):
    for b in np.arange(0.05, 1.5, 0.01):
        M = np.stack([np.ones_like(Ns), Ns ** -a, Ds ** -b], 1)
        coef, *_ = np.linalg.lstsq(M, Ls, rcond=None)
        if (coef < 0).any():
            continue
        err = np.mean((M @ coef - Ls) ** 2)
        if best is None or err < best[0]:
            best = (err, a, b, coef)
err, alpha, beta, (E0, A, Bc) = best
print(f"fit: L = {E0:.3f} + {A:.3g} / N^{alpha:.2f} + {Bc:.3g} / D^{beta:.2f}"
      f"  (rms error {np.sqrt(err):.4f} nats over {len(Ls)} checkpoints)")

# 5. Compute-optimal allocation. With C = 6ND, minimising the fitted L over N gives
# N_opt = G (C/6)^(beta/(alpha+beta)), G = (alpha A / (beta B))^(1/(alpha+beta)).
# Compare with the runs themselves: at each budget C, which width did best?
a_exp = beta / (alpha + beta)
Gc = (alpha * A / (beta * Bc)) ** (1 / (alpha + beta))
Dgrid = sorted({k[1] for k in results})
for C in [1e10, 3e10, 1e11, 3e11]:
    runs = [(np.interp(np.log(C / (6 * n_params(H))), np.log(Dgrid), [results[(n_params(H), d)] for d in Dgrid]), H)
            for H in widths if Dgrid[0] <= C / (6 * n_params(H)) <= Dgrid[-1]]   # runs that reached C
    best = min(runs)
    N_opt = Gc * (C / 6) ** a_exp
    print(f"C = {C:.0e} FLOPs: best of {len(runs)} runs H = {best[1]} (N = {n_params(best[1]):,}, loss {best[0]:.3f});"
          f" the fit's N_opt = {N_opt:,.0f}, D_opt = {C / (6 * N_opt):,.0f}")
print(f"according to the fit, N_opt grows as C^{a_exp:.2f} and D_opt as C^{1 - a_exp:.2f}")

# 1,115,394 characters, vocabulary 65; 1,003,854 for training
# gradient check: max error 2.9e-11
# H =  16: N =   5,392, held-out loss after 896,000 tokens: 2.169 nats (3.130 bits) per character
# H =  32: N =  11,296, held-out loss after 896,000 tokens: 2.075 nats (2.994 bits) per character
# H =  64: N =  24,640, held-out loss after 896,000 tokens: 1.982 nats (2.859 bits) per character
# H = 128: N =  57,472, held-out loss after 896,000 tokens: 1.881 nats (2.713 bits) per character
# H = 256: N = 147,712, held-out loss after 896,000 tokens: 1.851 nats (2.670 bits) per character
# H = 512: N = 426,496, held-out loss after 896,000 tokens: 1.816 nats (2.620 bits) per character
# (trained in 187 s)
# fit: L = 1.609 + 134 / N^0.70 + 63 / D^0.41  (rms error 0.0186 nats over 36 checkpoints)
# C = 1e+10 FLOPs: best of 6 runs H = 32 (N = 11,296, loss 2.258); the fit's N_opt = 8,123, D_opt = 205,182
# C = 3e+10 FLOPs: best of 5 runs H = 32 (N = 11,296, loss 2.113); the fit's N_opt = 12,188, D_opt = 410,228
# C = 1e+11 FLOPs: best of 4 runs H = 64 (N = 24,640, loss 2.001); the fit's N_opt = 19,014, D_opt = 876,534
# C = 3e+11 FLOPs: best of 3 runs H = 128 (N = 57,472, loss 1.885); the fit's N_opt = 28,531, D_opt = 1,752,492
# according to the fit, N_opt grows as C^0.37 and D_opt as C^0.63
