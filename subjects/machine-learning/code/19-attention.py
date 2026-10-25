import numpy as np

rng = np.random.default_rng(0)

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

# 1. Why divide by sqrt(d)? Dot products of random d-dimensional vectors have variance d.
for d in [4, 64, 1024]:
    q, k = rng.standard_normal((1000, d)), rng.standard_normal((1000, 16, d))
    s = np.einsum("nd,ntd->nt", q, k)
    for name, sc in [("unscaled", s), ("scaled  ", s / np.sqrt(d))]:
        a = softmax(sc)
        print(f"d = {d:4d} {name}: score std {sc.std():6.2f}   mean max weight {a.max(1).mean():.2f}"
              f"   mean entropy {-(a * np.log(a + 1e-30)).sum(1).mean():.2f} nats (uniform: {np.log(16):.2f})")

# 2. Associative recall: 8 (key, value) pairs, then a query key; answer with its value.
V, P = 16, 8
def make(n):
    keys = np.array([rng.choice(V, P, replace=False) for _ in range(n)])
    vals = rng.integers(0, V, (n, P))
    which = rng.integers(0, P, n)
    X = np.concatenate([np.eye(V)[keys], np.eye(V)[vals]], -1)        # (n, P, 2V): each pair as one vector
    return X, np.eye(V)[keys[np.arange(n), which]], vals[np.arange(n), which]

def attend(Wq, Wk, Wv, Wo, X, Q):
    """One attention head: query from Q, keys and values from the pairs, then a linear readout."""
    d = Wq.shape[0]
    q, k, v = Q @ Wq.T, X @ Wk.T, X @ Wv.T                            # (n,d), (n,P,d), (n,P,d)
    a = softmax(np.einsum("nd,npd->np", q, k) / np.sqrt(d))           # attention weights over the pairs
    o = np.einsum("np,npd->nd", a, v)
    return o @ Wo.T, (q, k, v, a, o)

def grads(W, X, Q, y):
    Wq, Wk, Wv, Wo = W; d = Wq.shape[0]
    F, (q, k, v, a, o) = attend(*W, X, Q)
    p = softmax(F); loss = -np.log(p[np.arange(len(y)), y]).mean()
    dF = p; dF[np.arange(len(y)), y] -= 1; dF /= len(y)
    dWo = dF.T @ o; do = dF @ Wo
    da = np.einsum("nd,npd->np", do, v); dv = np.einsum("np,nd->npd", a, do)
    ds = a * (da - (da * a).sum(1, keepdims=True)) / np.sqrt(d)      # backward through the softmax
    dq = np.einsum("np,npd->nd", ds, k); dk = np.einsum("np,nd->npd", ds, q)
    return loss, [dq.T @ Q, np.einsum("npd,npe->de", dk, X), np.einsum("npd,npe->de", dv, X), dWo]

d = 32
W = [rng.standard_normal((d, V)) / 4, rng.standard_normal((d, 2 * V)) / 6, rng.standard_normal((d, 2 * V)) / 6, rng.standard_normal((V, d)) / 6]
X, Q, y = make(4)                                                     # finite-difference check first
_, g = grads(W, X, Q, y)
num = np.zeros_like(W[1])
for i in np.ndindex(W[1].shape):
    W[1][i] += 1e-6; up = grads(W, X, Q, y)[0]; W[1][i] -= 2e-6; dn = grads(W, X, Q, y)[0]; W[1][i] += 1e-6
    num[i] = (up - dn) / 2e-6
print(f"gradient check (key projection): max relative error {np.abs(num - g[1]).max() / np.abs(num).max():.1e}")

m = [np.zeros_like(w) for w in W]; v2 = [np.zeros_like(w) for w in W]
for t in range(1, 3001):
    X, Q, y = make(128)
    loss, g = grads(W, X, Q, y)
    for i in range(4):
        m[i] = 0.9 * m[i] + 0.1 * g[i]; v2[i] = 0.999 * v2[i] + 0.001 * g[i] ** 2
        W[i] -= 3e-3 * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v2[i] / (1 - 0.999 ** t)) + 1e-8)
    if t in (1, 500, 1000, 3000):
        Xt, Qt, yt = make(2000); Ft, (_, _, _, at, _) = attend(*W, Xt, Qt)
        right = np.argmax(np.einsum("npv,nv->np", Xt[:, :, :V], Qt), 1)  # which pair holds the queried key
        print(f"step {t:4d}: loss {loss:.3f}   test acc {np.mean(Ft.argmax(1) == yt):.3f}"
              f"   mean attention on the right pair {at[np.arange(2000), right].mean():.2f}")

# 3. The same task without attention: average the pairs, then an MLP on [average, query]
H = 256
A1 = rng.standard_normal((H, 3 * V)) * np.sqrt(2 / (3 * V)); A2 = rng.standard_normal((V, H)) * np.sqrt(2 / H)
mm = [0, 0]; vv = [0, 0]
for t in range(1, 3001):
    X, Q, y = make(128); Z = np.c_[X.mean(1), Q]
    h = np.maximum(Z @ A1.T, 0); p = softmax(h @ A2.T)
    dF = p; dF[np.arange(128), y] -= 1; dF /= 128
    gs = [((dF @ A2) * (h > 0)).T @ Z, dF.T @ h]
    for i, Wm in enumerate([A1, A2]):
        mm[i] = 0.9 * mm[i] + 0.1 * gs[i]; vv[i] = 0.999 * vv[i] + 0.001 * gs[i] ** 2
        Wm -= 3e-3 * (mm[i] / (1 - 0.9 ** t)) / (np.sqrt(vv[i] / (1 - 0.999 ** t)) + 1e-8)
Xt, Qt, yt = make(2000)
acc = np.mean((np.maximum(np.c_[Xt.mean(1), Qt] @ A1.T, 0) @ A2.T).argmax(1) == yt)
print(f"average-then-MLP baseline ({A1.size + A2.size} parameters): test acc {acc:.3f}   (chance {1 / V:.3f})")

# d =    4 unscaled: score std   2.01   mean max weight 0.42   mean entropy 1.76 nats (uniform: 2.77)
# d =    4 scaled  : score std   1.01   mean max weight 0.23   mean entropy 2.38 nats (uniform: 2.77)
# d =   64 unscaled: score std   8.05   mean max weight 0.84   mean entropy 0.42 nats (uniform: 2.77)
# d =   64 scaled  : score std   1.01   mean max weight 0.25   mean entropy 2.35 nats (uniform: 2.77)
# d = 1024 unscaled: score std  32.03   mean max weight 0.96   mean entropy 0.10 nats (uniform: 2.77)
# d = 1024 scaled  : score std   1.00   mean max weight 0.24   mean entropy 2.36 nats (uniform: 2.77)
# gradient check (key projection): max relative error 1.8e-07
# step    1: loss 2.786   test acc 0.058   mean attention on the right pair 0.12
# step  500: loss 0.003   test acc 1.000   mean attention on the right pair 0.83
# step 1000: loss 0.001   test acc 1.000   mean attention on the right pair 0.86
# step 3000: loss 0.000   test acc 1.000   mean attention on the right pair 0.89
# average-then-MLP baseline (16384 parameters): test acc 0.258   (chance 0.062)
