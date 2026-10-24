import numpy as np

rng = np.random.default_rng(0)

def spirals(n, noise=0.08):
    """Two interleaved spirals in the plane, labels 0 and 1."""
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi       # angle; sqrt spreads points evenly along the arm
    y = rng.integers(0, 2, n)
    r = t / (3 * np.pi)
    X = np.c_[r * np.cos(t + np.pi * y), r * np.sin(t + np.pi * y)] + noise * rng.standard_normal((n, 2))
    return X, y

def init(sizes, rng):
    """He-style initialisation: weights ~ N(0, 2 / fan_in), biases 0 (lesson 14 explains why)."""
    return [(rng.standard_normal((m, k)) * np.sqrt(2 / k), np.zeros(m)) for k, m in zip(sizes[:-1], sizes[1:])]

def forward(params, X):
    """ReLU hidden layers, one output logit. Returns the logit and the cache backprop needs."""
    a, cache = X, []
    for i, (W, b) in enumerate(params):
        z = a @ W.T + b
        cache.append((a, z))
        a = np.maximum(z, 0) if i < len(params) - 1 else z
    return a[:, 0], cache

def loss_and_grads(params, X, y):
    """Mean log-loss and its gradient with respect to every W and b (lesson 13 derives this)."""
    F, cache = forward(params, X)
    loss = np.mean(np.logaddexp(0, F) - y * F)
    delta = ((1 / (1 + np.exp(-F))) - y)[:, None] / len(y)     # dLoss/dz at the output
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        a, z = cache[i]
        grads[i] = (delta.T @ a, delta.sum(0))
        if i > 0:
            delta = (delta @ params[i][0]) * (cache[i - 1][1] > 0)   # back through W, then through ReLU
    return loss, grads

def train(sizes, X, y, steps=4000, lr=0.01, seed=0):
    """Full-batch Adam (lesson 07)."""
    params = init(sizes, np.random.default_rng(seed))
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]; v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    for t in range(1, steps + 1):
        _, grads = loss_and_grads(params, X, y)
        for i, ((W, b), (gW, gb)) in enumerate(zip(params, grads)):
            m[i] = (0.9 * m[i][0] + 0.1 * gW, 0.9 * m[i][1] + 0.1 * gb)
            v[i] = (0.999 * v[i][0] + 0.001 * gW**2, 0.999 * v[i][1] + 0.001 * gb**2)
            c1, c2 = 1 - 0.9**t, 1 - 0.999**t
            params[i] = (W - lr * m[i][0] / c1 / (np.sqrt(v[i][0] / c2) + 1e-8), b - lr * m[i][1] / c1 / (np.sqrt(v[i][1] / c2) + 1e-8))
    return params

Xtr, ytr = spirals(400); Xte, yte = spirals(4000)
acc = lambda p, X, y: np.mean((forward(p, X)[0] > 0) == y)

# gradient check: compare backprop with central differences on a small network
p = init([2, 5, 1], np.random.default_rng(1))
_, g = loss_and_grads(p, Xtr[:20], ytr[:20])
W = p[0][0]; num = np.zeros_like(W)
for idx in np.ndindex(W.shape):
    for s in (+1, -1):
        W[idx] += s * 1e-6; num[idx] += s * loss_and_grads(p, Xtr[:20], ytr[:20])[0] / 2e-6; W[idx] -= s * 1e-6
print(f"gradient check, first layer: max relative error {np.max(np.abs(num - g[0][0])) / np.max(np.abs(num)):.1e}")

for sizes in [[2, 1], [2, 2, 1], [2, 8, 1], [2, 64, 1], [2, 64, 64, 1]]:
    params = train(sizes, Xtr, ytr)
    n_par = sum(W.size + b.size for W, b in params)
    print(f"layers {str(sizes):16s} {n_par:5d} parameters   train acc {acc(params, Xtr, ytr):.3f}   test acc {acc(params, Xte, yte):.3f}")

# gradient check, first layer: max relative error 1.6e-09
# layers [2, 1]               3 parameters   train acc 0.693   test acc 0.646
# layers [2, 2, 1]            9 parameters   train acc 0.693   test acc 0.646
# layers [2, 8, 1]           33 parameters   train acc 0.858   test acc 0.810
# layers [2, 64, 1]         257 parameters   train acc 0.995   test acc 0.948
# layers [2, 64, 64, 1]    4417 parameters   train acc 1.000   test acc 0.948
