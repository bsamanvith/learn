import numpy as np

rng = np.random.default_rng(0)

def spirals(n, noise=0.08):                       # lesson 12's data
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    y = rng.integers(0, 2, n); r = t / (3 * np.pi)
    return np.c_[r * np.cos(t + np.pi * y), r * np.sin(t + np.pi * y)] + noise * rng.standard_normal((n, 2)), y

def init(sizes, scale, seed):
    """scale(fan_in) gives the standard deviation of the weights."""
    r = np.random.default_rng(seed)
    return [(r.standard_normal((m, k)) * scale(k), np.zeros(m)) for k, m in zip(sizes[:-1], sizes[1:])]

def forward(params, X, norm=False):
    a, cache = X, []
    for i, (W, b) in enumerate(params):
        z = a @ W.T + b
        if norm and i < len(params) - 1:          # layer norm without a learned gain: per-example standardisation
            mu, sd = z.mean(1, keepdims=True), np.sqrt(z.var(1, keepdims=True) + 1e-5)
            zn = (z - mu) / sd
        else:
            zn, sd = z, None
        cache.append((a, zn, sd))
        a = np.maximum(zn, 0) if i < len(params) - 1 else zn
    return a[:, 0], cache

def grads(params, X, y, norm=False):
    F, cache = forward(params, X, norm)
    loss = np.mean(np.logaddexp(0, F) - y * F)
    d = ((1 / (1 + np.exp(-F))) - y)[:, None] / len(y)
    out = [None] * len(params)
    for i in reversed(range(len(params))):
        a, zn, sd = cache[i]
        if sd is not None:                        # backward through the standardisation
            d = (d - d.mean(1, keepdims=True) - zn * (d * zn).mean(1, keepdims=True)) / sd
        out[i] = (d.T @ a, d.sum(0))
        d = (d @ params[i][0]) * (cache[i - 1][1] > 0) if i > 0 else None
    return loss, out

schemes = {"N(0, 0.01²)": lambda k: 0.01, "Xavier 1/fan_in": lambda k: np.sqrt(1 / k), "He 2/fan_in": lambda k: np.sqrt(2 / k)}

# 1. Signal propagation through 50 ReLU layers of width 256 at initialisation
X = rng.standard_normal((1000, 256))
for name, scale in schemes.items():
    params = init([256] * 51 + [1], scale, seed=1)
    _, cache = forward(params, X)
    stds = [np.std(c[1]) for c in cache[:-1]]
    print(f"{name:16s} pre-activation std at layers 1, 10, 50: {stds[0]:.2e} {stds[9]:.2e} {stds[49]:.2e}")

# 2. Train a 20-hidden-layer, width-64 network on the spirals with SGD + momentum
Xtr, ytr = spirals(400); Xte, yte = spirals(4000)
def train(scale, norm=False, steps=1000, lr=0.003, seed=0):
    params = init([2] + [64] * 20 + [1], scale, seed)
    vel = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    hist = []
    for t in range(steps):
        loss, g = grads(params, Xtr, ytr, norm)
        if t == 0:
            g1 = np.linalg.norm(g[0][0])          # size of the first layer's gradient at initialisation
        hist.append(loss)
        for i, ((W, b), (gW, gb)) in enumerate(zip(params, g)):
            vel[i] = (0.9 * vel[i][0] - lr * gW, 0.9 * vel[i][1] - lr * gb)
            params[i] = (W + vel[i][0], b + vel[i][1])
    return hist, np.mean((forward(params, Xte, norm)[0] > 0) == yte), g1

for name, scale, norm in [("N(0, 0.01²)", schemes["N(0, 0.01²)"], False), ("Xavier", schemes["Xavier 1/fan_in"], False),
                          ("He", schemes["He 2/fan_in"], False), ("Xavier + norm", schemes["Xavier 1/fan_in"], True)]:
    hist, acc, g1 = train(scale, norm)
    print(f"{name:14s} |grad W1| at init {g1:.1e}   train loss at 0/250/1000 steps:"
          f" {hist[0]:.3f} {hist[249]:.3f} {hist[-1]:.3f}   test acc {acc:.3f}")

# N(0, 0.01²)      pre-activation std at layers 1, 10, 50: 1.59e-01 4.83e-10 1.86e-47
# Xavier 1/fan_in  pre-activation std at layers 1, 10, 50: 9.96e-01 4.39e-02 1.16e-07
# He 2/fan_in      pre-activation std at layers 1, 10, 50: 1.41e+00 1.41e+00 3.88e+00
# N(0, 0.01²)    |grad W1| at init 2.4e-27   train loss at 0/250/1000 steps: 0.693 0.690 0.690   test acc 0.496
# Xavier         |grad W1| at init 2.0e-05   train loss at 0/250/1000 steps: 0.693 0.690 0.689   test acc 0.496
# He             |grad W1| at init 2.6e-02   train loss at 0/250/1000 steps: 0.727 0.147 0.000   test acc 0.947
# Xavier + norm  |grad W1| at init 2.0e-02   train loss at 0/250/1000 steps: 0.759 0.434 0.002   test acc 0.955
