import numpy as np

rng = np.random.default_rng(0)

# 1. Langevin dynamics with the true score of a 1-D mixture: 0.8 N(-2, 0.5^2) + 0.2 N(2, 0.5^2).
w, mu, s = np.array([0.8, 0.2]), np.array([-2.0, 2.0]), 0.5
def score(x, extra=0.0):
    """d/dx log p(x), for the mixture convolved with N(0, extra) (extra = 0 gives the mixture itself)."""
    v = s ** 2 + extra
    comp = w * np.exp(-(x[:, None] - mu) ** 2 / (2 * v))
    return np.sum(comp * (mu - x[:, None]) / v, 1) / comp.sum(1)

def langevin(x, eps, steps, extra=0.0):
    for _ in range(steps):
        x = x + eps * score(x, extra) + np.sqrt(2 * eps) * rng.standard_normal(x.shape)
    return x

x = langevin(rng.uniform(-6, 6, 5000), 0.01, 2000)
print(f"Langevin from uniform starts:  fraction near +2 = {np.mean(x > 0):.3f}   (true 0.2)")
x = langevin(np.full(5000, -2.0), 0.01, 2000)
print(f"Langevin started in the left mode: fraction near +2 = {np.mean(x > 0):.3f}   (it can't cross)")
x = rng.standard_normal(5000) * np.sqrt(s ** 2 + 4 + 16)
for extra in [16, 8, 4, 2, 1, 0.5, 0.25, 0.1, 0.03, 0]:     # annealed: from heavy noise down to none
    x = langevin(x, 0.02 * (s ** 2 + extra), 500, extra)
print(f"annealed Langevin:            fraction near +2 = {np.mean(x > 0):.3f}")

# 2. Data with labels: three overlapping Gaussian blobs in the plane.
centres = 1.5 * np.stack([np.cos(2 * np.pi * np.arange(3) / 3), np.sin(2 * np.pi * np.arange(3) / 3)], 1)
SD = 0.8
def data(n):
    y = rng.integers(0, 3, n)
    return centres[y] + SD * rng.standard_normal((n, 2)), y
def posterior(X):
    """Bayes-optimal class probabilities p(y | x) under the true data distribution."""
    l = -np.sum((X[:, None] - centres[None]) ** 2, 2) / (2 * SD ** 2)
    e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

T = 100
beta = np.linspace(1e-4, 0.2, T); alpha = 1 - beta; abar = np.cumprod(alpha)

def embed(t, k=16):
    f = np.exp(-np.log(1000) * np.arange(k // 2) / (k // 2))
    return np.concatenate([np.sin(t[:, None] * f), np.cos(t[:, None] * f)], 1)

# 3. A conditional noise predictor eps(x_t, t, y), where y = 3 means "no label".
sizes = [2 + 16 + 4, 256, 256, 2]
W = [rng.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes, sizes[1:])]
b = [np.zeros(n) for n in sizes[1:]]
params = W + b; m1 = [np.zeros_like(p) for p in params]; m2 = [np.zeros_like(p) for p in params]

def net(x, t, y):
    hs = [np.concatenate([x, embed(t), np.eye(4)[y]], 1)]
    for i in range(3):
        z = hs[-1] @ W[i] + b[i]
        hs.append(np.maximum(z, 0) if i < 2 else z)
    return hs

for step in range(1, 15001):
    x0, y = data(256)
    y = np.where(rng.random(256) < 0.2, 3, y)                   # drop the label 20% of the time
    t = rng.integers(0, T, 256); eps = rng.standard_normal(x0.shape)
    xt = np.sqrt(abar[t])[:, None] * x0 + np.sqrt(1 - abar[t])[:, None] * eps
    hs = net(xt, t, y)
    g = 2 * (hs[-1] - eps) / 256
    grads = [None] * 6
    for i in reversed(range(3)):
        grads[i], grads[3 + i] = hs[i].T @ g, g.sum(0)
        if i:
            g = (g @ W[i].T) * (hs[i] > 0)
    lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * step / 15000))
    for i, (p, gr) in enumerate(zip(params, grads)):
        m1[i] = 0.9 * m1[i] + 0.1 * gr; m2[i] = 0.999 * m2[i] + 0.001 * gr * gr
        p -= lr * (m1[i] / (1 - 0.9 ** step)) / (np.sqrt(m2[i] / (1 - 0.999 ** step)) + 1e-8)

def eps_hat(x, t, y, guidance):
    """Classifier-free guidance: (1 + w) eps(x, y) - w eps(x, no label)."""
    n = len(x); tt = np.full(n, t)
    e_c = net(x, tt, np.full(n, y))[-1]
    if guidance == 0:
        return e_c
    return (1 + guidance) * e_c - guidance * net(x, tt, np.full(n, 3))[-1]

def ancestral(n, y, guidance=0.0):
    x = rng.standard_normal((n, 2))
    for t in reversed(range(T)):
        e = eps_hat(x, t, y, guidance)
        x = (x - beta[t] / np.sqrt(1 - abar[t]) * e) / np.sqrt(alpha[t]) + (np.sqrt(beta[t]) * rng.standard_normal(x.shape) if t else 0)
    return x

def ddim(n, y, steps):
    """Deterministic DDIM sampling on an evenly spaced subset of the steps."""
    ts = np.linspace(T - 1, 0, steps).round().astype(int)
    x = rng.standard_normal((n, 2))
    for i, t in enumerate(ts):
        e = eps_hat(x, t, y, 0.0)
        x0 = (x - np.sqrt(1 - abar[t]) * e) / np.sqrt(abar[t])
        if i + 1 < len(ts):
            tp = ts[i + 1]; x = np.sqrt(abar[tp]) * x0 + np.sqrt(1 - abar[tp]) * e
        else:
            x = x0
    return x

def report(X, y):
    """How typical of class y the samples are, where their centre is, and how spread out."""
    return (f"mean p(y = {y} | x) {posterior(X)[:, y].mean():.3f},  mean ({X[:, 0].mean():.2f}, {X[:, 1].mean():.2f}),"
            f"  spread {np.sqrt(np.mean(np.sum((X - X.mean(0)) ** 2, 1) / 2)):.3f}")

print(f"real data of class 0 (centre (1.50, 0.00), spread {SD}):")
print(f"                                  {report(centres[0] + SD * rng.standard_normal((2000, 2)), 0)}")
for g in [0.0, 1.0, 3.0]:
    print(f"ancestral, 100 steps, guidance {g:.0f}: {report(ancestral(2000, 0, g), 0)}")
for k in [100, 20, 5]:
    print(f"DDIM, {k:3d} steps, no guidance:     {report(ddim(2000, 0, k), 0)}")

# Langevin from uniform starts:  fraction near +2 = 0.475   (true 0.2)
# Langevin started in the left mode: fraction near +2 = 0.010   (it can't cross)
# annealed Langevin:            fraction near +2 = 0.207
# real data of class 0 (centre (1.50, 0.00), spread 0.8):
#                                   mean p(y = 0 | x) 0.863,  mean (1.51, 0.03),  spread 0.810
# ancestral, 100 steps, guidance 0: mean p(y = 0 | x) 0.867,  mean (1.51, 0.01),  spread 0.797
# ancestral, 100 steps, guidance 1: mean p(y = 0 | x) 0.971,  mean (1.92, -0.01),  spread 0.685
# ancestral, 100 steps, guidance 3: mean p(y = 0 | x) 0.992,  mean (2.20, -0.02),  spread 0.664
# DDIM, 100 steps, no guidance:     mean p(y = 0 | x) 0.874,  mean (1.50, 0.02),  spread 0.778
# DDIM,  20 steps, no guidance:     mean p(y = 0 | x) 0.897,  mean (1.52, -0.05),  spread 0.708
# DDIM,   5 steps, no guidance:     mean p(y = 0 | x) 0.953,  mean (1.47, -0.01),  spread 0.470
