import numpy as np

rng = np.random.default_rng(0)

# The noise schedule: T = 100 steps, beta rising linearly from 1e-4 to 0.2 (a short chain suits 2-D data).
T = 100
beta = np.linspace(1e-4, 0.2, T)
alpha = 1 - beta
abar = np.cumprod(alpha)                                   # abar[t] = alpha_1 ... alpha_t
print(f"signal left at the end: sqrt(abar_T) = {np.sqrt(abar[-1]):.2e}")

# 1. The forward process jumps straight to any step: q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I).
# Simulate the chain step by step from x_0 = 2 and compare with the formula.
x = np.full(200_000, 2.0)
for t in range(T):
    x = np.sqrt(alpha[t]) * x + np.sqrt(beta[t]) * rng.standard_normal(x.shape)
    if t + 1 in (10, 30):
        print(f"t = {t + 1}: simulated mean {x.mean():.4f}, var {x.var():.4f};"
              f"  formula mean {np.sqrt(abar[t]) * 2:.4f}, var {1 - abar[t]:.4f}")

# 2. Data: points on a spiral, with a little noise.
def data(n):
    s = rng.uniform(0.25, 1, n) ** 0.5 * 3 * np.pi
    return np.stack([s * np.cos(s), s * np.sin(s)], 1) / 4 + 0.03 * rng.standard_normal((n, 2))

# 3. The noise-prediction network eps_theta(x_t, t): an MLP on [x_t, sinusoidal embedding of t].
def embed(t, k=16):
    f = np.exp(-np.log(1000) * np.arange(k // 2) / (k // 2))
    a = t[:, None] * f[None] * 1.0
    return np.concatenate([np.sin(a), np.cos(a)], 1)

sizes = [2 + 16, 256, 256, 2]
W = [rng.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes, sizes[1:])]
b = [np.zeros(n) for n in sizes[1:]]
params = W + b
m1 = [np.zeros_like(p) for p in params]; m2 = [np.zeros_like(p) for p in params]

def net(x, t):
    hs = [np.concatenate([x, embed(t)], 1)]
    for i in range(3):
        z = hs[-1] @ W[i] + b[i]
        hs.append(np.maximum(z, 0) if i < 2 else z)
    return hs

def loss_and_grads(x0):
    """The DDPM loss: pick t, add noise eps, ask the network for eps back; squared error."""
    n = len(x0)
    t = rng.integers(0, T, n)
    eps = rng.standard_normal(x0.shape)
    xt = np.sqrt(abar[t])[:, None] * x0 + np.sqrt(1 - abar[t])[:, None] * eps
    hs = net(xt, t)
    g = 2 * (hs[-1] - eps) / n
    gW, gb = [None] * 3, [None] * 3
    for i in reversed(range(3)):
        gW[i], gb[i] = hs[i].T @ g, g.sum(0)
        if i:
            g = (g @ W[i].T) * (hs[i] > 0)
    return np.mean(np.sum((hs[-1] - eps) ** 2, 1)), gW + gb

for step in range(1, 15001):
    loss, grads = loss_and_grads(data(256))
    lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * step / 15000))
    for i, (p, g) in enumerate(zip(params, grads)):
        m1[i] = 0.9 * m1[i] + 0.1 * g; m2[i] = 0.999 * m2[i] + 0.001 * g * g
        p -= lr * (m1[i] / (1 - 0.9 ** step)) / (np.sqrt(m2[i] / (1 - 0.999 ** step)) + 1e-8)
    if step in (1, 1000, 5000, 15000):
        print(f"step {step:5d}: loss {loss:.3f}")

# 4. Ancestral sampling: start from pure noise and apply the learned reverse step T times.
def sample(n, keep=()):
    x = rng.standard_normal((n, 2)); kept = {T: x}
    for t in reversed(range(T)):
        eps = net(x, np.full(n, t))[-1]
        mean = (x - beta[t] / np.sqrt(1 - abar[t]) * eps) / np.sqrt(alpha[t])
        x = mean + (np.sqrt(beta[t]) * rng.standard_normal(x.shape) if t > 0 else 0)
        if t in keep:
            kept[t] = x
    return x, kept

S, kept = sample(2000, keep=(75, 50, 25, 10))
ref = data(20000)
def median_nn(A, B):
    """Median over the points of A of the distance to the nearest point of B."""
    return np.median([np.sqrt(np.min(np.sum((B - a) ** 2, 1))) for a in A])
D2, G2 = data(2000), rng.standard_normal((2000, 2))
print("median distance from 500 points to the nearest of 20,000 fresh data points (are they on the spiral?):")
print(f"  samples {median_nn(S[:500], ref):.4f}   fresh data {median_nn(D2[:500], ref):.4f}   Gaussian noise {median_nn(G2[:500], ref):.4f}")
print("median distance from 500 fresh data points to the nearest of 2,000 points (is all of the spiral covered?):")
print(f"  samples {median_nn(data(500), S):.4f}   fresh data {median_nn(data(500), D2):.4f}   Gaussian noise {median_nn(data(500), G2):.4f}")

# signal left at the end: sqrt(abar_T) = 4.63e-03
# t = 10: simulated mean 1.9105, var 0.0878;  formula mean 1.9096, var 0.0883
# t = 30: simulated mean 1.2749, var 0.5942;  formula mean 1.2756, var 0.5932
# step     1: loss 2.519
# step  1000: loss 0.422
# step  5000: loss 0.409
# step 15000: loss 0.472
# median distance from 500 points to the nearest of 20,000 fresh data points (are they on the spiral?):
#   samples 0.0046   fresh data 0.0032   Gaussian noise 0.6706
# median distance from 500 fresh data points to the nearest of 2,000 points (is all of the spiral covered?):
#   samples 0.0128   fresh data 0.0106   Gaussian noise 0.0579
