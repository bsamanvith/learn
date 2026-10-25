"""Problem book assignment — a diffusion model draws a spiral.

DDPM on 500 points of a 2D spiral, in NumPy. First with the *exact* optimal denoiser for
the training set (a softmax-weighted average of the training points, by Tweedie's formula):
it reproduces training points, nothing new. Then with a small MLP trained to predict the
noise: it draws new points along the spiral. Finally two ways to break sampling.

Run: .venv/bin/python subjects/machine-learning/sims/diffusion_spiral.py [--plot]
"""

import sys
import numpy as np

rng = np.random.default_rng(0)


def spiral(n, rng, noise=0.02):
    s = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi + np.pi / 2       # angle, denser further out
    pts = np.c_[s * np.cos(s), s * np.sin(s)] / (3 * np.pi)
    return pts + noise * rng.normal(size=pts.shape)


data = spiral(500, rng)
curve = spiral(20000, np.random.default_rng(99), noise=0.0)          # the noiseless curve, for scoring
T = 200
beta = np.linspace(1e-4, 0.08, T)
alpha = 1 - beta
abar = np.cumprod(alpha)
print(f"Step 0 · {len(data)} training points on a spiral; T = {T} steps, β from {beta[0]:.0e} to {beta[-1]}")
print(f"         ᾱ_T = {abar[-1]:.2e}: x_T keeps √ᾱ_T = {np.sqrt(abar[-1]):.3f} of the signal")


def dist_to(points, ref):
    d2 = ((points[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2.min(1))


def score(samples, name):
    on_curve = np.median(dist_to(samples, curve))
    nearest = np.median(dist_to(samples, data))
    print(f"         {name:<34} median distance to the spiral {on_curve:.4f}   to the nearest training point {nearest:.4f}")
    return samples


# Step 1 · forward process check.
t = 99
x0 = data[rng.integers(0, len(data), 200000)]
xt = np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * rng.normal(size=x0.shape)
print(f"\nStep 1 · at t = {t + 1}: E[x_t] = {xt.mean(0).round(3)}, Var(x_t) per coordinate = {xt.var(0).round(3)}"
      f"  (theory: ᾱ Var(x0) + 1 − ᾱ = {(abar[t] * data.var(0) + 1 - abar[t]).round(3)})")


# The reverse step, given any predictor of x0.
def sample(predict_x0, n, seed, noise=True, steps=None):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 2))
    ts = range(T - 1, -1, -1)
    for t in ts:
        x0_hat = predict_x0(x, t)
        if t == 0:
            x = x0_hat
            break
        ab, ab_prev = abar[t], abar[t - 1]
        mean = (np.sqrt(alpha[t]) * (1 - ab_prev) * x + np.sqrt(ab_prev) * beta[t] * x0_hat) / (1 - ab)
        var = beta[t] * (1 - ab_prev) / (1 - ab)
        x = mean + (np.sqrt(var) * r.normal(size=x.shape) if noise else 0)
    return x


# Step 2 · the exact denoiser for the empirical distribution: E[x0 | x_t] by Bayes over the 500 points.
def ideal_x0(x, t):
    d2 = ((x[:, None, :] - np.sqrt(abar[t]) * data[None]) ** 2).sum(-1)
    logw = -d2 / (2 * (1 - abar[t]))
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return w @ data


print("\nStep 2 · sampling with the exact optimal denoiser of the training set")
ideal = score(sample(ideal_x0, 1000, seed=1), "ideal denoiser")


# Step 3 · a small MLP that predicts the noise ε from (x_t, t).
def temb(t, dim=16):
    f = np.exp(np.linspace(0, np.log(1000), dim // 2))
    a = np.outer(t / T, f)
    return np.c_[np.sin(a), np.cos(a)]


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def __call__(self, X, keep=False):
        hs, zs = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            zs.append(z)
            hs.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        if keep:
            self.cache = hs, zs
        return hs[-1]

    def step(self, dout, lr):
        hs, zs = self.cache
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i], gb[i] = hs[i].T @ d, d.sum(0)
            if i:
                d = (d @ self.W[i].T) * (zs[i - 1] > 0)
        self.t += 1
        for k, g in enumerate(gW + gb):
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g**2
            self.params[k] -= lr * (self.m[k] / (1 - 0.9**self.t)) / (np.sqrt(self.v[k] / (1 - 0.999**self.t)) + 1e-8)


net = MLP([2 + 16, 128, 128, 128, 2], np.random.default_rng(2))
train_rng = np.random.default_rng(3)
steps, batch = 15000, 256
print(f"\nStep 3 · training an MLP 18 → 128 → 128 → 128 → 2 to predict the noise, {steps} Adam steps of batch {batch}")
for s in range(1, steps + 1):
    x0 = data[train_rng.integers(0, len(data), batch)]
    t = train_rng.integers(0, T, batch)
    eps = train_rng.normal(size=x0.shape)
    xt = np.sqrt(abar[t])[:, None] * x0 + np.sqrt(1 - abar[t])[:, None] * eps
    pred = net(np.c_[xt, temb(t)], keep=True)
    loss = np.mean((pred - eps) ** 2)
    lr = 1e-3 if s < 10000 else 3e-4
    net.step(2 * (pred - eps) / pred.size, lr)
    if s in (1, 1000, 5000, steps):
        print(f"         step {s:>6}: noise-prediction loss {loss:.4f}")


def mlp_x0(x, t):
    eps = net(np.c_[x, temb(np.full(len(x), t))])
    return (x - np.sqrt(1 - abar[t]) * eps) / np.sqrt(abar[t])


learned = score(sample(mlp_x0, 1000, seed=1), "trained MLP")
fresh = score(spiral(1000, np.random.default_rng(7)), "(fresh real data, for comparison)")

# Step 4 · break it.
print("\nStep 4 · breaking the sampler")
nonoise = score(sample(mlp_x0, 1000, seed=1, noise=False), "no noise in the reverse steps")
print(f"         spread of samples: std {nonoise.std(0).round(3)} against the data's {data.std(0).round(3)}")

def one_shot(n, seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 2))
    return mlp_x0(x, T - 1)


oneshot = score(one_shot(1000, 1), "jump from x_T straight to x̂0")
print(f"         spread of one-shot samples: std {oneshot.std(0).round(3)}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
    for a, pts, title in zip(ax, [data, ideal, learned, nonoise, oneshot],
                             ["training data (500)", "exact denoiser: copies", "trained MLP: new points",
                              "no noise while sampling", "one jump from noise"]):
        a.scatter(*pts.T, s=3)
        a.set(title=title, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal", xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
