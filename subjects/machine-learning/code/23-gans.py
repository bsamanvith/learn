import copy
import numpy as np

rng = np.random.default_rng(0)

# 1. The optimal discriminator and the Jensen-Shannon divergence, in 1-D on a grid.
# p_data = N(0, 1), p_g = N(1, 1.5^2). D*(x) = p_data / (p_data + p_g), and V(G, D*) = -log 4 + 2 JSD.
x = np.linspace(-12, 12, 200_001); dx = x[1] - x[0]
pd = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
pg = np.exp(-(x - 1) ** 2 / (2 * 1.5 ** 2)) / (1.5 * np.sqrt(2 * np.pi))
Dstar = pd / (pd + pg)
V = np.sum(pd * np.log(Dstar) + pg * np.log(1 - Dstar)) * dx
m = (pd + pg) / 2
jsd = 0.5 * np.sum(pd * np.log(pd / m)) * dx + 0.5 * np.sum(pg * np.log(pg / m)) * dx
print(f"V(G, D*) = {V:.4f};  -log 4 + 2 JSD = {-np.log(4) + 2 * jsd:.4f};  JSD = {jsd:.4f} nats (max log 2 = 0.6931)")
for t in [0.0, 0.5, 1.0]:                      # any other discriminator does worse
    D = np.clip(Dstar + t * 0.2 * np.sin(x), 1e-9, 1 - 1e-9)
    print(f"  perturbed discriminator ({t:.1f}): V = {np.sum(pd * np.log(D) + pg * np.log(1 - D)) * dx:.4f}")

# 2. A small MLP with a hand-written backward pass, used for both networks.
class MLP:
    def __init__(self, sizes):
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.t = 0
    def forward(self, x):
        self.hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.where(x > 0, x, 0.2 * x)                        # leaky ReLU
            self.hs.append(x)
        return x
    def backward(self, g):
        """Given dLoss/d(output), store parameter gradients and return dLoss/d(input)."""
        self.gW, self.gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * np.where(self.hs[i + 1] > 0, 1, 0.2)
            self.gW[i], self.gb[i] = self.hs[i].T @ g, g.sum(0)
            g = g @ self.W[i].T
        return g
    def adam(self, lr, b1=0.5, b2=0.999):
        self.t += 1
        for p, g, m, v in zip(self.W + self.b, self.gW + self.gb, self.m, self.v):
            m *= b1; m += (1 - b1) * g; v *= b2; v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + 1e-8)

sig = lambda a: 1 / (1 + np.exp(-a))
centres = 2 * np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)
def data(n):                                                      # 8 Gaussians on a circle of radius 2
    return centres[rng.integers(0, 8, n)] + 0.05 * rng.standard_normal((n, 2))

def coverage(S):
    """How many of the 8 modes get at least 1% of the samples, and what fraction is near any mode."""
    d = np.linalg.norm(S[:, None] - centres[None], axis=2); near = d.min(1) < 0.15
    counts = np.bincount(d.argmin(1)[near], minlength=8)
    return int(np.sum(counts > 0.01 * len(S))), near.mean()

def d_step(G, D, lr=3e-4):
    """One discriminator step: logistic regression, real (label 1) against generated (label 0)."""
    real, fake = data(128), G.forward(rng.standard_normal((128, 2)))
    a = D.forward(np.vstack([real, fake]))[:, 0]
    y = np.r_[np.ones(128), np.zeros(128)]
    D.backward(((sig(a) - y) / 256)[:, None]); D.adam(lr)

def g_step(G, D, loss, lr=3e-4):
    """One generator step; returns the mean size of the gradient that reaches the samples."""
    fake = G.forward(rng.standard_normal((128, 2)))
    a = D.forward(fake)[:, 0]
    if loss == "minimax":                                         # descend log(1 - D(G(z)))
        ga = -sig(a) / 128                                        # d/da log(1 - sigmoid(a))
    else:                                                         # "non-saturating": descend -log D(G(z))
        ga = -(1 - sig(a)) / 128                                  # d/da -log sigmoid(a)
    gx = D.backward(ga[:, None])
    G.backward(gx); G.adam(lr)
    return np.linalg.norm(gx, axis=1).mean() * 128

# Part 2: a GAN with the non-saturating loss, snapshots of its samples along the way
G, D = MLP([2, 128, 128, 2]), MLP([2, 128, 128, 1])
snapshots = {}
for t in range(0, 8001):
    if t in (0, 500, 2000, 8000):
        S = G.forward(rng.standard_normal((2000, 2))); snapshots[t] = S
        k, f = coverage(S)
        print(f"step {t:4d}: {k}/8 modes covered, {f:.2f} of samples within 0.15 of a mode, mean D(G(z)) {np.mean(sig(D.forward(S)[:, 0])):.2f}")
    if t < 8000:
        d_step(G, D); g_step(G, D, "non-saturating")

# Part 3: saturation. Give the discriminator a head start of 1,000 steps against the initial generator,
# then train each version for 3,000 steps from that same state.
G0, D0 = MLP([2, 128, 128, 2]), MLP([2, 128, 128, 1])
for _ in range(1000):
    d_step(G0, D0, lr=1e-3)
print(f"after the head start: mean D(G(z)) = {np.mean(sig(D0.forward(G0.forward(rng.standard_normal((2000, 2))))[:, 0])):.4f}")
for loss in ["minimax", "non-saturating"]:
    G, D = copy.deepcopy(G0), copy.deepcopy(D0)
    sizes = []
    for t in range(3000):
        d_step(G, D); sizes.append(g_step(G, D, loss))
    k, f = coverage(G.forward(rng.standard_normal((2000, 2))))
    print(f"{loss:15s}: gradient reaching the samples, first 100 steps {np.mean(sizes[:100]):.4f};"
          f" after 3,000 steps {k}/8 modes, {f:.2f} near a mode")

# V(G, D*) = -1.1877;  -log 4 + 2 JSD = -1.1877;  JSD = 0.0993 nats (max log 2 = 0.6931)
#   perturbed discriminator (0.0): V = -1.1877
#   perturbed discriminator (0.5): V = -1.2147
#   perturbed discriminator (1.0): V = -1.2749
# step    0: 0/8 modes covered, 0.01 of samples within 0.15 of a mode, mean D(G(z)) 0.42
# step  500: 1/8 modes covered, 0.06 of samples within 0.15 of a mode, mean D(G(z)) 0.46
# step 2000: 6/8 modes covered, 0.11 of samples within 0.15 of a mode, mean D(G(z)) 0.39
# step 8000: 8/8 modes covered, 0.57 of samples within 0.15 of a mode, mean D(G(z)) 0.39
# after the head start: mean D(G(z)) = 0.0699
# minimax        : gradient reaching the samples, first 100 steps 0.7862; after 3,000 steps 3/8 modes, 0.73 near a mode
# non-saturating : gradient reaching the samples, first 100 steps 5.0442; after 3,000 steps 3/8 modes, 0.83 near a mode
