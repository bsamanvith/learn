"""Assignment A7: a variational autoencoder and a diffusion model, in NumPy.

Fill in every function marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

The networks are given: MLP has forward(x) and backward(g), as in A5, so that
you can concentrate on the probabilistic parts. Every function that returns a
gradient returns the gradient of the quantity it computes (not of its negative),
averaged over the batch the same way the quantity is.
"""
import gzip
import os
import urllib.request
import numpy as np


# ---------------------------------------------------------------- given: an MLP with a hand-written backward pass

class MLP:
    """Fully connected network with ReLU between layers and a linear output.
    forward(x) caches; backward(g) takes dL/d(output), stores self.grads (same order as
    self.params) and returns dL/d(input)."""

    def __init__(self, sizes, rng):
        self.params = []
        for a, b in zip(sizes, sizes[1:]):
            self.params += [rng.standard_normal((a, b)) * np.sqrt(2 / a), np.zeros(b)]
        self.grads = [np.zeros_like(p) for p in self.params]

    def forward(self, x):
        self.hs = [x]
        n = len(self.params) // 2
        for i in range(n):
            x = x @ self.params[2 * i] + self.params[2 * i + 1]
            if i < n - 1:
                x = np.maximum(x, 0)
            self.hs.append(x)
        return x

    def backward(self, g):
        n = len(self.params) // 2
        for i in reversed(range(n)):
            if i < n - 1:
                g = g * (self.hs[i + 1] > 0)
            self.grads[2 * i], self.grads[2 * i + 1] = self.hs[i].T @ g, g.sum(0)
            g = g @ self.params[2 * i].T
        return g


class Adam:
    def __init__(self, nets, lr=1e-3):
        self.nets, self.lr, self.t = nets, lr, 0
        self.m = [[np.zeros_like(p) for p in n.params] for n in nets]
        self.v = [[np.zeros_like(p) for p in n.params] for n in nets]

    def step(self, sign=1.0):
        """Descend the stored gradients (sign=-1 ascends them)."""
        self.t += 1
        for net, ms, vs in zip(self.nets, self.m, self.v):
            for p, g, m, v in zip(net.params, net.grads, ms, vs):
                g = sign * g
                m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
                p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


# ---------------------------------------------------------------- Part 1: the VAE's pieces (lesson 22)

def gaussian_kl(mu, logvar):
    """KL( N(mu, diag exp(logvar)) || N(0, I) ) for each row. Returns (kl of shape (N,),
    dkl/dmu, dkl/dlogvar), the gradients of each row's KL with respect to that row."""
    raise NotImplementedError  # TODO


def bernoulli_loglik(x, logits):
    """log p(x | logits) for independent Bernoulli pixels with means sigmoid(logits), summed
    over each row. Returns (loglik of shape (N,), dloglik/dlogits). x may be in [0, 1].
    Must be stable for logits of any size."""
    raise NotImplementedError  # TODO


def reparameterise(mu, logvar, eps):
    """z = mu + sigma * eps with sigma = exp(logvar / 2). Returns (z, dz/dmu, dz/dlogvar),
    the elementwise derivatives (arrays shaped like mu)."""
    raise NotImplementedError  # TODO


def vae_elbo(enc, dec, x, eps):
    """The batch-average ELBO with one sample per example, using the noise eps given (N, k).
    enc maps x to [mu, logvar] concatenated (2k outputs); dec maps z to logits.
    Returns (elbo, mean reconstruction term, mean KL) and leaves in enc.grads and dec.grads
    the gradients of the batch-average ELBO."""
    raise NotImplementedError  # TODO


def iwae_bound(enc, dec, x, K, rng):
    """The importance-weighted estimate log (1/K) sum_k p(x, z_k) / q(z_k | x), z_k ~ q(z | x),
    for each row of x (shape (N,)). Use a log-sum-exp so that it never overflows."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: diffusion (lessons 24-25)

def make_schedule(T, beta_1, beta_T):
    """Linear betas from beta_1 to beta_T. Returns (beta, alpha, abar), arrays of length T,
    where index t - 1 holds step t, and abar[t - 1] = alpha_1 * ... * alpha_t."""
    raise NotImplementedError  # TODO


def q_sample(x0, t, eps, abar):
    """Sample x_t given x_0, using the closed-form marginal. t is an integer array (N,) of
    steps in 1..T; eps is the standard normal noise to use."""
    raise NotImplementedError  # TODO


def q_posterior(x0, xt, t, beta, abar):
    """Mean and variance of q(x_{t-1} | x_t, x_0) for steps t >= 2 (an integer array).
    Returns (mean of shape like x0, variance of shape (N,))."""
    raise NotImplementedError  # TODO


def time_features(t, T, k=16):
    """Sinusoidal features of the step (given)."""
    f = np.exp(-np.log(1000) * np.arange(k // 2) / (k // 2))
    a = (t / T * 100.0)[:, None] * f
    return np.concatenate([np.sin(a), np.cos(a)], 1)


def predict_eps(net, xt, t, T):
    """The network's noise estimate: net applied to [x_t, time_features(t)] (given)."""
    return net.forward(np.concatenate([xt, time_features(t, T)], 1))


def ddpm_loss(net, x0, t, eps, abar, T):
    """The simplified DDPM loss: mean over the batch of ||eps - eps_theta(x_t, t)||^2, where
    x_t = q_sample(x0, t, eps). Returns the loss and leaves its gradient in net.grads."""
    raise NotImplementedError  # TODO


def p_sample_step(net, xt, t, z, beta, alpha, abar, T):
    """One ancestral step from x_t to x_{t-1} (t an int, the same for the whole batch), with
    variance beta_t, using the noise z; at t == 1 return the mean, with no noise."""
    raise NotImplementedError  # TODO


def ddim_step(net, xt, t, s, abar, T):
    """One deterministic DDIM step from step t to an earlier step s (ints, s < t; s == 0
    means the clean data, with abar_0 = 1)."""
    raise NotImplementedError  # TODO


def sample(net, n, d, schedule, rng, ddim_steps=None):
    """Generate n samples of dimension d, starting from N(0, I) at step T: ancestral sampling
    through every step if ddim_steps is None, otherwise DDIM through ddim_steps steps evenly
    spaced from T down to 0 (np.linspace(T, 0, ddim_steps + 1), rounded to integers)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- given: data

def load_mnist(cache=os.path.expanduser("~/.cache/learn-mnist")):
    """Download MNIST once (about 11 MB) and return (Xtr, ytr, Xte, yte): images flattened
    to 784 floats in [0, 1], labels as ints."""
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    os.makedirs(cache, exist_ok=True)
    out = []
    for name in names:
        path = os.path.join(cache, name + ".gz")
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + name + ".gz", path)
        with gzip.open(path) as f:
            raw = np.frombuffer(f.read(), np.uint8)
        out.append(raw[16:].reshape(-1, 784) / 255.0 if "images" in name else raw[8:].astype(int))
    return out


def two_moons(n, rng, noise=0.05):
    """Points on two interleaved half circles, scaled to roughly unit variance."""
    s = rng.uniform(0, np.pi, n); top = rng.random(n) < 0.5
    x = np.where(top, np.cos(s), 1 - np.cos(s)); y = np.where(top, np.sin(s), 0.5 - np.sin(s))
    return (np.stack([x - 0.5, y - 0.25], 1) / 0.7) + noise * rng.standard_normal((n, 2))
