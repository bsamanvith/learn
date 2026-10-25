"""Assignment A2: an optimiser zoo, in NumPy.

Fill in every function and method marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

Given: MNIST, a two-layer MLP with its backward pass, and two test functions. Yours: the
softmax cross-entropy loss, seven optimisers, three learning-rate schedules, and the metrics
used to compare the trained models (accuracy, the confusion matrix and calibration error).

Every optimiser works on a list of NumPy arrays `params`, updated in place, and is called as
opt.step(grads) with a list of gradient arrays of the same shapes. The learning rate is the
attribute opt.lr, which a schedule may change between steps.
"""
import gzip
import os
import urllib.request
import numpy as np


# ---------------------------------------------------------------- given: data, model, test functions

def load_mnist(cache=os.path.expanduser("~/.cache/learn-mnist")):
    """Download MNIST once (about 11 MB) and return (Xtr, ytr, Xte, yte): images flattened
    to 784 floats in [0, 1], labels as ints. 60,000 training and 10,000 test images."""
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


class MLP:
    """784 -> hidden (ReLU) -> 10 logits. params = [W1, b1, W2, b2]; forward caches what
    backward needs; backward(dlogits) returns the gradients in the same order."""
    def __init__(self, rng, hidden=128, n_in=784, n_out=10):
        self.params = [rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden)), np.zeros(hidden),
                       rng.normal(0, np.sqrt(1 / hidden), (hidden, n_out)), np.zeros(n_out)]

    def forward(self, X):
        W1, b1, W2, b2 = self.params
        self.X, self.h = X, np.maximum(X @ W1 + b1, 0)
        return self.h @ W2 + b2

    def backward(self, dlogits):
        W1, b1, W2, b2 = self.params
        dh = dlogits @ W2.T * (self.h > 0)
        return [self.X.T @ dh, dh.sum(0), self.h.T @ dlogits, dlogits.sum(0)]


def quadratic(kappa):
    """f(x) = 0.5 (x_1^2 + kappa x_2^2), a valley with condition number kappa. Returns (f, grad)."""
    f = lambda x: 0.5 * (x[0] ** 2 + kappa * x[1] ** 2)
    g = lambda x: np.array([x[0], kappa * x[1]])
    return f, g


def rosenbrock(x):
    """f = (1 - x)^2 + 100 (y - x^2)^2, minimum 0 at (1, 1). Returns (f, grad)."""
    f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    g = np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])
    return f, g


# ---------------------------------------------------------------- Part 1: the loss

def softmax_cross_entropy(logits, y):
    """Mean cross-entropy of integer labels y (B,) under softmax(logits) (B, K), computed stably
    (subtract the row maximum). Returns (loss, dloss/dlogits)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: optimisers

class SGD:
    """p <- p - lr * g."""
    def __init__(self, params, lr):
        self.params, self.lr = params, lr

    def step(self, grads):
        raise NotImplementedError  # TODO


class Momentum:
    """Heavy ball: v <- beta v + g; p <- p - lr v. With nesterov=True, step along the look-ahead
    instead: p <- p - lr (g + beta v), using the updated v."""
    def __init__(self, params, lr, beta=0.9, nesterov=False):
        self.params, self.lr, self.beta, self.nesterov = params, lr, beta, nesterov
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        raise NotImplementedError  # TODO


class AdaGrad:
    """s <- s + g^2; p <- p - lr g / (sqrt(s) + eps)."""
    def __init__(self, params, lr, eps=1e-8):
        self.params, self.lr, self.eps = params, lr, eps
        self.s = [np.zeros_like(p) for p in params]

    def step(self, grads):
        raise NotImplementedError  # TODO


class RMSProp:
    """s <- rho s + (1 - rho) g^2; p <- p - lr g / (sqrt(s) + eps)."""
    def __init__(self, params, lr, rho=0.9, eps=1e-8):
        self.params, self.lr, self.rho, self.eps = params, lr, rho, eps
        self.s = [np.zeros_like(p) for p in params]

    def step(self, grads):
        raise NotImplementedError  # TODO


class Adam:
    """t <- t + 1; m <- b1 m + (1 - b1) g; v <- b2 v + (1 - b2) g^2;
    p <- p - lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps).
    weight_decay > 0 adds weight_decay * p to the gradient first (L2 regularisation, coupled)."""
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, weight_decay
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0

    def step(self, grads):
        raise NotImplementedError  # TODO


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov and Hutter): shrink p <- p - lr * weight_decay * p,
    then take the Adam step with the plain gradient (no weight_decay * p added to it)."""
    def step(self, grads):
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 3: schedules

def step_decay(t, base_lr, every, factor=0.1):
    """base_lr * factor ** (t // every), for step t = 0, 1, ..."""
    raise NotImplementedError  # TODO


def warmup_cosine(t, base_lr, warmup, total, min_lr=0.0):
    """Linear warm-up from base_lr / warmup at t = 0 to base_lr at t = warmup - 1, then cosine decay
    from base_lr at t = warmup to min_lr at t = total:
        min_lr + 0.5 (base_lr - min_lr)(1 + cos(pi (t - warmup) / (total - warmup)))."""
    raise NotImplementedError  # TODO


def inverse_sqrt(t, base_lr, warmup):
    """The original transformer's schedule, scaled so that its peak is base_lr:
    base_lr * min((t + 1) / warmup, sqrt(warmup / (t + 1)))."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 4: metrics

def confusion_matrix(y_true, y_pred, k=10):
    """C[i, j] = number of examples of class i predicted as class j."""
    raise NotImplementedError  # TODO


def expected_calibration_error(probs, y, bins=10):
    """ECE with equal-width bins on the confidence (the largest predicted probability):
    sum over bins of (bin size / n) * |accuracy in bin - mean confidence in bin|. A confidence of
    exactly 1 goes in the last bin; empty bins contribute nothing."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- given: a training loop

def train(make_opt, epochs=3, batch=128, schedule=None, seed=0, data=None, hidden=128):
    """Train an MLP on MNIST with the optimiser make_opt(params) for `epochs` epochs.
    schedule(t), if given, sets opt.lr before step t. Returns (model, losses per step, test accuracy)."""
    Xtr, ytr, Xte, yte = data if data is not None else load_mnist()
    rng = np.random.default_rng(seed)
    model = MLP(rng, hidden); opt = make_opt(model.params)
    losses, t = [], 0
    for _ in range(epochs):
        order = rng.permutation(len(ytr))
        for i in range(0, len(ytr), batch):
            idx = order[i:i + batch]
            if schedule is not None:
                opt.lr = schedule(t)
            loss, g = softmax_cross_entropy(model.forward(Xtr[idx]), ytr[idx])
            opt.step(model.backward(g)); losses.append(loss); t += 1
    acc = np.mean(model.forward(Xte).argmax(1) == yte)
    return model, np.array(losses), acc
