"""Assignment A4: a tiny autograd engine and an MLP on MNIST.

Fill in every function marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

The checker tests each piece on its own and tells you what is wrong, so you can work
through them in any order, but Part 1 is used by everything after it.
Lesson 13 builds the scalar version of this engine; here every node holds a whole array.
"""
import gzip
import os
import urllib.request
import numpy as np


# ---------------------------------------------------------------- Part 1: the engine (lesson 13)

def unbroadcast(grad, shape):
    """Sum grad over the axes that NumPy broadcasting added or stretched, so that the
    result has exactly `shape`. Example: a (4, 3) gradient for a (3,) bias sums over
    axis 0; a (4, 3) gradient for a (4, 1) input sums over axis 1, keeping it."""
    raise NotImplementedError  # TODO


class Tensor:
    """An array that remembers how it was computed.

    data: the np.ndarray (float64). grad: an array of the same shape, or None until
    backward() has reached this tensor. Each operation returns a new Tensor whose
    _parents are its inputs and whose _backward() adds this tensor's grad, times the
    local derivative, into each parent's grad (with +=, starting from zeros)."""

    def __init__(self, data, parents=(), requires_grad=True):
        self.data = np.asarray(data, dtype=float)
        self.grad = None
        self._parents = parents
        self._backward = lambda: None
        self.requires_grad = requires_grad

    shape = property(lambda self: self.data.shape)

    def _acc(self, g):
        """Add g (already the right shape) into self.grad."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    def __add__(self, other):
        # A worked example of the pattern every operation follows (it still needs unbroadcast).
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other))
        def _backward():
            self._acc(unbroadcast(out.grad, self.shape))
            other._acc(unbroadcast(out.grad, other.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        """Elementwise product, with broadcasting and plain numbers allowed, like __add__."""
        raise NotImplementedError  # TODO

    def __matmul__(self, other):
        """2-D matrix product only: (n, k) @ (k, m)."""
        raise NotImplementedError  # TODO

    def sum(self, axis=None, keepdims=False):
        raise NotImplementedError  # TODO

    def relu(self):
        raise NotImplementedError  # TODO

    def exp(self):
        raise NotImplementedError  # TODO

    def log(self):
        raise NotImplementedError  # TODO

    @property
    def T(self):
        raise NotImplementedError  # TODO

    def __neg__(self): return self * -1.0
    def __sub__(self, other): return self + (-other if isinstance(other, Tensor) else -np.asarray(other, float))
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def mean(self): return self.sum() * (1.0 / self.data.size)

    def backward(self):
        """Set this tensor's grad to ones and run every _backward in reverse topological
        order. Use an explicit stack, not recursion: the checker builds a graph 5,000
        operations deep, which overflows Python's recursion limit."""
        raise NotImplementedError  # TODO


def cross_entropy(logits, targets):
    """Mean cross-entropy of softmax(logits) for integer class targets, as ONE node.

    logits: Tensor (N, K); targets: int array (N,). Forward: use the log-sum-exp trick
    (subtract each row's max) so logits of 1000 don't overflow. Backward: the gradient
    with respect to the logits is (softmax - onehot) / N (lesson 06)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: layers (lessons 12, 14)

class Linear:
    """y = x @ W + b with W of shape (fan_in, fan_out), He initialisation
    W ~ N(0, 2 / fan_in) drawn with rng.standard_normal, and b = 0."""
    def __init__(self, fan_in, fan_out, rng):
        raise NotImplementedError  # TODO

    def __call__(self, x):
        raise NotImplementedError  # TODO

    def params(self):
        return [self.W, self.b]


def dropout(x, p, training, rng):
    """Inverted dropout. In training, zero each entry independently with probability p
    (mask = rng.random(x.shape) >= p) and scale the survivors by 1 / (1 - p), so the
    expected value is unchanged; the gradient flows only through the survivors, with the
    same scale. When not training, or p == 0, return x unchanged."""
    raise NotImplementedError  # TODO


class MLP:
    """Linear layers with ReLU (and dropout, if p > 0) between them; no activation after
    the last layer. sizes = [784, 256, 10] means one hidden layer of 256."""
    def __init__(self, sizes, rng, p_drop=0.0):
        self.layers = [Linear(k, m, rng) for k, m in zip(sizes[:-1], sizes[1:])]
        self.p_drop, self.rng, self.training = p_drop, rng, True

    def __call__(self, x):
        raise NotImplementedError  # TODO

    def params(self):
        return [p for layer in self.layers for p in layer.params()]


# ---------------------------------------------------------------- Part 3: optimisers (lessons 07, 14)

class SGD:
    """Heavy-ball momentum as in PyTorch: g = grad + weight_decay * p; v = momentum * v + g;
    p -= lr * v. Parameters whose grad is None are skipped."""
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.params, self.lr, self.mu, self.wd = params, lr, momentum, weight_decay
        self.v = [np.zeros_like(p.data) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError  # TODO


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019). With t counting steps
    from 1: m = b1 m + (1 - b1) g; v = b2 v + (1 - b2) g^2; m_hat = m / (1 - b1^t);
    v_hat = v / (1 - b2^t); p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params, self.lr, (self.b1, self.b2), self.eps, self.wd = params, lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 4: MNIST (provided)

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


def evaluate(model, X, y, batch=1000):
    """Accuracy of model on (X, y), with dropout switched off."""
    model.training = False
    correct = sum(int((model(Tensor(X[i:i + batch], requires_grad=False)).data.argmax(1) == y[i:i + batch]).sum())
                  for i in range(0, len(y), batch))
    model.training = True
    return correct / len(y)
