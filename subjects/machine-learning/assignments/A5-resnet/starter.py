"""Assignment A5: a residual convolutional network in NumPy.

Fill in every function marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

Every layer has the same interface (lesson 20's code uses it too):
    out = layer.forward(x)     computes the output and caches what backward needs
    dx  = layer.backward(g)    takes dLoss/d(out), stores dLoss/d(parameter) in
                               layer.grads[name] for each name in layer.params,
                               and returns dLoss/dx
Images are arrays of shape (N, C, H, W): batch, channels, height, width.
Linear and ReLU are written for you as worked examples of the pattern.
"""
import gzip
import os
import urllib.request
import numpy as np

rng = np.random.default_rng(0)


class Layer:
    params = {}
    training = True

    def __init__(self):
        self.params, self.grads = {}, {}


# ---------------------------------------------------------------- worked examples

class Linear(Layer):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.params = {"W": rng.standard_normal((n_in, n_out)) * np.sqrt(1 / n_in), "b": np.zeros(n_out)}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, g):
        self.grads = {"W": self.x.T @ g, "b": g.sum(0)}
        return g @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask


# ---------------------------------------------------------------- Part 1: convolution (lesson 15)

def out_size(n, k, stride, pad):
    return (n + 2 * pad - k) // stride + 1


def im2col(x, k, stride=1, pad=0):
    """Unfold every k x k window of x (N, C, H, W), after zero-padding by `pad` on each side,
    into one row. Returns (N * OH * OW, C * k * k). Rows are ordered by (n, output row,
    output column); within a row, entries are ordered by (channel, kernel row, kernel column),
    which is the order of W.reshape(F, C * k * k) for a weight array W of shape (F, C, k, k)."""
    raise NotImplementedError  # TODO


def col2im(cols, shape, k, stride=1, pad=0):
    """The adjoint of im2col: scatter-add each row of cols back to the window it came from.
    shape is the (N, C, H, W) of the original x. Where windows overlap, contributions add.
    Returns an array of that shape (the padding is cut off again)."""
    raise NotImplementedError  # TODO


class Conv2d(Layer):
    """F filters of shape (C, k, k), applied with the given stride and zero padding
    (default k // 2, which keeps the size when stride is 1). He initialisation."""

    def __init__(self, C, F, k, stride=1, pad=None, bias=True):
        super().__init__()
        self.k, self.stride, self.pad = k, stride, k // 2 if pad is None else pad
        self.params = {"W": rng.standard_normal((F, C, k, k)) * np.sqrt(2 / (C * k * k))}
        if bias:
            self.params["b"] = np.zeros(F)

    def forward(self, x):
        """Output (N, F, OH, OW): out[n, f] = sum over c, i, j of W[f, c, i, j] * window + b[f]."""
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO


class MaxPool2d(Layer):
    """Non-overlapping 2 x 2 max pooling (H and W must be even). Where a window has a tie,
    the gradient goes to exactly one of the tied entries (the first in row-major order)."""

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO


class GlobalAvgPool(Layer):
    """(N, C, H, W) -> (N, C): the mean over all positions of each channel."""

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: batch norm and the loss (lessons 14, 16)

class BatchNorm2d(Layer):
    """Per-channel batch normalisation. In training mode, normalise with the batch's mean and
    (biased) variance over (N, H, W), then scale by gamma and shift by beta; also update
    running = (1 - momentum) * running + momentum * batch statistic, using the unbiased
    variance (times n / (n - 1), n = N * H * W) for running_var. In evaluation mode
    (self.training False), normalise with the running statistics instead."""

    def __init__(self, C, momentum=0.1, eps=1e-5, gamma=1.0):
        super().__init__()
        self.params = {"gamma": np.full(C, float(gamma)), "beta": np.zeros(C)}
        self.running_mean, self.running_var = np.zeros(C), np.ones(C)
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO


def cross_entropy(logits, y):
    """Mean softmax cross-entropy of logits (N, K) against integer labels y (N,).
    Returns (loss, dloss/dlogits). Must not overflow for logits in the thousands."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 3: the residual block (lesson 16)

class ResidualBlock(Layer):
    """out = relu(main(x) + shortcut(x)), with
        main(x)     = bn2(conv2(relu(bn1(conv1(x)))))
        conv1: 3 x 3, C_in -> C_out, the given stride, no bias;  conv2: 3 x 3, C_out -> C_out, no bias
        bn2's gamma starts at 0, so that the block starts as relu(shortcut(x)).
    shortcut(x) is x itself when stride == 1 and C_in == C_out; otherwise it is
    bn_s(conv_s(x)) with a 1 x 1 convolution of the given stride, no bias, no padding."""

    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.main = [Conv2d(C_in, C_out, 3, stride, bias=False), BatchNorm2d(C_out), ReLU(),
                     Conv2d(C_out, C_out, 3, 1, bias=False), BatchNorm2d(C_out, gamma=0.0)]
        self.short = [] if stride == 1 and C_in == C_out else [Conv2d(C_in, C_out, 1, stride, pad=0, bias=False), BatchNorm2d(C_out)]
        self.relu = ReLU()

    def sublayers(self):
        return self.main + self.short + [self.relu]

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- given: the network container

class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def all_layers(self):
        out = []
        for l in self.layers:
            out += l.sublayers() if hasattr(l, "sublayers") else [l]
        return out

    def train(self, mode=True):
        for l in self.all_layers():
            l.training = mode

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g

    def parameters(self):
        """(layer, name) for every parameter; the array is layer.params[name], its gradient layer.grads[name]."""
        return [(l, n) for l in self.all_layers() for n in l.params]


def resnet(channels=(16, 32, 64), n_classes=10, C_in=1):
    """Stem conv + BN + ReLU + 2 x 2 max pooling, one residual block per stage (the 2nd and 3rd
    halve the size), global average pooling, and a linear classifier. For 28 x 28 MNIST
    images the stages work at 14 x 14, 7 x 7 and 4 x 4."""
    a, b, c = channels
    return Sequential(Conv2d(C_in, a, 3, bias=False), BatchNorm2d(a), ReLU(), MaxPool2d(),
                      ResidualBlock(a, a), ResidualBlock(a, b, 2), ResidualBlock(b, c, 2),
                      GlobalAvgPool(), Linear(c, n_classes))


# ---------------------------------------------------------------- Part 4: training

class SGD:
    """SGD with momentum and coupled weight decay, for every (layer, name) in params:
        g = grad + weight_decay * p;  v = momentum * v + g;  p = p - lr * v
    with v starting at 0. Don't decay batch-norm parameters or biases: apply weight
    decay only to parameters named "W"."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        self.params, self.lr, self.momentum, self.wd = params, lr, momentum, weight_decay
        self.v = {}

    def step(self):
        raise NotImplementedError  # TODO


def augment(X, rng, max_shift=2):
    """Shift each image (N, C, H, W) by its own random offset (dy, dx), each uniform on the
    integers -max_shift..max_shift, filling the uncovered border with zeros."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- given: data and evaluation

def load_mnist(cache=os.path.expanduser("~/.cache/learn-mnist")):
    """Download MNIST once (about 11 MB) and return (Xtr, ytr, Xte, yte): images as
    (N, 1, 28, 28) floats in [0, 1], labels as ints. 60,000 training and 10,000 test images."""
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
        out.append(raw[16:].reshape(-1, 1, 28, 28) / 255.0 if "images" in name else raw[8:].astype(int))
    return out


def evaluate(model, X, y, batch=500):
    """Accuracy of model on (X, y), in evaluation mode (batch norm uses running statistics)."""
    model.train(False)
    correct = sum(int((model.forward(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum()) for i in range(0, len(y), batch))
    model.train(True)
    return correct / len(y)
