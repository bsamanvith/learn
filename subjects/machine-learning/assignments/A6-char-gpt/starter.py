"""Assignment A6: a character-level GPT, in NumPy.

Fill in every function and method marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

Conventions. Activations have shape (B, T, d): batch, time, features. Every backward(g) takes
the gradient of the loss with respect to the layer's output, stores the gradients of the
layer's parameters in self.grads (a dict with the same keys as self.params), and returns the
gradient with respect to the layer's input. Linear and Embedding are given as examples.
"""
import os
import urllib.request
import numpy as np


# ---------------------------------------------------------------- given: two layers, as examples

class Linear:
    def __init__(self, n_in, n_out, rng, std=None):
        self.params = {"W": (rng.standard_normal((n_in, n_out)) * (std or np.sqrt(1 / n_in))).astype(np.float32),
                       "b": np.zeros(n_out, np.float32)}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, g):
        x2, g2 = self.x.reshape(-1, self.x.shape[-1]), g.reshape(-1, g.shape[-1])
        self.grads = {"W": x2.T @ g2, "b": g2.sum(0)}
        return g @ self.params["W"].T


class Embedding:
    def __init__(self, n, d, rng):
        self.params = {"E": (rng.standard_normal((n, d)) * 0.02).astype(np.float32)}

    def forward(self, idx):
        self.idx = idx
        return self.params["E"][idx]

    def backward(self, g):
        dE = np.zeros_like(self.params["E"])
        np.add.at(dE, self.idx, g)
        self.grads = {"E": dE}


# ---------------------------------------------------------------- Part 1: the pieces

def softmax(z, axis=-1):
    """Softmax along an axis, stable for any size of input (subtract the max)."""
    raise NotImplementedError  # TODO


def cross_entropy(logits, targets):
    """Mean cross-entropy (natural log) of logits (..., V) against integer targets (...).
    Returns (loss, dloss/dlogits)."""
    raise NotImplementedError  # TODO


class LayerNorm:
    """y = gain * (x - mean) / sqrt(var + eps) + bias over the last axis."""
    def __init__(self, d, eps=1e-5):
        self.eps = eps
        self.params = {"g": np.ones(d, np.float32), "b": np.zeros(d, np.float32)}

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO


class GELU:
    """The tanh approximation: 0.5 x (1 + tanh(c (x + 0.044715 x^3))), c = sqrt(2 / pi)."""
    params = {}

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: causal self-attention

class CausalSelfAttention:
    """Multi-head attention in which position t attends to positions 1..t only.
    The query, key and value projections are one Linear(d, 3d); the output projection is
    Linear(d, d). self.att must hold the attention weights (B, heads, T, T) after forward."""
    def __init__(self, d, heads, rng, proj_std=None):
        self.h = heads
        self.qkv = Linear(d, 3 * d, rng)
        self.proj = Linear(d, d, rng, std=proj_std)
        self.params = {}

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO

    def step(self, x, cache):
        """Process ONE new position for generation. x is (B, 1, d); cache is a dict holding
        the keys and values of all earlier positions as arrays (B, heads, t, d_head) under
        "k" and "v" (empty at the start: cache == {}). Append this position's key and value
        to the cache and return the output for this position, (B, 1, d). Must equal the last
        position of forward() on the whole sequence."""
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 3: the model

class Block:
    """Pre-norm block: x = x + attn(ln1(x)); x = x + fc2(gelu(fc1(ln2(x)))), with fc1: d -> 4d."""
    def __init__(self, d, heads, rng, proj_std):
        self.ln1, self.attn = LayerNorm(d), CausalSelfAttention(d, heads, rng, proj_std)
        self.ln2, self.fc1, self.act, self.fc2 = LayerNorm(d), Linear(d, 4 * d, rng), GELU(), Linear(4 * d, d, rng, std=proj_std)
        self.params = {}

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO

    def step(self, x, cache):
        """One position with a KV cache, as in CausalSelfAttention.step."""
        raise NotImplementedError  # TODO


class GPT:
    """Token and position embeddings, n_layer blocks, a final LayerNorm and a linear head."""
    def __init__(self, vocab, block_size, d=96, heads=4, n_layer=4, seed=0):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.tok, self.pos = Embedding(vocab, d, rng), Embedding(block_size, d, rng)
        proj_std = 0.02 / np.sqrt(2 * n_layer)               # GPT-2: scale residual projections down
        self.blocks = [Block(d, heads, rng, proj_std) for _ in range(n_layer)]
        self.ln_f, self.head = LayerNorm(d), Linear(d, vocab, rng, std=0.02)

    def modules(self):
        """Every object with params, in a fixed order (used by the optimiser and the checker)."""
        out = [self.tok, self.pos]
        for b in self.blocks:
            out += [b.ln1, b.attn.qkv, b.attn.proj, b.ln2, b.fc1, b.fc2]
        return out + [self.ln_f, self.head]

    def forward(self, idx):
        """idx (B, T) of token ids, T <= block_size. Returns logits (B, T, vocab)."""
        raise NotImplementedError  # TODO

    def backward(self, g):
        """g = dloss/dlogits. Fills every module's grads."""
        raise NotImplementedError  # TODO

    def step(self, idx_t, t, caches):
        """Logits (B, vocab) for one new token idx_t (B,) at position t, using and extending
        caches (one dict per block)."""
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 4: training

def get_batch(data, B, T, rng):
    """B random windows of data (a 1-D int array): inputs x (B, T) and targets y (B, T),
    where y is x shifted by one position (y[:, i] = data[start + i + 1])."""
    raise NotImplementedError  # TODO


def lr_schedule(step, max_lr, warmup, total, min_lr):
    """Linear warm-up from 0 to max_lr over `warmup` steps (step 1 gives max_lr / warmup), then
    cosine decay to min_lr at step `total`, and min_lr after that."""
    raise NotImplementedError  # TODO


def clip_grad_norm(modules, max_norm):
    """Scale all gradients by one factor so that their global L2 norm is at most max_norm.
    Returns the norm before clipping."""
    raise NotImplementedError  # TODO


class AdamW:
    """Adam with decoupled weight decay (Loshchilov and Hutter): each step,
        m = b1 m + (1 - b1) g,  v = b2 v + (1 - b2) g^2,
        p -= lr * ( m_hat / (sqrt(v_hat) + eps) + wd * p ),
    with bias-corrected m_hat, v_hat. Weight decay applies only to parameters with 2 or more
    dimensions (weight matrices and embeddings), never to biases or LayerNorm gains."""
    def __init__(self, modules, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
        self.modules, self.b1, self.b2, self.eps, self.wd, self.t = modules, b1, b2, eps, wd, 0
        self.m = [{k: np.zeros_like(p) for k, p in mod.params.items()} for mod in modules]
        self.v = [{k: np.zeros_like(p) for k, p in mod.params.items()} for mod in modules]

    def step(self, lr):
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 5: generation

def generate(model, prompt, n_new, rng, temperature=1.0, top_k=None):
    """Extend the 1-D int array prompt by n_new tokens, one at a time, using model.step and a
    KV cache (process the prompt through step() too). Sample from softmax(logits / temperature)
    restricted to the top_k most likely tokens if top_k is given; temperature == 0 means
    greedy (argmax). Stop growing the context at block_size: the total length prompt + n_new
    must not exceed model.block_size. Returns the full sequence as a 1-D int array."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- given: data

def load_shakespeare(cache=os.path.expanduser("~/.cache/learn-text")):
    """Tiny Shakespeare (1.1 MB). Returns (text, chars): the text and its sorted character set."""
    path = os.path.join(cache, "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
                                   "tinyshakespeare/input.txt", path)
    text = open(path).read()
    return text, sorted(set(text))
