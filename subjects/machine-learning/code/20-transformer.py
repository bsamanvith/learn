import numpy as np

rng = np.random.default_rng(0)

# ---- layers, each with forward(x) and backward(grad of output) -> grad of input ----
class Linear:
    def __init__(self, n_in, n_out):
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(1 / n_in); self.b = np.zeros(n_out)
    def forward(self, x):
        self.x = x; return x @ self.W + self.b
    def backward(self, g):
        self.dW = (self.x.reshape(-1, self.x.shape[-1]).T @ g.reshape(-1, g.shape[-1])); self.db = g.reshape(-1, g.shape[-1]).sum(0)
        return g @ self.W.T
    def params(self): return [(self, "W"), (self, "b")]

class LayerNorm:
    def __init__(self, d):
        self.g = np.ones(d); self.b = np.zeros(d)
    def forward(self, x):
        self.mu, self.sd = x.mean(-1, keepdims=True), np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        self.xh = (x - self.mu) / self.sd; return self.g * self.xh + self.b
    def backward(self, gy):
        self.dg = (gy * self.xh).reshape(-1, len(self.g)).sum(0); self.db = gy.reshape(-1, len(self.b)).sum(0)
        gx = gy * self.g                                          # lesson 14, exercise 5
        return (gx - gx.mean(-1, keepdims=True) - self.xh * (gx * self.xh).mean(-1, keepdims=True)) / self.sd
    def params(self): return [(self, "g"), (self, "b")]

class CausalSelfAttention:
    def __init__(self, d, heads):
        self.h, self.qkv, self.proj = heads, Linear(d, 3 * d), Linear(d, d)
    def forward(self, x):
        B, T, d = x.shape; h, dh = self.h, d // self.h
        q, k, v = np.split(self.qkv.forward(x), 3, -1)
        q, k, v = [z.reshape(B, T, h, dh).transpose(0, 2, 1, 3) for z in (q, k, v)]       # (B, h, T, dh)
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)                       # causal mask
        a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        self.cache = (q, k, v, a)
        return self.proj.forward((a @ v).transpose(0, 2, 1, 3).reshape(B, T, d))
    def backward(self, g):
        q, k, v, a = self.cache; B, h, T, dh = q.shape
        go = self.proj.backward(g).reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        da, dv = go @ v.transpose(0, 1, 3, 2), a.transpose(0, 1, 3, 2) @ go
        ds = a * (da - (da * a).sum(-1, keepdims=True)) / np.sqrt(dh)                   # lesson 19's formulas
        dq, dk = ds @ k, ds.transpose(0, 1, 3, 2) @ q
        merge = lambda z: z.transpose(0, 2, 1, 3).reshape(B, T, h * dh)
        return self.qkv.backward(np.concatenate([merge(dq), merge(dk), merge(dv)], -1))
    def params(self): return self.qkv.params() + self.proj.params()

class MLP:
    def __init__(self, d):
        self.fc, self.out = Linear(d, 4 * d), Linear(4 * d, d)
    def forward(self, x):
        self.z = self.fc.forward(x); return self.out.forward(np.maximum(self.z, 0))
    def backward(self, g):
        return self.fc.backward(self.out.backward(g) * (self.z > 0))
    def params(self): return self.fc.params() + self.out.params()

class Block:
    """Pre-norm transformer block: x + attn(LN(x)), then x + mlp(LN(x))."""
    def __init__(self, d, heads):
        self.ln1, self.attn, self.ln2, self.mlp = LayerNorm(d), CausalSelfAttention(d, heads), LayerNorm(d), MLP(d)
    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.mlp.forward(self.ln2.forward(x))
    def backward(self, g):
        g = g + self.ln2.backward(self.mlp.backward(g))          # the residual path passes g straight through
        return g + self.ln1.backward(self.attn.backward(g))
    def params(self): return self.ln1.params() + self.attn.params() + self.ln2.params() + self.mlp.params()

class GPT:
    """Token and position embeddings, a stack of blocks, a final norm, and a linear head to logits."""
    def __init__(self, vocab, T, d=48, heads=4, layers=2):
        self.tok, self.pos = rng.standard_normal((vocab, d)) * 0.1, rng.standard_normal((T, d)) * 0.1
        self.blocks = [Block(d, heads) for _ in range(layers)]
        self.ln, self.head = LayerNorm(d), Linear(d, vocab)
    def forward(self, idx):
        self.idx = idx
        x = self.tok[idx] + self.pos[:idx.shape[1]]
        for b in self.blocks:
            x = b.forward(x)
        return self.head.forward(self.ln.forward(x))
    def backward(self, g):
        g = self.ln.backward(self.head.backward(g))
        for b in reversed(self.blocks):
            g = b.backward(g)
        self.dtok = np.zeros_like(self.tok); np.add.at(self.dtok, self.idx, g)
        self.dpos = np.zeros_like(self.pos); self.dpos[:g.shape[1]] = g.sum(0)
    def params(self):
        ps = [(self, "tok"), (self, "pos")]
        for b in self.blocks:
            ps += b.params()
        return ps + self.ln.params() + self.head.params()

def loss_and_grad(model, idx, targets, mask):
    """Cross-entropy on the positions where mask is True; returns the loss and fills in gradients."""
    F = model.forward(idx)
    Z = F - F.max(-1, keepdims=True); P = np.exp(Z) / np.exp(Z).sum(-1, keepdims=True)
    n = mask.sum()
    loss = -np.log(P[mask, targets[mask]]).sum() / n
    G = P.copy(); G[mask, targets[mask]] -= 1; G[~mask] = 0
    model.backward(G / n)
    return loss

grad_name = lambda obj, name: "d" + name

# The task: reverse a sequence of 8 digits. Input "3 1 4 1 5 9 2 6 > 6 2 9 5 1 4 1 3"; the model is a
# language model trained to predict each next token, and is scored only on the reversed half.
L, SEP = 8, 10
def make(n):
    a = rng.integers(0, 10, (n, L))
    seq = np.concatenate([a, np.full((n, 1), SEP), a[:, ::-1]], 1)
    mask = np.zeros((n, 2 * L), bool); mask[:, L:] = True             # predict positions after the separator
    return seq[:, :-1], seq[:, 1:], mask

model = GPT(vocab=11, T=2 * L)
print(f"parameters: {sum(getattr(o, n).size for o, n in model.params())}")

# gradient check on a few entries of every parameter tensor
idx, tgt, mask = make(3)
loss_and_grad(model, idx, tgt, mask)
worst = 0.0
for o, n in model.params():
    P, G = getattr(o, n), getattr(o, grad_name(o, n))
    for i in [tuple(rng.integers(0, s) for s in P.shape) for _ in range(3)]:
        old = P[i]; P[i] = old + 1e-5
        up = loss_and_grad(model, idx, tgt, mask); P[i] = old - 1e-5
        dn = loss_and_grad(model, idx, tgt, mask); P[i] = old
        worst = max(worst, abs((up - dn) / 2e-5 - G[i]))
    loss_and_grad(model, idx, tgt, mask)
print(f"gradient check: max |backprop - finite difference| over sampled entries {worst:.1e}")

# Train with Adam
state, losses = {}, []
for t in range(1, 3001):
    idx, tgt, mask = make(64)
    loss = loss_and_grad(model, idx, tgt, mask); losses.append(loss)
    lr = 3e-3 * min(1, t / 200)                                       # linear warm-up
    for o, n in model.params():
        g = getattr(o, grad_name(o, n)); m, v = state.get((id(o), n), (0, 0))
        m, v = 0.9 * m + 0.1 * g, 0.99 * v + 0.01 * g * g; state[(id(o), n)] = (m, v)
        setattr(o, n, getattr(o, n) - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.99 ** t)) + 1e-8))
    if t in (1, 500, 1000, 2000, 3000):
        print(f"step {t:4d}  loss {loss:.3f}")

# Generate: feed the digits and the separator, then sample greedily one token at a time
a = rng.integers(0, 10, (1000, L)); seq = np.concatenate([a, np.full((1000, 1), SEP)], 1)
for _ in range(L):
    seq = np.concatenate([seq, model.forward(seq)[:, -1].argmax(-1)[:, None]], 1)
print(f"test: {np.mean(np.all(seq[:, L + 1:] == a[:, ::-1], 1)):.3f} of 1,000 sequences reversed exactly;"
      f" per-digit accuracy {np.mean(seq[:, L + 1:] == a[:, ::-1]):.3f}")
print("example:", " ".join(map(str, a[0])), "->", " ".join(map(str, seq[0, L + 1:])))

# parameters: 58475
# gradient check: max |backprop - finite difference| over sampled entries 4.9e-11
# step    1  loss 2.674
# step  500  loss 0.000
# step 1000  loss 0.000
# step 2000  loss 0.000
# step 3000  loss 0.000
# test: 1.000 of 1,000 sequences reversed exactly; per-digit accuracy 1.000
# example: 5 8 2 1 8 8 9 8 -> 8 9 8 8 1 2 8 5
