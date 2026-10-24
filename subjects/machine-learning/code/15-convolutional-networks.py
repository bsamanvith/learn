import numpy as np

rng = np.random.default_rng(0)

def im2col(X, k):
    """X: (N, C, H, W) -> (N, H-k+1, W-k+1, C*k*k): every k-by-k patch as a row."""
    N, C, H, W = X.shape
    s = X.strides
    P = np.lib.stride_tricks.as_strided(X, (N, H - k + 1, W - k + 1, C, k, k), (s[0], s[2], s[3], s[1], s[2], s[3]))
    return P.reshape(N, H - k + 1, W - k + 1, C * k * k)

def conv_forward(X, K, b):
    """'Valid' convolution (really cross-correlation, as in every library). K: (F, C, k, k)."""
    F, C, k, _ = K.shape
    cols = im2col(X, k)                                    # (N, H', W', C k k)
    return (cols @ K.reshape(F, -1).T + b).transpose(0, 3, 1, 2), cols   # (N, F, H', W')

def conv_backward(dY, X, K, cols):
    """Gradients for the kernel, the bias and the input."""
    F, C, k, _ = K.shape
    dYr = dY.transpose(0, 2, 3, 1)                         # (N, H', W', F)
    dK = (dYr.reshape(-1, F).T @ cols.reshape(-1, C * k * k)).reshape(K.shape)
    db = dYr.sum((0, 1, 2))
    dcols = (dYr @ K.reshape(F, -1)).reshape(*dYr.shape[:3], C, k, k)
    dX = np.zeros_like(X)
    for i in range(k):                                     # scatter each patch's gradient back
        for j in range(k):
            dX[:, :, i:i + dcols.shape[1], j:j + dcols.shape[2]] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return dK, db, dX

# gradient check on a small random problem
X = rng.standard_normal((2, 3, 6, 6)); K = rng.standard_normal((4, 3, 3, 3)); b = rng.standard_normal(4)
Y, cols = conv_forward(X, K, b); R = rng.standard_normal(Y.shape)
dK, db, dX = conv_backward(R, X, K, cols)
num = np.zeros_like(X)
for idx in np.ndindex(X.shape):
    for s in (1, -1):
        X[idx] += s * 1e-6; num[idx] += s * np.sum(conv_forward(X, K, b)[0] * R) / 2e-6; X[idx] -= s * 1e-6
print(f"gradient check, input of a conv layer: max relative error {np.max(np.abs(num - dX)) / np.max(np.abs(num)):.1e}")

# data: a 5x5 "+" or "x" at a random place in a 16x16 noisy image
plus = np.zeros((5, 5)); plus[2, :] = plus[:, 2] = 1
cross = np.eye(5) + np.eye(5)[::-1]; cross[2, 2] = 1
def make(n, cols_from, cols_to):
    """Patterns placed with their left edge in columns [cols_from, cols_to)."""
    Xs = 0.3 * rng.standard_normal((n, 1, 16, 16)); y = rng.integers(0, 2, n)
    for i in range(n):
        r, c = rng.integers(0, 12), rng.integers(cols_from, cols_to)
        Xs[i, 0, r:r + 5, c:c + 5] += plus if y[i] else cross
    return Xs, y
Xtr, ytr = make(2000, 0, 6)                                # training: patterns only in the left part
Xin, yin = make(1000, 0, 6)                                # test, same positions
Xout, yout = make(1000, 6, 12)                             # test, positions never seen in training

def adam(params, grads, state, t, lr=0.01):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = state.get(i, (0, 0))
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        state[i] = (m, v)
        p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

def xent(F, y):                                            # mean cross-entropy and its gradient
    Z = F - F.max(1, keepdims=True); P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    loss = -np.log(P[np.arange(len(y)), y]).mean(); P[np.arange(len(y)), y] -= 1
    return loss, P / len(y)

# model A: an MLP on the 256 pixels
W1 = rng.standard_normal((256, 64)) * np.sqrt(2 / 256); b1 = np.zeros(64)
W2 = rng.standard_normal((64, 2)) * np.sqrt(2 / 64); b2 = np.zeros(2)
mlp = lambda X: (np.maximum(X.reshape(len(X), -1) @ W1 + b1, 0) @ W2 + b2)
state = {}
for t in range(1, 301):
    i = rng.integers(0, 2000, 100); A = Xtr[i].reshape(100, -1); Hh = np.maximum(A @ W1 + b1, 0)
    _, d = xent(Hh @ W2 + b2, ytr[i]); dH = (d @ W2.T) * (Hh > 0)
    adam([W1, b1, W2, b2], [A.T @ dH, dH.sum(0), Hh.T @ d, d.sum(0)], state, t)
acc = lambda f, X, y: np.mean(f(X).argmax(1) == y)
print(f"MLP  {W1.size + b1.size + W2.size + b2.size:6d} parameters   test acc, seen positions {acc(mlp, Xin, yin):.3f}   unseen positions {acc(mlp, Xout, yout):.3f}")

# model B: 8 conv filters of 5x5, ReLU, global max pool, then a linear layer
K = rng.standard_normal((8, 1, 5, 5)) * np.sqrt(2 / 25); bk = np.zeros(8)
V = rng.standard_normal((8, 2)) * np.sqrt(2 / 8); bv = np.zeros(2)
def cnn(X, cache=False):
    Z, cols = conv_forward(X, K, bk); A = np.maximum(Z, 0)
    flat = A.reshape(len(X), 8, -1); idx = flat.argmax(2); pooled = flat.max(2)   # (N, 8)
    out = pooled @ V + bv
    return (out, (Z, cols, A, idx, pooled)) if cache else out
state = {}
for t in range(1, 301):
    i = rng.integers(0, 2000, 100)
    out, (Z, cols, A, idx, pooled) = cnn(Xtr[i], cache=True)
    _, d = xent(out, ytr[i]); dpooled = d @ V.T
    dA = np.zeros((100, 8, A.shape[2] * A.shape[3])); np.put_along_axis(dA, idx[:, :, None], dpooled[:, :, None], 2)
    dZ = dA.reshape(A.shape) * (Z > 0)
    dK, dbk, _ = conv_backward(dZ, Xtr[i], K, cols)
    adam([K, bk, V, bv], [dK, dbk, pooled.T @ d, d.sum(0)], state, t)
print(f"CNN  {K.size + bk.size + V.size + bv.size:6d} parameters   test acc, seen positions {acc(cnn, Xin, yin):.3f}   unseen positions {acc(cnn, Xout, yout):.3f}")

# gradient check, input of a conv layer: max relative error 7.1e-10
# MLP   16578 parameters   test acc, seen positions 0.916   unseen positions 0.510
# CNN     226 parameters   test acc, seen positions 1.000   unseen positions 1.000
