import numpy as np

rng = np.random.default_rng(1)
K, n_per = 3, 100
centres = np.array([[-2.0, 0.0], [2.0, 0.5], [0.0, 2.5]])
X = np.vstack([c + rng.standard_normal((n_per, 2)) for c in centres])
y = np.repeat(np.arange(K), n_per)
Y = np.eye(K)[y]                                   # one-hot targets, shape (n, K)
n = len(y)

def log_softmax(Z):                                # stable: subtract the row max first
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))

def loss_and_grad(W, b):
    logp = log_softmax(X @ W.T + b)                # (n, K) log-probabilities
    loss = -(Y * logp).sum() / n                   # mean cross-entropy
    G = (np.exp(logp) - Y) / n                     # dL/dz = p - y, per example
    return loss, G.T @ X, G.sum(axis=0)            # dL/dW (K x d), dL/db (K)

W, b = np.zeros((K, 2)), np.zeros(K)
for t in range(501):
    loss, gW, gb = loss_and_grad(W, b)
    if t in (0, 10, 100, 500):
        acc = np.mean((X @ W.T + b).argmax(1) == y)
        print(f"step {t:3d}: loss {loss:.4f}  accuracy {acc:.3f}")
    W -= 0.5 * gW; b -= 0.5 * gb

# check one gradient entry against a finite difference
E = np.zeros_like(W); E[1, 0] = 1e-6
num = (loss_and_grad(W + E, b)[0] - loss_and_grad(W - E, b)[0]) / 2e-6
print("dL/dW[1,0]: analytic", round(loss_and_grad(W, b)[1][1, 0], 8), " numeric", round(num, 8))
print("naive softmax of [1000, 0]:", np.exp([1000.0, 0]) / np.exp([1000.0, 0]).sum())
print("stable log-softmax:        ", log_softmax(np.array([[1000.0, 0.0]])))

# step   0: loss 1.0986  accuracy 0.333
# step  10: loss 0.2982  accuracy 0.927
# step 100: loss 0.2183  accuracy 0.917
# step 500: loss 0.2130  accuracy 0.913
# dL/dW[1,0]: analytic -0.00035108  numeric -0.00035108
# naive softmax of [1000, 0]: [nan  0.]         (with an overflow warning)
# stable log-softmax:         [[    0. -1000.]]
