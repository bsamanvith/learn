import numpy as np

rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-z))

# The task: read a sequence of T random bits (as +-1 inputs, plus noise) and output the FIRST bit.
def make(n, T):
    bits = rng.integers(0, 2, (n, T))
    return (2 * bits - 1)[:, :, None] + 0.2 * rng.standard_normal((n, T, 1)), bits[:, 0]

class RNN:
    """h_t = tanh(W h_{t-1} + U x_t + b); the prediction reads h_T."""
    def __init__(self, H, r):
        self.P = {"W": np.linalg.qr(r.standard_normal((H, H)))[0], "U": r.standard_normal((H, 1)), "b": np.zeros(H),
                  "v": r.standard_normal(H) / np.sqrt(H), "c": np.zeros(1)}
    def loss_grads(self, X, y):
        P, (N, T, _) = self.P, X.shape
        hs = [np.zeros((N, len(P["b"])))]
        for t in range(T):
            hs.append(np.tanh(hs[-1] @ P["W"].T + X[:, t] @ P["U"].T + P["b"]))
        F = hs[-1] @ P["v"] + P["c"][0]
        d = (sig(F) - y) / N
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["v"], G["c"] = hs[-1].T @ d, np.array([d.sum()])
        dh = np.outer(d, P["v"])
        for t in reversed(range(T)):                     # backpropagation through time
            dz = dh * (1 - hs[t + 1] ** 2)
            G["W"] += dz.T @ hs[t]; G["U"] += dz.T @ X[:, t]; G["b"] += dz.sum(0)
            dh = dz @ P["W"]
        return np.mean(np.logaddexp(0, F) - y * F), G, F

class LSTM:
    """Gates i, f, o and candidate g from [h_{t-1}, x_t]; c_t = f*c_{t-1} + i*g; h_t = o*tanh(c_t)."""
    def __init__(self, H, r):
        self.H = H
        self.P = {"W": r.standard_normal((4 * H, H + 1)) / np.sqrt(H + 1), "b": np.zeros(4 * H),
                  "v": r.standard_normal(H) / np.sqrt(H), "c": np.zeros(1)}
        self.P["b"][H:2 * H] = 1.0                       # forget-gate bias 1: remember by default
    def loss_grads(self, X, y):
        P, H, (N, T, _) = self.P, self.H, X.shape
        h, c, cache = np.zeros((N, H)), np.zeros((N, H)), []
        for t in range(T):
            z = np.c_[h, X[:, t]] @ P["W"].T + P["b"]
            i, f, o, g = sig(z[:, :H]), sig(z[:, H:2 * H]), sig(z[:, 2 * H:3 * H]), np.tanh(z[:, 3 * H:])
            c_prev, h_prev = c, h
            c = f * c_prev + i * g; h = o * np.tanh(c)
            cache.append((h_prev, c_prev, i, f, o, g, c))
        F = h @ P["v"] + P["c"][0]
        d = (sig(F) - y) / N
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["v"], G["c"] = h.T @ d, np.array([d.sum()])
        dh, dc = np.outer(d, P["v"]), np.zeros((N, H))
        for t in reversed(range(T)):
            h_prev, c_prev, i, f, o, g, c = cache[t]
            tc = np.tanh(c)
            dc = dc + dh * o * (1 - tc ** 2)
            dz = np.c_[dc * g * i * (1 - i), dc * c_prev * f * (1 - f), dh * tc * o * (1 - o), dc * i * (1 - g ** 2)]
            G["W"] += dz.T @ np.c_[h_prev, X[:, t]]; G["b"] += dz.sum(0)
            dh = dz @ P["W"][:, :H]
            dc = dc * f                                  # the cell path: multiplied only by the forget gate
        return np.mean(np.logaddexp(0, F) - y * F), G, F

def train(model, T, steps=600, lr=0.01, seed=0):
    m = {k: 0 for k in model.P}; v = {k: 0 for k in model.P}
    for t in range(1, steps + 1):
        X, y = make(64, T)
        _, G, _ = model.loss_grads(X, y)
        norm = np.sqrt(sum((g ** 2).sum() for g in G.values()))
        for k, g in G.items():
            g = g * min(1, 1 / norm)                     # clip the global gradient norm at 1
            m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g
            model.P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    X, y = make(2000, T)
    return np.mean((model.loss_grads(X, y)[2] > 0) == y)

# 1. How much does h_T depend on h_0 in an untrained tanh RNN? Norm of the Jacobian dh_T/dh_0.
for scale in [0.5, 1.0, 2.0]:
    r = np.random.default_rng(1); W = scale * np.linalg.qr(r.standard_normal((32, 32)))[0]
    h, J = r.standard_normal(32) * 0.1, np.eye(32)
    norms = {}
    for t in range(1, 51):
        h = np.tanh(W @ h + r.standard_normal(32) * 0.5)
        J = (1 - h ** 2)[:, None] * W @ J
        if t in (1, 10, 50):
            norms[t] = np.linalg.norm(J, 2)
    print(f"W = {scale} x orthogonal: ||dh_t/dh_0|| at t = 1, 10, 50: " + "  ".join(f"{norms[t]:.1e}" for t in (1, 10, 50)))

# 2. Remember the first bit across T steps
for T in [5, 20, 50]:
    a_rnn = train(RNN(32, np.random.default_rng(2)), T)
    a_lstm = train(LSTM(32, np.random.default_rng(2)), T)
    print(f"T = {T:2d}: test accuracy  RNN {a_rnn:.3f}   LSTM {a_lstm:.3f}")

# W = 0.5 x orthogonal: ||dh_t/dh_0|| at t = 1, 10, 50: 5.0e-01  3.1e-04  2.5e-19
# W = 1.0 x orthogonal: ||dh_t/dh_0|| at t = 1, 10, 50: 1.0e+00  1.9e-01  4.6e-06
# W = 2.0 x orthogonal: ||dh_t/dh_0|| at t = 1, 10, 50: 2.0e+00  2.5e+01  3.6e+03
# T =  5: test accuracy  RNN 1.000   LSTM 1.000
# T = 20: test accuracy  RNN 0.983   LSTM 0.999
# T = 50: test accuracy  RNN 0.504   LSTM 0.896
