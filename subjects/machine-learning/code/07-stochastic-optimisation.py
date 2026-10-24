import numpy as np

rng = np.random.default_rng(0)
n, d, sigma = 2000, 20, 0.5
X = rng.standard_normal((n, d)) * np.linspace(0.2, 3.0, d)   # features on very different scales
y = X @ rng.standard_normal(d) + sigma * rng.standard_normal(n)
w_best = np.linalg.lstsq(X, y, rcond=None)[0]
L = lambda w: 0.5 * np.mean((X @ w - y) ** 2)
excess = lambda w: L(w) - L(w_best)                # how far above the best possible loss
h = np.linalg.eigvalsh(X.T @ X / n)                # curvatures (Hessian eigenvalues)

def train(update, epochs=30, B=32, seed=1):
    rng = np.random.default_rng(seed)
    w, state, t = np.zeros(d), {}, 0
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(n), n // B):
            g = X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)   # minibatch gradient
            t += 1
            w = update(w, g, state, t)
    return excess(w)

def sgd(lr):
    return lambda w, g, s, t: w - lr * g

def sgd_cosine(lr0, T):                            # learning rate decays from lr0 to 0 over T steps
    return lambda w, g, s, t: w - lr0 * 0.5 * (1 + np.cos(np.pi * t / T)) * g

def momentum(lr, beta=0.9):
    def update(w, g, s, t):
        s["v"] = beta * s.get("v", 0) + g
        return w - lr * s["v"]
    return update

def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    def update(w, g, s, t):
        s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
        s["v"] = b2 * s.get("v", 0) + (1 - b2) * g * g
        m_hat, v_hat = s["m"] / (1 - b1**t), s["v"] / (1 - b2**t)   # bias correction
        return w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return update

print(f"curvatures from {h[0]:.3f} to {h[-1]:.2f}: condition number {h[-1] / h[0]:.0f}")
for lr in [0.1, 0.03, 0.01]:
    floor = lr * sigma**2 / (2 * 32) * np.sum(h / (2 - lr * h))     # predicted noise floor
    print(f"SGD lr={lr:<5} excess {train(sgd(lr)):.1e}   predicted floor {floor:.1e}")
T = 30 * (n // 32)
print(f"SGD cosine from 0.1   excess {train(sgd_cosine(0.1, T)):.1e}")
print(f"momentum lr=0.01      excess {train(momentum(0.01)):.1e}")
print(f"Adam lr=0.003         excess {train(adam(0.003)):.1e}")

# curvatures from 0.041 to 9.48: condition number 231
# SGD lr=0.1   excess 1.8e-02   predicted floor 1.9e-02
# SGD lr=0.03  excess 7.0e-03   predicted floor 4.3e-03
# SGD lr=0.01  excess 1.4e-03   predicted floor 1.3e-03
# SGD cosine from 0.1   excess 6.2e-06
# momentum lr=0.01      excess 2.9e-02
# Adam lr=0.003         excess 2.2e-04
