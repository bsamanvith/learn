import numpy as np

rng = np.random.default_rng(0)

def num_grad(f, X, h=1e-6):                        # central differences, one entry at a time
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        E = np.zeros_like(X); E[idx] = h
        G[idx] = (f(X + E) - f(X - E)) / (2 * h)
    return G

def check(name, f, grad, X):
    err = np.max(np.abs(grad(X) - num_grad(f, X))) / max(1.0, np.max(np.abs(grad(X))))
    print(f"{name:42s} relative error {err:.1e}")

n, d = 8, 4
A = rng.standard_normal((n, d)); y = rng.standard_normal(n)
M = rng.standard_normal((d, d)); B = rng.standard_normal((3, d)); C = rng.standard_normal((d, 3))
P = rng.standard_normal((d, d)); P = P @ P.T + d * np.eye(d)       # symmetric positive definite

check("||Aw - y||^2  ->  2 A^T (Aw - y)", lambda w: np.sum((A @ w - y) ** 2), lambda w: 2 * A.T @ (A @ w - y), rng.standard_normal(d))
check("w^T M w  ->  (M + M^T) w", lambda w: w @ M @ w, lambda w: (M + M.T) @ w, rng.standard_normal(d))
check("tr(B W C)  ->  B^T C^T", lambda W: np.trace(B @ W @ C), lambda W: B.T @ C.T, rng.standard_normal((d, d)))
check("log det S  ->  S^{-1} (S sym. pos. def.)", lambda S: np.linalg.slogdet(S)[1], lambda S: np.linalg.inv(S).T, P)
check("tr(S^{-1} K)  ->  -S^{-1} K S^{-1}", lambda S: np.trace(np.linalg.solve(S, M @ M.T)),
      lambda S: -(np.linalg.inv(S) @ M @ M.T @ np.linalg.inv(S)).T, P)

# The Gaussian MLE of the covariance is the sample covariance: the gradient vanishes there
X = rng.multivariate_normal(np.zeros(d), P, size=500)
S_hat = X.T @ X / len(X)
nll = lambda S: 0.5 * len(X) * np.linalg.slogdet(S)[1] + 0.5 * np.trace(np.linalg.solve(S, X.T @ X))
print("gradient of Gaussian NLL at the sample covariance: max |entry| =", f"{np.max(np.abs(num_grad(nll, S_hat))):.1e}")

# Vector-Jacobian products: backprop never builds the Jacobian
W1, W2 = rng.standard_normal((300, 200)), rng.standard_normal((100, 300))
f = lambda x: W2 @ np.tanh(W1 @ x)                 # R^200 -> R^100
x = rng.standard_normal(200); v = rng.standard_normal(100)
J = W2 @ (np.diag(1 - np.tanh(W1 @ x) ** 2) @ W1)  # the full 100 x 200 Jacobian
vjp = W1.T @ ((1 - np.tanh(W1 @ x) ** 2) * (W2.T @ v))   # v^T J computed right to left, no Jacobian
print("v^T J by building J vs by a VJP: max difference", f"{np.max(np.abs(v @ J - vjp)):.1e}")

# ||Aw - y||^2  ->  2 A^T (Aw - y)           relative error 1.1e-10
# w^T M w  ->  (M + M^T) w                   relative error 2.4e-11
# tr(B W C)  ->  B^T C^T                     relative error 2.0e-10
# log det S  ->  S^{-1} (S sym. pos. def.)   relative error 5.1e-10
# tr(S^{-1} K)  ->  -S^{-1} K S^{-1}         relative error 6.0e-10
# gradient of Gaussian NLL at the sample covariance: max |entry| = 2.3e-07
# v^T J by building J vs by a VJP: max difference 5.7e-14
