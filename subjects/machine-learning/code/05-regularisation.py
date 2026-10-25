import numpy as np

rng = np.random.default_rng(0)
n, p = 60, 8
X = rng.standard_normal((n, p))
X[:, 1] = 0.7 * X[:, 0] + 0.7 * X[:, 1]          # make features 1 and 2 correlated
X = (X - X.mean(0)) / X.std(0)                    # standardise: penalties need a common scale
beta = np.array([3.0, 0, -2.0, 0, 1.5, 0, 0, 0])
y = X @ beta + rng.standard_normal(n)
y = y - y.mean()                                  # centre y, so there is no intercept to fit

def ridge(X, y, lam):                             # via the SVD: shrink each direction by d^2/(d^2+lam)
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ (d / (d**2 + lam) * (U.T @ y))

def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def lasso(X, y, lam, iters=1000):                 # minimise (1/2n)||y - Xw||^2 + lam ||w||_1
    n = len(y)
    w = np.zeros(X.shape[1]); r = y.copy()
    sq = (X**2).sum(0) / n
    for _ in range(iters):
        for j in range(len(w)):
            r += X[:, j] * w[j]                   # remove feature j's contribution
            w[j] = soft(X[:, j] @ r / n, lam) / sq[j]
            r -= X[:, j] * w[j]
    return w

np.set_printoptions(precision=2, suppress=True)
print("least squares ", ridge(X, y, 0.0))
print("ridge  λ=30   ", ridge(X, y, 30.0))
print("lasso  λ=0.3  ", lasso(X, y, 0.3))
U, d, Vt = np.linalg.svd(X, full_matrices=False)
for lam in [0, 30, 300]:
    print(f"df(λ={lam}) =", round(np.sum(d**2 / (d**2 + lam)), 2))

# least squares  [ 3.17 -0.25 -1.98  0.2   1.7  -0.03  0.17 -0.18]
# ridge  λ=30    [ 1.82  0.37 -1.29  0.03  1.16  0.07  0.11 -0.22]
# lasso  λ=0.3   [ 2.67  0.   -1.72  0.    1.43 -0.    0.   -0.  ]
# df(λ=0) = 8.0
# df(λ=30) = 5.08
# df(λ=300) = 1.3
