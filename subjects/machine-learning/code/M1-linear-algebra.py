import numpy as np

rng = np.random.default_rng(0)

# 1. Power iteration: repeatedly apply A and renormalise -> top eigenvector of a symmetric matrix
A = rng.standard_normal((5, 5)); A = A @ A.T       # symmetric positive semidefinite
v = rng.standard_normal(5)
for _ in range(200):
    v = A @ v; v /= np.linalg.norm(v)
lam, Q = np.linalg.eigh(A)                         # eigenvalues in ascending order
print("power iteration eigenvalue", round(v @ A @ v, 6), "  eigh:", round(lam[-1], 6))
print("eigenvalue ratio lam2/lam1 =", round(lam[-2] / lam[-1], 3), "(sets the convergence speed)")

# 2. PCA two ways: eigenvectors of the covariance, or the SVD of the centred data
n = 500
Z = rng.standard_normal((n, 3)) * [3.0, 1.0, 0.2]  # variances 9, 1, 0.04 along hidden axes
R = np.linalg.qr(rng.standard_normal((3, 3)))[0]   # a random rotation
X = Z @ R.T + [5.0, -2.0, 1.0]
Xc = X - X.mean(0)
evals, evecs = np.linalg.eigh(Xc.T @ Xc / n)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print("covariance eigenvalues   ", np.round(evals[::-1], 3))
print("singular values^2 / n    ", np.round(s**2 / n, 3))
print("explained variance ratio ", np.round(s**2 / np.sum(s**2), 4))
print("same first direction (up to sign):", np.allclose(abs(Vt[0]), abs(evecs[:, -1])))

# 3. Eckart-Young: the best rank-k approximation keeps the top k singular values
M = rng.standard_normal((40, 30)) @ np.diag(0.8 ** np.arange(30)) @ rng.standard_normal((30, 30))
U, s, Vt = np.linalg.svd(M, full_matrices=False)
for k in [1, 5, 10]:
    Mk = U[:, :k] * s[:k] @ Vt[:k]
    print(f"rank {k:2d}: error {np.linalg.norm(M - Mk):8.4f}   sqrt(sum of dropped s^2) {np.sqrt(np.sum(s[k:]**2)):8.4f}")

# 4. The pseudo-inverse solves least squares, even when X^T X is singular
X2 = rng.standard_normal((6, 4)); X2[:, 3] = X2[:, 0] + X2[:, 1]   # rank 3
y2 = rng.standard_normal(6)
w = np.linalg.pinv(X2) @ y2
print("rank", np.linalg.matrix_rank(X2), "| normal equations hold:", np.allclose(X2.T @ X2 @ w, X2.T @ y2),
      "| min-norm solution has w orthogonal to null space:", np.isclose(w @ [1, 1, 0, -1], 0))

# power iteration eigenvalue 8.079323   eigh: 8.079323
# eigenvalue ratio lam2/lam1 = 0.819 (sets the convergence speed)
# covariance eigenvalues    [8.606 0.918 0.042]
# singular values^2 / n     [8.606 0.918 0.042]
# explained variance ratio  [0.8997 0.0959 0.0044]
# same first direction (up to sign): True
# rank  1: error  38.3931   sqrt(sum of dropped s^2)  38.3931
# rank  5: error  15.2187   sqrt(sum of dropped s^2)  15.2187
# rank 10: error   4.4866   sqrt(sum of dropped s^2)   4.4866
# rank 3 | normal equations hold: True | min-norm solution has w orthogonal to null space: True
