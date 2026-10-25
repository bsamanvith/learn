import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 5.0])

# One input: the covariance formula.
w = np.mean((x - x.mean()) * (y - y.mean())) / np.mean((x - x.mean()) ** 2)
b = y.mean() - w * x.mean()
print(w, b)                      # 1.5 0.6667

# Any number of inputs: the normal equations, bias absorbed as a column of ones.
X = np.column_stack([x, np.ones_like(x)])
w_vec = np.linalg.solve(X.T @ X, X.T @ y)   # never form the inverse explicitly
print(w_vec)                     # [1.5 0.6667]

r = X @ w_vec - y
print(r.sum(), (r * x).sum())    # both ~0: residuals are orthogonal to the columns
print(1 - np.mean(r**2) / np.var(y))         # R^2 = 0.964
