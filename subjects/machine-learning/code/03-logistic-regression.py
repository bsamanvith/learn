import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0])   # hours studied
y = np.array([0.0, 0.0, 1.0, 1.0])   # passed?
X = np.column_stack([x, np.ones_like(x)])

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def log_loss(w):
  p = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)   # avoid log(0)
  return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def grad(w):
  return X.T @ (sigmoid(X @ w) - y) / len(y)      # (1/n) X^T (p - y)

w = np.zeros(2)
for step in range(5000):
  w -= 0.5 * grad(w)
print(w, log_loss(w))   # w keeps growing: this data is separable
