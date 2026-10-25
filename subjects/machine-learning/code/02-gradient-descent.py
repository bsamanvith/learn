import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 5.0])

def loss(w, b):
  return np.mean((w * x + b - y) ** 2)

def grad(w, b):
  r = w * x + b - y                    # residuals: guess minus truth
  return 2 * np.mean(r * x), 2 * np.mean(r)

# The largest stable learning rate is 2 / (largest eigenvalue of the Hessian).
H = 2 * np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
print("max stable eta:", 2 / np.linalg.eigvalsh(H).max())   # ~0.18

w, b, eta = 0.0, 0.0, 0.1
for step in range(200):
  gw, gb = grad(w, b)
  w, b = w - eta * gw, b - eta * gb    # the update
print(w, b, loss(w, b))                  # 1.5 0.667 0.0556
