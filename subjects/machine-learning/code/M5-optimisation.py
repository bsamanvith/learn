import numpy as np

rng = np.random.default_rng(0)

# 1. Gradient descent on a mu-strongly convex, L-smooth quadratic meets the (1 - mu/L)^t bound
A = rng.standard_normal((30, 10)); H = A.T @ A / 30 + 0.1 * np.eye(10)
b = rng.standard_normal(10)
f = lambda x: 0.5 * x @ H @ x - b @ x
x_star = np.linalg.solve(H, b)
mu, L = np.linalg.eigvalsh(H)[[0, -1]]
x = np.zeros(10); gap0 = f(x) - f(x_star)
for t in range(1, 101):
    x = x - (1 / L) * (H @ x - b)
    if t in (10, 50, 100):
        print(f"t={t:3d}: f - f* = {f(x) - f(x_star):.3e}   bound (1 - mu/L)^t (f0 - f*) = {(1 - mu / L) ** t * gap0:.3e}")

# 2. Lagrange multipliers: the maximum-entropy die with mean 4.5 (Jaynes' dice problem)
faces = np.arange(1, 7)
def gibbs(beta):                                   # stationarity of the Lagrangian gives p_k ∝ exp(beta * k)
    w = np.exp(beta * faces); return w / w.sum()
lo, hi = -5.0, 5.0                                 # solve mean(beta) = 4.5 by bisection (mean is increasing in beta)
for _ in range(100):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if gibbs(mid) @ faces < 4.5 else (lo, mid)
p = gibbs(lo)
H_p = -np.sum(p * np.log(p))
print("max-entropy die:", np.round(p, 4), " mean", round(p @ faces, 4), " beta", round(lo, 4), " entropy", round(H_p, 4))
for _ in range(3):                                 # any other distribution with mean 4.5 has lower entropy
    q = p + 0.02 * rng.standard_normal(6)
    # project q back onto {sum = 1, mean = 4.5}: subtract the component in span{1, faces}
    C = np.stack([np.ones(6), faces]); q = q - C.T @ np.linalg.solve(C @ C.T, C @ q - [1, 4.5])
    print("  perturbed: mean", round(q @ faces, 4), " entropy", round(-np.sum(q * np.log(q)), 4))

# 3. KKT: minimise ||x - c||^2 subject to x >= 0, by projected gradient descent
c = np.array([1.5, -0.7, 0.3, -2.0])
x = np.zeros(4)
for _ in range(200):
    x = np.maximum(x - 0.1 * 2 * (x - c), 0)      # gradient step, then project onto x >= 0
lam = 2 * (x - c)                                  # multipliers from stationarity: 2(x - c) - lam = 0
clean = lambda v: np.round(v, 6) + 0.0             # tidy -0.0 and 1e-16 noise for printing
print("x* =", clean(x), "  multipliers =", clean(lam))
print("complementary slackness lam * x =", clean(lam * x), "  all lam >= 0:", np.all(lam > -1e-9))

# t= 10: f - f* = 4.374e-01   bound (1 - mu/L)^t (f0 - f*) = 3.192e+00
# t= 50: f - f* = 3.144e-06   bound (1 - mu/L)^t (f0 - f*) = 8.763e-03
# t=100: f - f* = 1.249e-12   bound (1 - mu/L)^t (f0 - f*) = 5.508e-06
# max-entropy die: [0.0544 0.0788 0.1142 0.1654 0.2398 0.3475]  mean 4.5  beta 0.371  entropy 1.6136
#   perturbed: mean 4.5  entropy 1.6083
#   perturbed: mean 4.5  entropy 1.6115
#   perturbed: mean 4.5  entropy 1.6078
# x* = [1.5 0.  0.3 0. ]   multipliers = [0.  1.4 0.  4. ]
# complementary slackness lam * x = [0. 0. 0. 0.]   all lam >= 0: True
