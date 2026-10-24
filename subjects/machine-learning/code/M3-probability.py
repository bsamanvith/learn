import numpy as np

rng = np.random.default_rng(0)

# 1. The MLE of a Gaussian's variance is biased by a factor (n-1)/n
n, sigma2 = 5, 4.0
samples = rng.normal(0.0, np.sqrt(sigma2), size=(200_000, n))
mle_var = samples.var(axis=1)                      # divides by n
print(f"E[MLE variance] ~ {mle_var.mean():.3f}   theory (n-1)/n * sigma^2 = {(n - 1) / n * sigma2:.3f}")
print(f"E[unbiased var] ~ {samples.var(axis=1, ddof=1).mean():.3f}   theory sigma^2 = {sigma2:.3f}")

# 2. Coin flips: MLE vs MAP vs posterior mean with a Beta(2, 2) prior
heads, flips, a, b = 3, 3, 2, 2
print(f"3 heads in 3 flips: MLE {heads / flips:.3f}   MAP {(heads + a - 1) / (flips + a + b - 2):.3f}"
      f"   posterior mean {(heads + a) / (flips + a + b):.3f}")

# 3. Central limit theorem: means of n exponential(1) draws (skewness 2) become Gaussian
for n in [1, 5, 30]:
    m = rng.exponential(1.0, size=(200_000, n)).mean(axis=1)
    z = (m - m.mean()) / m.std()
    print(f"n={n:2d}: std of mean {m.std():.4f} (theory {1 / np.sqrt(n):.4f})   skewness {np.mean(z**3):.3f} (theory {2 / np.sqrt(n):.3f})")

# 4. Conditioning a bivariate Gaussian: x2 | x1 is Gaussian with a linear mean
mu = np.array([1.0, -1.0]); S = np.array([[2.0, 1.2], [1.2, 1.0]])
X = rng.multivariate_normal(mu, S, size=2_000_000)
near = np.abs(X[:, 0] - 2.0) < 0.01                # condition on x1 close to 2
cond_mean = mu[1] + S[1, 0] / S[0, 0] * (2.0 - mu[0])
cond_var = S[1, 1] - S[1, 0] ** 2 / S[0, 0]
print(f"x2 | x1=2: sampled mean {X[near, 1].mean():.3f} var {X[near, 1].var():.3f}"
      f"   formula mean {cond_mean:.3f} var {cond_var:.3f}")

# 5. Law of total variance: Var(Y) = E[Var(Y|Z)] + Var(E[Y|Z]) for a mixture
Z = rng.integers(0, 2, 1_000_000)                  # which component
Y = np.where(Z == 0, rng.normal(0, 1, Z.size), rng.normal(4, 2, Z.size))
print(f"Var(Y) sampled {Y.var():.3f}   E[Var(Y|Z)] + Var(E[Y|Z]) = {0.5 * 1 + 0.5 * 4 + 4.0:.3f}")

# E[MLE variance] ~ 3.207   theory (n-1)/n * sigma^2 = 3.200
# E[unbiased var] ~ 4.009   theory sigma^2 = 4.000
# 3 heads in 3 flips: MLE 1.000   MAP 0.800   posterior mean 0.714
# n= 1: std of mean 1.0014 (theory 1.0000)   skewness 2.001 (theory 2.000)
# n= 5: std of mean 0.4464 (theory 0.4472)   skewness 0.900 (theory 0.894)
# n=30: std of mean 0.1829 (theory 0.1826)   skewness 0.366 (theory 0.365)
# x2 | x1=2: sampled mean -0.397 var 0.285   formula mean -0.400 var 0.280
# Var(Y) sampled 6.501   E[Var(Y|Z)] + Var(E[Y|Z]) = 6.500
