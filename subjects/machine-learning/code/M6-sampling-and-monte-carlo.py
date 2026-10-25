import math
import numpy as np

rng = np.random.default_rng(0)

# ---- Part 1: plain Monte Carlo. The error falls as 1/sqrt(n) whatever the dimension.
# The integrand is f(x) = prod_i (pi/2) sin(pi x_i) on [0, 1]^d, whose integral is exactly 1.
def f(x):
    return np.prod(np.pi / 2 * np.sin(np.pi * x), axis=1)

ns = [100, 1_000, 10_000, 100_000]
mc_err = {}
for d in (1, 10):
    mc_err[d] = []
    for n in ns:
        est = [f(rng.random((n, d))).mean() for _ in range(200)]
        mc_err[d].append(np.sqrt(np.mean((np.array(est) - 1) ** 2)))
    slope = np.polyfit(np.log10(ns), np.log10(mc_err[d]), 1)[0]
    sd = f(rng.random((1_000_000, d))).std()
    print(f"d = {d:2d}: RMS error at n = 100 ... 100,000: {', '.join(f'{e:.4f}' for e in mc_err[d])};"
          f" slope {slope:.3f}; sigma_f = {sd:.3f}, sigma_f / sqrt(100,000) = {sd / math.sqrt(1e5):.4f}")
# A grid with the same number of points: 100,000 points is 100,000 per axis in 1D, but under 4 per axis in 10D.
m = 100_000
grid1 = (np.arange(m) + 0.5) / m
print(f"midpoint rule with {m:,} points: d = 1 error {abs(f(grid1[:, None]).mean() - 1):.1e};"
      f" d = 10 allows {m ** 0.1:.2f} points per axis")
g3 = (np.arange(3) + 0.5) / 3
pts = np.array(np.meshgrid(*[g3] * 10)).reshape(10, -1).T
print(f"midpoint rule in d = 10 with 3 points per axis ({len(pts):,} points): error {abs(f(pts).mean() - 1):.3f}")

# ---- Part 2: importance sampling for a rare event, P(Z > 4) for Z ~ N(0, 1).
true_p = 0.5 * math.erfc(4 / math.sqrt(2))
n = 10_000
naive = [(rng.standard_normal(n) > 4).mean() for _ in range(200)]
def is_est(shift, n):
    y = rng.standard_normal(n) + shift                       # proposal N(shift, 1)
    w = np.exp(-shift * y + shift ** 2 / 2)                  # p(y) / q(y)
    return np.mean((y > 4) * w)
ests = {s: [is_est(s, n) for _ in range(200)] for s in (2.0, 4.0, 6.0)}
print(f"P(Z > 4) = {true_p:.4e}. With n = 10,000: naive estimates are 0 in {np.mean(np.array(naive) == 0):.0%} of 200 runs,"
      f" relative RMS error {np.sqrt(np.mean((np.array(naive) - true_p) ** 2)) / true_p:.2f}")
for s, e in ests.items():
    e = np.array(e)
    print(f"  importance sampling from N({s:.0f}, 1): relative RMS error {np.sqrt(np.mean((e - true_p) ** 2)) / true_p:.4f}")
for s in (2.0, 4.0, 6.0):                                     # effective sample size of the weights f(y) w(y)
    y = rng.standard_normal(n) + s; wy = np.exp(-s * y + s ** 2 / 2) * (y > 4)
    print(f"  N({s:.0f}, 1): fraction of samples in the event {np.mean(y > 4):.3f}, effective sample size {wy.sum() ** 2 / (wy ** 2).sum():,.0f} of {n:,}")

# ---- Part 3: variance reduction. E[e^U], U ~ Uniform(0, 1), = e - 1.
n, runs = 1_000, 2_000
U = rng.random((runs, n))
plain = np.exp(U).mean(1)
anti = (0.5 * (np.exp(U[:, : n // 2]) + np.exp(1 - U[:, : n // 2]))).mean(1)      # n/2 pairs = n evaluations
c = -np.cov(np.exp(U[0]), U[0])[0, 1] / np.var(U[0], ddof=1)                    # coefficient from a pilot run
cv = (np.exp(U) + c * (U - 0.5)).mean(1)
for name, e in [("plain", plain), ("antithetic", anti), ("control variate", cv)]:
    print(f"{name:15s}: standard deviation of the estimate {e.std():.2e} (variance ratio to plain {e.var() / plain.var():.4f})")
print(f"control variate coefficient from the pilot run {c:.4f}; the optimum -Cov(e^U, U) / Var(U) = {-12 * (1 - 0.5 * (math.e - 1)):.4f}")

# ---- Part 4: Metropolis-Hastings on a two-mode target, p(x) proportional to
# exp(-(x-3)^2/2) + exp(-(x+3)^2/2). Random-walk proposals of three widths.
def log_p(x):
    return np.logaddexp(-(x - 3) ** 2 / 2, -(x + 3) ** 2 / 2)

def mh(width, steps, x0=0.0):
    x, lp, out, acc = x0, log_p(x0), np.empty(steps), 0
    z = rng.standard_normal(steps); u = rng.random(steps)
    for t in range(steps):
        y = x + width * z[t]
        lq = log_p(y)
        if math.log(u[t]) < lq - lp:                         # symmetric proposal: accept with min(1, p(y)/p(x))
            x, lp, acc = y, lq, acc + 1
        out[t] = x
    return out, acc / steps

def ess(chain, max_lag=2000):
    c = chain - chain.mean(); var = c.var()
    rho_sum = 0.0
    for k in range(1, max_lag):
        rho = np.dot(c[:-k], c[k:]) / (len(c) * var)
        if rho < 0.05:
            break
        rho_sum += rho
    return len(chain) / (1 + 2 * rho_sum)

chains = {}
for width in (0.1, 2.5, 50.0):
    ch, acc = mh(width, 100_000)
    chains[width] = ch
    print(f"MH, proposal width {width:4}: acceptance {acc:.2f}; fraction of samples with x > 0 {np.mean(ch > 0):.3f} (true 0.5);"
          f" E[x^2] {np.mean(ch ** 2):.2f} (true 10.00); effective sample size {ess(ch):,.0f} of 100,000")

# ---- Part 5: two gradient estimators for d/dmu E[x^2], x ~ N(mu, sigma^2); the true value is 2 mu.
mu, sigma, n = 1.0, 0.5, 100_000
z = rng.standard_normal(n); x = mu + sigma * z
score = x ** 2 * (x - mu) / sigma ** 2                          # score function: f(x) d/dmu log p(x)
score_b = (x ** 2 - (mu ** 2 + sigma ** 2)) * (x - mu) / sigma ** 2   # with the baseline E[f]
reparam = 2 * x                                                # reparameterisation: d/dmu f(mu + sigma z)
for name, g in [("score function", score), ("score function with baseline", score_b), ("reparameterisation", reparam)]:
    print(f"{name:29s}: mean {g.mean():.4f} (true {2 * mu:.1f}), standard deviation of one sample {g.std():.3f}")

# d =  1: RMS error at n = 100 ... 100,000: 0.0500, 0.0167, 0.0048, 0.0017; slope -0.495; sigma_f = 0.483, sigma_f / sqrt(100,000) = 0.0015
# d = 10: RMS error at n = 100 ... 100,000: 0.2670, 0.0830, 0.0256, 0.0083; slope -0.504; sigma_f = 2.669, sigma_f / sqrt(100,000) = 0.0084
# midpoint rule with 100,000 points: d = 1 error 4.1e-11; d = 10 allows 3.16 points per axis
# midpoint rule in d = 10 with 3 points per axis (59,049 points): error 0.586
# P(Z > 4) = 3.1671e-05. With n = 10,000: naive estimates are 0 in 71% of 200 runs, relative RMS error 1.59
#   importance sampling from N(2, 1): relative RMS error 0.0732
#   importance sampling from N(4, 1): relative RMS error 0.0220
#   importance sampling from N(6, 1): relative RMS error 0.0614
#   N(2, 1): fraction of samples in the event 0.022, effective sample size 181 of 10,000
#   N(4, 1): fraction of samples in the event 0.506, effective sample size 1,831 of 10,000
#   N(6, 1): fraction of samples in the event 0.976, effective sample size 296 of 10,000
# plain          : standard deviation of the estimate 1.52e-02 (variance ratio to plain 1.0000)
# antithetic     : standard deviation of the estimate 2.83e-03 (variance ratio to plain 0.0349)
# control variate: standard deviation of the estimate 1.97e-03 (variance ratio to plain 0.0169)
# control variate coefficient from the pilot run -1.6884; the optimum -Cov(e^U, U) / Var(U) = -1.6903
# MH, proposal width  0.1: acceptance 0.97; fraction of samples with x > 0 0.195 (true 0.5); E[x^2] 10.01 (true 10.00); effective sample size 30 of 100,000
# MH, proposal width  2.5: acceptance 0.49; fraction of samples with x > 0 0.511 (true 0.5); E[x^2] 10.00 (true 10.00); effective sample size 4,786 of 100,000
# MH, proposal width 50.0: acceptance 0.05; fraction of samples with x > 0 0.493 (true 0.5); E[x^2] 10.00 (true 10.00); effective sample size 2,708 of 100,000
# score function               : mean 2.0217 (true 2.0), standard deviation of one sample 4.704
# score function with baseline : mean 2.0174 (true 2.0), standard deviation of one sample 3.273
# reparameterisation           : mean 2.0017 (true 2.0), standard deviation of one sample 1.003
