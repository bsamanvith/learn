import numpy as np

rng = np.random.default_rng(0)

# 1. A three-state weather chain. Rows of P are "from", columns "to"; each row sums to 1.
states = ["sunny", "cloudy", "rainy"]
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.3, 0.5]])
w, U = np.linalg.eig(P.T)                                  # pi P = pi  <=>  P^T pi^T = pi^T
pi = np.real(U[:, np.argmin(np.abs(w - 1))]); pi /= pi.sum()
print("stationary distribution:", ", ".join(f"{s} {v:.4f}" for s, v in zip(states, pi)))
mu = np.array([0.0, 0.0, 1.0])                             # start: rainy for sure
for n in [1, 2, 5, 10, 20]:
    print(f"  after {n:2d} days from 'rainy': {np.round(mu @ np.linalg.matrix_power(P, n), 4)}")
x, counts = 2, np.zeros(3)
for _ in range(100_000):                                   # the ergodic theorem: time averages
    x = rng.choice(3, p=P[x]); counts[x] += 1
print("fraction of 100,000 simulated days:", np.round(counts / counts.sum(), 4))
print("eigenvalues of P:", np.round(np.sort(np.abs(np.linalg.eigvals(P)))[::-1], 4))

# 2. Mixing on a cycle of 20 states. The plain walk (step left or right) is periodic: on an even
# cycle it alternates between even and odd states forever. The lazy walk (stay with prob. 1/2)
# converges, at a rate set by its second-largest eigenvalue.
m = 20
walk = np.zeros((m, m))
for i in range(m):
    walk[i, (i + 1) % m] = walk[i, (i - 1) % m] = 0.5
lazy = 0.5 * np.eye(m) + 0.5 * walk
uniform = np.full(m, 1 / m)
lam2 = np.sort(np.abs(np.linalg.eigvals(lazy)))[-2]
tv = lambda p, q: 0.5 * np.abs(p - q).sum()
for name, Q in [("plain", walk), ("lazy", lazy)]:
    d = [tv(np.linalg.matrix_power(Q, n)[0], uniform) for n in [10, 50, 100, 200, 400]]
    print(f"{name:5s} walk, TV distance to uniform after 10/50/100/200/400 steps:", " ".join(f"{v:.2e}" for v in d))
print(f"lazy walk: second eigenvalue {lam2:.5f} (= (1 + cos(2 pi / 20)) / 2 = {(1 + np.cos(2 * np.pi / m)) / 2:.5f});"
      f" the distance shrinks by about this factor per step")

# 3. Absorbing chains: gambler's ruin. Start with k of N = 10 coins; win a coin with p = 0.45.
N, p = 10, 0.45
Q = np.zeros((N - 1, N - 1)); Rb = np.zeros((N - 1, 2))   # transient states 1..9; absorbing 0 and N
for k in range(1, N):
    if k + 1 < N: Q[k - 1, k] = p
    else: Rb[k - 1, 1] = p
    if k - 1 > 0: Q[k - 1, k - 2] = 1 - p
    else: Rb[k - 1, 0] = 1 - p
Nf = np.linalg.inv(np.eye(N - 1) - Q)                       # fundamental matrix: expected visits
win = (Nf @ Rb)[:, 1]; duration = Nf.sum(1)
r = (1 - p) / p
print(f"start with 5: P(reach {N}) = {win[4]:.4f} (formula (1 - r^5) / (1 - r^10) = {(1 - r ** 5) / (1 - r ** N):.4f}),"
      f" expected number of bets {duration[4]:.2f}")
sims = []
for _ in range(20000):
    k, t = 5, 0
    while 0 < k < N:
        k += 1 if rng.random() < p else -1; t += 1
    sims.append((k == N, t))
sims = np.array(sims)
print(f"simulated 20,000 games: P(reach {N}) = {sims[:, 0].mean():.4f}, mean number of bets {sims[:, 1].mean():.2f}")

# 4. PageRank: a random surfer follows a random link with probability 0.85, otherwise jumps to a
# random page. The jump makes the chain irreducible and aperiodic, so pi exists and is unique.
links = {0: [1, 2], 1: [2], 2: [0], 3: [2], 4: [2, 3], 5: [4, 6], 6: [4, 5, 7], 7: []}  # 7 has no links
n_pages = 8
G = np.zeros((n_pages, n_pages))
for i, out in links.items():
    G[i, out if out else range(n_pages)] = 1 / (len(out) if out else n_pages)  # a dead end jumps anywhere
M = 0.85 * G + 0.15 / n_pages
rank = np.full(n_pages, 1 / n_pages)
for it in range(100):
    new = rank @ M
    if np.abs(new - rank).sum() < 1e-12:
        break
    rank = new
print(f"PageRank after {it} power iterations:", np.round(rank, 4))

# 5. Markov chain Monte Carlo. Metropolis-Hastings builds a chain whose stationary distribution is
# a target known only up to a constant: here a mixture of two Gaussians at -3 and +3.
log_target = lambda x: np.logaddexp(-0.5 * (x + 3) ** 2, -0.5 * (x - 3) ** 2)
def metropolis(step, n=50_000, x=0.0, seed=1):
    r = np.random.default_rng(seed)
    xs, acc = np.empty(n), 0
    for i in range(n):
        y = x + step * r.standard_normal()                  # symmetric proposal
        if np.log(r.random()) < log_target(y) - log_target(x):
            x, acc = y, acc + 1
        xs[i] = x
    return xs, acc / n
def ess(xs):
    """Effective sample size: n / (1 + 2 * sum of autocorrelations), summed until they turn negative."""
    z = xs - xs.mean(); n = len(z)
    f = np.fft.rfft(z, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n] / (z @ z)
    s = 0.0
    for k in range(1, n):
        if ac[k] < 0:
            break
        s += ac[k]
    return n / (1 + 2 * s)
for step in [0.1, 1.0, 4.0, 30.0]:
    xs, a = metropolis(step)
    print(f"step {step:5.1f}: acceptance {a:.3f}, fraction of time at x > 0 {np.mean(xs > 0):.3f},"
          f" mean {xs.mean():+.3f}, effective sample size {ess(xs):8.1f} of 50,000")

# stationary distribution: sunny 0.4565, cloudy 0.2826, rainy 0.2609
#   after  1 days from 'rainy': [0.2 0.3 0.5]
#   after  2 days from 'rainy': [0.33 0.31 0.36]
#   after  5 days from 'rainy': [0.4429 0.2862 0.2708]
#   after 10 days from 'rainy': [0.4562 0.2827 0.2611]
#   after 20 days from 'rainy': [0.4565 0.2826 0.2609]
# fraction of 100,000 simulated days: [0.4594 0.2812 0.2594]
# eigenvalues of P: [1.     0.4732 0.1268]
# plain walk, TV distance to uniform after 10/50/100/200/400 steps: 6.41e-01 5.00e-01 5.00e-01 5.00e-01 5.00e-01
# lazy  walk, TV distance to uniform after 10/50/100/200/400 steps: 5.35e-01 1.84e-01 5.30e-02 4.45e-03 3.13e-05
# lazy walk: second eigenvalue 0.97553 (= (1 + cos(2 pi / 20)) / 2 = 0.97553); the distance shrinks by about this factor per step
# start with 5: P(reach 10) = 0.2683 (formula (1 - r^5) / (1 - r^10) = 0.2683), expected number of bets 23.17
# simulated 20,000 games: P(reach 10) = 0.2671, mean number of bets 23.40
# PageRank after 52 power iterations: [0.3134 0.1554 0.3426 0.0418 0.0461 0.0324 0.0359 0.0324]
# step   0.1: acceptance 0.968, fraction of time at x > 0 0.895, mean +2.296, effective sample size     11.6 of 50,000
# step   1.0: acceptance 0.716, fraction of time at x > 0 0.484, mean -0.091, effective sample size    181.8 of 50,000
# step   4.0: acceptance 0.407, fraction of time at x > 0 0.505, mean +0.026, effective sample size   5201.8 of 50,000
# step  30.0: acceptance 0.083, fraction of time at x > 0 0.502, mean -0.002, effective sample size   2052.7 of 50,000
