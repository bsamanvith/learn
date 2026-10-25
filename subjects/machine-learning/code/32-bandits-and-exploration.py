import numpy as np

# A 10-armed Bernoulli bandit: arm a pays 1 with probability mu[a]. The best arm pays 0.9;
# the others are spread below it. Every algorithm is run 200 times for T = 50,000 pulls,
# vectorised across the runs.
mu = np.array([0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2])
K, T, runs = len(mu), 50_000, 200
gaps = mu.max() - mu
rows = np.arange(runs)

def run(choose, update=None, seed=0):
    """choose(t, counts, sums, rng) -> arms (runs,). Returns the regret curve averaged over runs
    and the mean number of pulls of each arm."""
    rng = np.random.default_rng(seed)
    counts, sums = np.zeros((runs, K)), np.zeros((runs, K))
    regret = np.zeros(T)
    for t in range(T):
        a = choose(t, counts, sums, rng)
        r = (rng.random(runs) < mu[a]).astype(float)
        counts[rows, a] += 1; sums[rows, a] += r
        if update:
            update(a, r)
        regret[t] = gaps[a].mean()
    return np.cumsum(regret), counts.mean(0)

def means(counts, sums):
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)   # untried arms first

def greedy(t, counts, sums, rng):
    m = means(counts, sums)
    return np.argmax(m + 1e-9 * rng.random(m.shape), 1)                  # random tie-breaking

def eps_greedy(eps):
    def choose(t, counts, sums, rng):
        explore = rng.random(runs) < eps
        return np.where(explore, rng.integers(0, K, runs), greedy(t, counts, sums, rng))
    return choose

def ucb1(t, counts, sums, rng):
    bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(counts, 1))
    return np.argmax(np.where(counts > 0, sums / np.maximum(counts, 1) + bonus, np.inf), 1)

def thompson(t, counts, sums, rng):
    return np.argmax(rng.beta(1 + sums, 1 + counts - sums), 1)            # one posterior sample per arm

H = np.zeros((runs, K)); baseline = np.zeros(runs)
def gradient_choose(t, counts, sums, rng):
    p = np.exp(H - H.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    gradient_choose.p = p
    return (p.cumsum(1) > rng.random(runs)[:, None]).argmax(1)
def gradient_update(a, r, alpha=0.1):
    global baseline
    p = gradient_choose.p
    onehot = np.zeros_like(p); onehot[rows, a] = 1
    H[:] += alpha * (r - baseline)[:, None] * (onehot - p)              # stochastic gradient of E[r]
    baseline += (r - baseline) / 100                                     # running average of reward

results = {}
for name, args in [("greedy", (greedy,)), ("epsilon-greedy 0.1", (eps_greedy(0.1),)),
                   ("epsilon-greedy 0.01", (eps_greedy(0.01),)), ("UCB1", (ucb1,)),
                   ("Thompson sampling", (thompson,)), ("gradient bandit", (gradient_choose, gradient_update))]:
    curve, pulls = run(*args)
    results[name] = curve
    print(f"{name:19s} regret after 1,000 / 10,000 / 50,000 pulls: {curve[999]:6.1f} {curve[9999]:7.1f} {curve[-1]:7.1f};"
          f" best arm pulled {pulls[0] / T:.3f} of the time")

# The Lai-Robbins lower bound: any consistent algorithm pulls each suboptimal arm a at least
# log(T) / KL(mu_a, mu*) times asymptotically, so regret >= sum_a gap_a / KL(mu_a, mu*) log T.
kl = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
c_lr = sum(gaps[a] / kl(mu[a], mu[0]) for a in range(1, K))
c_ucb = sum(8 / gaps[a] for a in range(1, K))
print(f"Lai-Robbins: regret grows at least as {c_lr:.2f} log T, which is {c_lr * np.log(T):.1f} at T = 50,000;"
      f" UCB1's guarantee, sum 8 / gap * log T, = {c_ucb * np.log(T):.0f}")

# greedy              regret after 1,000 / 10,000 / 50,000 pulls:   48.3   410.7  2010.7; best arm pulled 0.639 of the time
# epsilon-greedy 0.1  regret after 1,000 / 10,000 / 50,000 pulls:   57.0   368.8  1695.8; best arm pulled 0.903 of the time
# epsilon-greedy 0.01 regret after 1,000 / 10,000 / 50,000 pulls:   48.4   260.4   438.1; best arm pulled 0.939 of the time
# UCB1                regret after 1,000 / 10,000 / 50,000 pulls:  139.6   422.6   638.2; best arm pulled 0.921 of the time
# Thompson sampling   regret after 1,000 / 10,000 / 50,000 pulls:   37.3    55.2    68.1; best arm pulled 0.992 of the time
# gradient bandit     regret after 1,000 / 10,000 / 50,000 pulls:  133.3   183.6   199.3; best arm pulled 0.982 of the time
# Lai-Robbins: regret grows at least as 10.10 log T, which is 109.2 at T = 50,000; UCB1's guarantee, sum 8 / gap * log T, = 3326
