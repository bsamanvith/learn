import numpy as np

def walk_episode(n, rng):
    """A random walk on states 1..n, started in the middle, stepping left or right with equal
    probability until it leaves at 0 or n + 1. Returns the visited states and the rewards."""
    s, states, rewards = (n + 1) // 2, [], []
    while 1 <= s <= n:
        states.append(s)
        s += rng.choice((-1, 1))
        rewards.append(0.0)
    return states, rewards, s

# ---- Part 1: the 5-state random walk. Reward 1 for leaving on the right, 0 otherwise, gamma = 1.
# The true value of state k is k / 6: the probability of leaving on the right.
n5 = 5
true5 = np.arange(1, n5 + 1) / (n5 + 1)
rms = lambda V, true: np.sqrt(np.mean((V - true) ** 2))

def run5(method, alpha, episodes=100, runs=100):
    """Average RMS error after each episode, over runs. V starts at 0.5 everywhere."""
    err = np.zeros(episodes)
    for r in range(runs):
        rng = np.random.default_rng(r)
        V = np.full(n5 + 2, 0.5); V[0] = V[-1] = 0.0                  # terminal values are 0
        for e in range(episodes):
            states, _, end = walk_episode(n5, rng)
            G = 1.0 if end == n5 + 1 else 0.0
            if method == "MC":                                       # every state's return is G
                for s in states:
                    V[s] += alpha * (G - V[s])
            else:                                                    # TD(0): bootstrap from V(s')
                for s, s2 in zip(states, states[1:] + [end]):
                    r_ = 1.0 if s2 == n5 + 1 else 0.0
                    V[s] += alpha * (r_ + V[s2] - V[s])
            err[e] += rms(V[1:-1], true5)
    return err / runs

curves5 = {}
for method, alphas in [("TD", [0.05, 0.1, 0.15]), ("MC", [0.01, 0.02, 0.03, 0.04])]:
    for a in alphas:
        curves5[(method, a)] = run5(method, a)
        c = curves5[(method, a)]
        print(f"5-state walk, {method} alpha = {a:.2f}: RMS error after 10, 50, 100 episodes:"
              f" {c[9]:.3f} {c[49]:.3f} {c[99]:.3f}")

# ---- Part 2: batch updating. Sutton and Barto's example 6.4: eight episodes,
# A,0,B,0 once; B,1 six times; B,0 once. Replay them until the estimates stop changing.
episodes = [[("A", 0, "B"), ("B", 0, None)]] + [[("B", 1, None)]] * 6 + [[("B", 0, None)]]
def batch(method, alpha=0.01, sweeps=20_000):
    V = {"A": 0.0, "B": 0.0, None: 0.0}
    for _ in range(sweeps):
        inc = {"A": 0.0, "B": 0.0}
        for ep in episodes:
            G = 0.0
            for s, r, s2 in reversed(ep):
                G += r
                target = G if method == "MC" else r + V[s2]
                inc[s] += target - V[s]
        for s in inc:                                                # one step per pass over the data
            V[s] += alpha * inc[s]
    return V["A"], V["B"]
print("batch MC: V(A) = %.4f, V(B) = %.4f;  batch TD: V(A) = %.4f, V(B) = %.4f" % (*batch("MC"), *batch("TD")))

# ---- Part 3: batch TD is the value of the maximum-likelihood model ("certainty equivalence").
rng = np.random.default_rng(7)
data = [walk_episode(n5, rng) for _ in range(5)]
counts = np.zeros((n5 + 2, n5 + 2)); rew = np.zeros(n5 + 2)
for states, _, end in data:
    for s, s2 in zip(states, states[1:] + [end]):
        counts[s, s2] += 1; rew[s] += 1.0 if s2 == n5 + 1 else 0.0
visited = counts.sum(1) > 0
P_hat = np.where(visited[:, None], counts / np.maximum(counts.sum(1, keepdims=True), 1), 0)
r_hat = np.where(visited, rew / np.maximum(counts.sum(1), 1), 0)
idx = np.flatnonzero(visited)
V_ce = np.zeros(n5 + 2)
V_ce[idx] = np.linalg.solve(np.eye(len(idx)) - P_hat[np.ix_(idx, idx)], r_hat[idx])
V = np.zeros(n5 + 2)
for _ in range(20_000):
    inc = np.zeros(n5 + 2)
    for states, _, end in data:
        for s, s2 in zip(states, states[1:] + [end]):
            inc[s] += (1.0 if s2 == n5 + 1 else 0.0) + V[s2] - V[s]
    V += 0.01 * inc
G_mc = np.zeros(n5 + 2); N_mc = np.zeros(n5 + 2)
for states, _, end in data:
    for s in states:
        G_mc[s] += 1.0 if end == n5 + 1 else 0.0; N_mc[s] += 1
print("5 episodes; visits per state:", N_mc[1:-1].astype(int).tolist())
print("batch TD:              ", np.round(V[1:-1], 4))
print("max-likelihood model:  ", np.round(V_ce[1:-1], 4))
print("batch MC (mean return):", np.round(G_mc[1:-1] / np.maximum(N_mc[1:-1], 1), 4))
print(f"RMS error: batch TD {rms(V[1:-1], true5):.3f}, batch MC {rms(G_mc[1:-1] / np.maximum(N_mc[1:-1], 1), true5):.3f}")

# ---- Part 4: n-step TD on the 19-state random walk (rewards -1 on the left, +1 on the right).
# Error averaged over the first 10 episodes and over states, for a grid of step sizes.
n19 = 19
true19 = np.arange(-(n19 - 1), n19, 2) / (n19 + 1)                 # -0.9, -0.8, ..., 0.9
alphas = np.linspace(0, 1, 21)[1:]
ns_ = [1, 2, 4, 8, 16, 32, 512]
nstep_err = {}
runs = 100
eps_all = []
for r in range(runs):                                               # the same episodes for every n and alpha
    rng = np.random.default_rng(1000 + r)
    eps_all.append([walk_episode(n19, rng) for _ in range(10)])
for nn in ns_:
    total = np.zeros(len(alphas))
    for r in range(runs):
        V = np.zeros((len(alphas), n19 + 2))                        # one value table per alpha
        for states, _, end in eps_all[r]:
            path = states + [end]
            Rw = [0.0] * (len(states) - 1) + [1.0 if end == n19 + 1 else -1.0]
            T = len(states)
            for t in range(T):
                h = min(t + nn, T)                                  # the n-step return from t
                G = sum(Rw[t:h])                                    # gamma = 1
                if h < T:
                    G = G + V[:, path[h]]
                s = path[t]
                V[:, s] += alphas * (G - V[:, s])
            total += np.sqrt(np.mean((V[:, 1:-1] - true19) ** 2, axis=1))
    nstep_err[nn] = total / (runs * 10)
    b = np.argmin(nstep_err[nn])
    print(f"19-state walk, n = {nn:3d}: best alpha {alphas[b]:.2f}, average RMS error over 10 episodes {nstep_err[nn][b]:.3f}")

# ---- Part 5: TD(lambda) with accumulating eligibility traces, online, on the same episodes.
lam_err = {}
np.seterr(over="ignore", invalid="ignore")                        # large alpha with long traces diverges
for lam in [0.0, 0.4, 0.8, 0.9, 0.95, 1.0]:
    total = np.zeros(len(alphas))
    for r in range(runs):
        V = np.zeros((len(alphas), n19 + 2))
        for states, _, end in eps_all[r]:
            z = np.zeros(n19 + 2)                                   # the eligibility trace
            for s, s2 in zip(states, states[1:] + [end]):
                rw = 0.0 if 1 <= s2 <= n19 else (1.0 if s2 == n19 + 1 else -1.0)
                delta = rw + V[:, s2] - V[:, s]                     # TD error, one per alpha
                z *= lam; z[s] += 1                                 # decay all traces, bump this state
                V += (alphas * delta)[:, None] * z[None, :]
            total += np.sqrt(np.mean((V[:, 1:-1] - true19) ** 2, axis=1))
    err = total / (runs * 10)
    ok = np.isfinite(err) & (err < 10)
    lam_err[lam] = np.where(ok, err, np.nan)
    b = np.nanargmin(lam_err[lam])
    print(f"19-state walk, TD(lambda = {lam:.2f}): best alpha {alphas[b]:.2f}, average RMS error {lam_err[lam][b]:.3f};"
          f" diverges for alpha >= {alphas[~ok][0]:.2f}" if (~ok).any() else
          f"19-state walk, TD(lambda = {lam:.2f}): best alpha {alphas[b]:.2f}, average RMS error {lam_err[lam][b]:.3f}")

# 5-state walk, TD alpha = 0.05: RMS error after 10, 50, 100 episodes: 0.175 0.056 0.036
# 5-state walk, TD alpha = 0.10: RMS error after 10, 50, 100 episodes: 0.129 0.056 0.054
# 5-state walk, TD alpha = 0.15: RMS error after 10, 50, 100 episodes: 0.096 0.075 0.068
# 5-state walk, MC alpha = 0.01: RMS error after 10, 50, 100 episodes: 0.213 0.143 0.092
# 5-state walk, MC alpha = 0.02: RMS error after 10, 50, 100 episodes: 0.199 0.108 0.076
# 5-state walk, MC alpha = 0.03: RMS error after 10, 50, 100 episodes: 0.189 0.101 0.089
# 5-state walk, MC alpha = 0.04: RMS error after 10, 50, 100 episodes: 0.183 0.106 0.103
# batch MC: V(A) = 0.0000, V(B) = 0.7500;  batch TD: V(A) = 0.7500, V(B) = 0.7500
# 5 episodes; visits per state: [2, 4, 13, 10, 4]
# batch TD:               [0.     0.3    0.6    0.7333 0.9333]
# max-likelihood model:   [0.     0.3    0.6    0.7333 0.9333]
# batch MC (mean return): [0.     0.25   0.4615 0.5    0.75  ]
# RMS error: batch TD 0.103, batch MC 0.119
# 19-state walk, n =   1: best alpha 0.80, average RMS error over 10 episodes 0.342
# 19-state walk, n =   2: best alpha 0.60, average RMS error over 10 episodes 0.270
# 19-state walk, n =   4: best alpha 0.40, average RMS error over 10 episodes 0.256
# 19-state walk, n =   8: best alpha 0.25, average RMS error over 10 episodes 0.267
# 19-state walk, n =  16: best alpha 0.15, average RMS error over 10 episodes 0.306
# 19-state walk, n =  32: best alpha 0.10, average RMS error over 10 episodes 0.358
# 19-state walk, n = 512: best alpha 0.05, average RMS error over 10 episodes 0.504
# 19-state walk, TD(lambda = 0.00): best alpha 0.80, average RMS error 0.342
# 19-state walk, TD(lambda = 0.40): best alpha 0.70, average RMS error 0.266
# 19-state walk, TD(lambda = 0.80): best alpha 0.30, average RMS error 0.254
# 19-state walk, TD(lambda = 0.90): best alpha 0.20, average RMS error 0.283; diverges for alpha >= 0.75
# 19-state walk, TD(lambda = 0.95): best alpha 0.10, average RMS error 0.330; diverges for alpha >= 0.50
# 19-state walk, TD(lambda = 1.00): best alpha 0.05, average RMS error 0.617; diverges for alpha >= 0.15
