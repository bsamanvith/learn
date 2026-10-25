"""Problem book assignment — the bandit casino.

Sutton and Barto's 10-armed testbed: each of 2,000 casinos has ten slot machines whose mean
payouts q*(a) are drawn from N(0, 1); a pull pays N(q*(a), 1). Agents play 1,000 pulls in
every casino and we average. Vectorised over casinos, so it runs in seconds.

Agents: greedy, ε-greedy, greedy with optimistic starting values, UCB, and Thompson
sampling with a N(0, 1) prior and known noise variance 1.

Run: .venv/bin/python subjects/machine-learning/sims/bandit_casino.py [--plot]
"""

import sys
import numpy as np

R, K, T = 2000, 10, 1000


def play(agent, seed=0, shift=False, **kw):
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1, (R, K))                    # true means, the same for every agent
    best = q.argmax(1)
    play_rng = np.random.default_rng(seed + 1)
    Q = np.full((R, K), kw.get("q0", 0.0))          # estimates
    N = np.zeros((R, K))
    rows = np.arange(R)
    reward, optimal, regret = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        if shift and t == T // 2:                   # the casino changes its machines
            q = q + rng.normal(0, 1, (R, K))
            best = q.argmax(1)
        if agent == "eps":
            explore = play_rng.random(R) < kw["eps"]
            a = np.where(explore, play_rng.integers(0, K, R), argmax_random(Q, play_rng))
        elif agent == "ucb":
            bonus = kw["c"] * np.sqrt(np.log(t + 1) / np.maximum(N, 1e-12))
            a = argmax_random(np.where(N == 0, np.inf, Q + bonus), play_rng)
        elif agent == "thompson":
            # Posterior for a mean with prior N(0, 1) and n observations of unit-variance noise:
            # N(sum/(n + 1), 1/(n + 1)). Here Q holds the running sum.
            a = (Q / (N + 1) + play_rng.normal(0, 1, (R, K)) / np.sqrt(N + 1)).argmax(1)
        r = play_rng.normal(q[rows, a], 1)
        N[rows, a] += 1
        if agent == "thompson":
            Q[rows, a] += r
        else:
            Q[rows, a] += (r - Q[rows, a]) / N[rows, a] if "q0" not in kw else 0.1 * (r - Q[rows, a])
        reward[t] = r.mean()
        optimal[t] = (a == best).mean()
        regret[t] = (q[rows, best] - q[rows, a]).mean()
    return reward, optimal, np.cumsum(regret)


def argmax_random(Q, rng):
    """argmax with ties broken at random (np.argmax always picks the first)."""
    return (Q + 1e-9 * rng.random(Q.shape)).argmax(1)


rng = np.random.default_rng(0)
q = rng.normal(0, 1, (R, K))
print(f"Step 0 · {R} casinos × {K} arms; mean of the best arm E[max q*] = {q.max(1).mean():.3f}")

AGENTS = [
    ("greedy", "eps", {"eps": 0.0}),
    ("ε = 0.01", "eps", {"eps": 0.01}),
    ("ε = 0.1", "eps", {"eps": 0.1}),
    ("optimistic Q0 = 5, α = 0.1", "eps", {"eps": 0.0, "q0": 5.0}),
    ("UCB, c = 2", "ucb", {"c": 2.0}),
    ("Thompson", "thompson", {}),
]
results = {}
print("\n         agent                         reward, last 100   optimal arm, last 100   total regret after 100 / 1000")
for name, kind, kw in AGENTS:
    rew, opt, reg = play(kind, **kw)
    results[name] = (rew, opt, reg)
    print(f"         {name:<30} {rew[-100:].mean():8.3f}          {100 * opt[-100:].mean():8.1f}%              {reg[99]:7.1f} / {reg[-1]:7.1f}")

rew, opt, reg = results["greedy"]
print(f"         (greedy is on the best arm in only {100 * opt[-1]:.1f}% of casinos at pull 1,000)")

# Step 3 · break it: at pull 500 every machine's mean moves by N(0, 1).
print("\nStep 3 · the casino changes at pull 500; on the new best arm, last 100 pulls:")
for name, kind, kw in AGENTS:
    _, opt, _ = play(kind, shift=True, **kw)
    print(f"         {name:<30} {100 * opt[-100:].mean():5.1f}%   (just before the change: {100 * opt[400:500].mean():5.1f}% on the old best)")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, (rew, opt, reg) in results.items():
        ax[0].plot(100 * np.convolve(opt, np.ones(20) / 20, "valid"), label=name)
        ax[1].plot(reg, label=name)
    ax[0].set(xlabel="pull", ylabel="% on the best arm", title="Who finds the best machine?")
    ax[1].set(xlabel="pull", ylabel="regret", xscale="log", title="Regret: linear for ε-greedy, log-like for UCB and Thompson")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.show()
