import numpy as np

# ---- Part 1: cliff walking (Sutton and Barto, example 6.6). A 4 x 12 grid; start bottom-left,
# goal bottom-right; the cells between them on the bottom row are a cliff: stepping in costs
# -100 and sends the agent back to the start. Every other step costs -1. gamma = 1.
H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]                        # up, right, down, left

def step(s, a):
    i, j = s[0] + moves[a][0], s[1] + moves[a][1]
    i, j = min(max(i, 0), H - 1), min(max(j, 0), W - 1)
    if i == 3 and 0 < j < 11:
        return START, -100.0
    return (i, j), -1.0

def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))           # break ties at random

def cliff_run(method, episodes=500, alpha=0.5, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((H, W, 4))
    totals = np.zeros(episodes)
    for e in range(episodes):
        s = START; a = eps_greedy(Q, s, eps, rng)
        while s != GOAL:
            s2, r = step(s, a)
            a2 = eps_greedy(Q, s2, eps, rng)
            if s2 == GOAL:
                target = r
            elif method == "SARSA":                                # the action actually taken next
                target = r + Q[s2][a2]
            elif method == "Q-learning":                           # the greedy action
                target = r + Q[s2].max()
            else:                                                  # Expected SARSA: average under the policy
                pi = np.full(4, eps / 4); pi[np.argmax(Q[s2])] += 1 - eps
                target = r + pi @ Q[s2]
            Q[s][a] += alpha * (target - Q[s][a])
            totals[e] += r
            s, a = s2, a2
    return totals, Q

def greedy_path(Q):
    s, path = START, []
    for _ in range(60):
        if s == GOAL:
            break
        a = int(np.argmax(Q[s])); path.append("^>v<"[a]); s, _ = step(s, a)
    return "".join(path), s == GOAL

cliff = {}
for method in ["SARSA", "Q-learning", "Expected SARSA"]:
    runs = [cliff_run(method, seed=r) for r in range(50)]
    curve = np.mean([t for t, _ in runs], axis=0)
    cliff[method] = curve
    path, ok = greedy_path(runs[0][1])
    print(f"{method:14s}: average reward per episode, episodes 401-500: {curve[400:].mean():7.2f};"
          f" greedy path of run 0 ({len(path)} steps{'' if ok else ', does not reach the goal'}): {path}")

# ---- Part 2: maximisation bias (Sutton and Barto, example 6.7). From A, "right" ends the
# episode with reward 0; "left" goes to B with reward 0. B has 10 actions, each ending the episode
# with reward drawn from N(-0.1, 1). So "left" is worth -0.1 and "right" is optimal.
rng = np.random.default_rng(0)
print(f"E[max of 10 noisy estimates of -0.1]: {np.mean(np.max(rng.normal(-0.1, 1, (100_000, 10)), axis=1)):+.3f}")
def bias_run(double, episodes=300, alpha=0.1, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    QA = [np.zeros(2), np.zeros(2)]                                # A: 0 = left, 1 = right
    QB = [np.zeros(10), np.zeros(10)]
    lefts = np.zeros(episodes)
    for e in range(episodes):
        qa = QA[0] + QA[1] if double else QA[0]
        a = int(rng.integers(2)) if rng.random() < eps else int(rng.choice(np.flatnonzero(qa == qa.max())))
        lefts[e] = a == 0
        if a == 1:
            k = int(rng.integers(2)) if double else 0
            QA[k][1] += alpha * (0 - QA[k][1])
            continue
        qb = QB[0] + QB[1] if double else QB[0]
        b = int(rng.integers(10)) if rng.random() < eps else int(rng.choice(np.flatnonzero(qb == qb.max())))
        r = rng.normal(-0.1, 1)
        if double:
            k = int(rng.integers(2))                               # update one estimate at random...
            best = int(np.argmax(QB[k]))                           # ...select with it,
            QA[k][0] += alpha * (0 + QB[1 - k][best] - QA[k][0])   # ...evaluate with the other
            QB[k][b] += alpha * (r - QB[k][b])
        else:
            QA[0][0] += alpha * (0 + QB[0].max() - QA[0][0])
            QB[0][b] += alpha * (r - QB[0][b])
    return lefts

for double in [False, True]:
    frac = np.mean([bias_run(double, seed=s) for s in range(2000)], axis=0)
    name = "double Q-learning" if double else "Q-learning"
    print(f"{name:17s}: fraction choosing 'left' at episodes 10, 50, 100, 300: {frac[9]:.3f} {frac[49]:.3f} {frac[99]:.3f} {frac[299]:.3f}"
          f" (the optimal epsilon-greedy policy: 0.050)")
    if double:
        frac_double = frac
    else:
        frac_q = frac

# ---- Part 3: Q-learning is off-policy. Learn Q* of lesson 33's gridworld while behaving
# uniformly at random, and compare with Q* from value iteration.
rows, cols, wall = 3, 4, (1, 1)
cells = [(i, j) for i in range(rows) for j in range(cols) if (i, j) != wall]
G = len(cells) + 1; END = G - 1
gidx = {c: k for k, c in enumerate(cells)}
term = {(0, 3): 1.0, (1, 3): -1.0}
P = np.zeros((4, G, G)); R = np.zeros((4, G))
for c in cells:
    s = gidx[c]
    for a in range(4):
        if c in term:
            P[a, s, END] = 1; R[a, s] = term[c]; continue
        for d, p in [(a, 0.8), ((a + 1) % 4, 0.1), ((a + 3) % 4, 0.1)]:
            nb = (c[0] + moves[d][0], c[1] + moves[d][1])
            P[a, s, gidx[nb] if nb in gidx else s] += p
        R[a, s] = -0.04
P[:, END, END] = 1
gamma = 0.99
Qstar = np.zeros((4, G))
for _ in range(2000):
    Qstar = R + gamma * P @ Qstar.max(0)
rng = np.random.default_rng(3)
cum = P.cumsum(2)                                                  # for fast sampling of s'
Q = np.zeros((4, G)); N = np.zeros((4, G))
s = int(rng.integers(G - 1))
for t in range(1, 2_000_001):
    a = int(rng.integers(4))                                       # behaviour: uniformly random
    s2 = min(int(np.searchsorted(cum[a, s], rng.random(), side="right")), G - 1)
    N[a, s] += 1
    Q[a, s] += (R[a, s] + gamma * Q[:, s2].max() - Q[a, s]) / N[a, s] ** 0.7   # sum alpha = inf, sum alpha^2 < inf
    s = s2 if s2 != END else int(rng.integers(G - 1))              # restart anywhere after an exit
    if t in (10_000, 100_000, 2_000_000):
        print(f"after {t:>9,} random steps: max |Q - Q*| = {np.abs(Q - Qstar)[:, :END].max():.4f},"
              f" greedy policy optimal in {np.sum(Q[:, :END].argmax(0) == Qstar[:, :END].argmax(0))} of {END} states")

# SARSA         : average reward per episode, episodes 401-500:  -26.32; greedy path of run 0 (17 steps): ^^^>>>>>>>>>>v>vv
# Q-learning    : average reward per episode, episodes 401-500:  -49.26; greedy path of run 0 (13 steps): ^>>>>>>>>>>>v
# Expected SARSA: average reward per episode, episodes 401-500:  -20.68; greedy path of run 0 (15 steps): ^^>>>>>>>>>>>vv
# E[max of 10 noisy estimates of -0.1]: +1.440
# Q-learning       : fraction choosing 'left' at episodes 10, 50, 100, 300: 0.868 0.823 0.471 0.118 (the optimal epsilon-greedy policy: 0.050)
# double Q-learning: fraction choosing 'left' at episodes 10, 50, 100, 300: 0.458 0.142 0.088 0.072 (the optimal epsilon-greedy policy: 0.050)
# after    10,000 random steps: max |Q - Q*| = 0.0932, greedy policy optimal in 10 of 11 states
# after   100,000 random steps: max |Q - Q*| = 0.0998, greedy policy optimal in 10 of 11 states
# after 2,000,000 random steps: max |Q - Q*| = 0.0160, greedy policy optimal in 11 of 11 states
