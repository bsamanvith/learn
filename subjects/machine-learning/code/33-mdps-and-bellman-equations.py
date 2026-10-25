import itertools
import numpy as np

# The 4 x 3 gridworld of Russell and Norvig. Row 0 is the top. The agent starts bottom-left;
# (1, 1) is a wall; entering (0, 3) pays +1 and entering (1, 3) pays -1, and both end the episode.
# A move goes the intended way with probability 0.8 and sideways with 0.1 each; bumping into
# the wall or the edge leaves the agent where it is. Every step costs a "living reward" r.
rows, cols, wall = 3, 4, (1, 1)
cells = [(i, j) for i in range(rows) for j in range(cols) if (i, j) != wall]
S = len(cells) + 1                                     # 11 cells plus one absorbing "end" state
END = S - 1
idx = {c: k for k, c in enumerate(cells)}
terminal = {(0, 3): 1.0, (1, 3): -1.0}
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]             # up, right, down, left
arrows = "^>v<"
start = idx[(2, 0)]

def mdp(living):
    """P[a, s, s'] and expected reward R[a, s] for the gridworld with the given living reward."""
    P = np.zeros((4, S, S)); R = np.zeros((4, S))
    for c in cells:
        s = idx[c]
        for a in range(4):
            if c in terminal:                          # the exit: collect the payoff and stop
                P[a, s, END] = 1; R[a, s] = terminal[c]; continue
            for d, p in [(a, 0.8), ((a + 1) % 4, 0.1), ((a + 3) % 4, 0.1)]:
                n = (c[0] + moves[d][0], c[1] + moves[d][1])
                n = n if n in idx else c               # off the grid or into the wall: stay
                P[a, s, idx[n]] += p
            R[a, s] = living
    P[:, END, END] = 1                                 # the end state loops with reward 0
    return P, R

def evaluate(P, R, pi, gamma):
    """V^pi from the Bellman expectation equation, (I - gamma P_pi) V = r_pi. pi is (S, 4)."""
    P_pi = np.einsum("sa,ast->st", pi, P); r_pi = np.einsum("sa,as->s", pi, R)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)

def q_from_v(P, R, V, gamma):
    return R + gamma * P @ V                           # Q[a, s] = R(s, a) + gamma sum_s' P V(s')

gamma = 0.99
P, R = mdp(-0.04)
show = lambda V: "\n".join("  " + " ".join("  wall " if (i, j) == wall else f"{V[idx[(i, j)]]:+.3f}" for j in range(cols)) for i in range(rows))

# 1. The uniformly random policy: solve the linear system, and check it against Monte Carlo.
uniform = np.full((S, 4), 0.25)
V_rand = evaluate(P, R, uniform, gamma)
print("V of the random policy (gamma = 0.99, living reward -0.04):"); print(show(V_rand))
rng = np.random.default_rng(0)
returns = []
for _ in range(20_000):
    s, G, disc = start, 0.0, 1.0
    while s != END:
        a = rng.integers(4)
        G += disc * R[a, s]; disc *= gamma
        s = rng.choice(S, p=P[a, s])
    returns.append(G)
returns = np.array(returns)
print(f"from the start: linear solve {V_rand[start]:+.4f}, Monte Carlo over 20,000 episodes"
      f" {returns.mean():+.4f} +- {returns.std() / np.sqrt(len(returns)):.4f}")

# 2. Every deterministic policy: 4^9 = 262,144 choices on the 9 non-terminal cells.
free = [idx[c] for c in cells if c not in terminal]
def all_values(P, R, gamma, chunk=16384):
    """V^pi for every deterministic policy, by batched linear solves."""
    out = []
    combos = np.array(list(itertools.product(range(4), repeat=len(free))), dtype=np.int8)
    for k in range(0, len(combos), chunk):
        acts = np.zeros((len(combos[k:k + chunk]), S), dtype=int); acts[:, free] = combos[k:k + chunk]
        P_pi = P[acts, np.arange(S)]                   # (n, S, S): row s is P[pi(s), s]
        r_pi = R[acts, np.arange(S)]
        out.append(np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi[..., None])[..., 0])
    return combos, np.concatenate(out)

combos, Vs = all_values(P, R, gamma)
print(f"{len(Vs):,} deterministic policies evaluated")
b0 = np.argmax(Vs[:, start])                           # the best policy judged from the start alone
V_max = Vs.max(0)                                      # the best value of each state, over all policies
dominant = np.flatnonzero(np.all(Vs >= V_max - 1e-10, axis=1))
V_star = Vs[dominant[0]]
print(f"best from the start: {Vs[b0, start]:+.4f}; but at the bottom-right cell it gets {Vs[b0, idx[(2, 3)]]:+.4f}"
      f" while some policy gets {V_max[idx[(2, 3)]]:+.4f} there")
print(f"policies that are best in every state at once: {len(dominant)}; their value from the start {V_star[start]:+.4f}")
Q_star = q_from_v(P, R, V_star, gamma)
print(f"Bellman optimality residual max_s |V*(s) - max_a Q*(s, a)|: {np.max(np.abs(V_star - Q_star.max(0))):.2e}")
print("V*:"); print(show(V_star))

def policy_string(Q):
    a = Q.argmax(0)
    return " | ".join("".join("#" if (i, j) == wall else ("+" if terminal.get((i, j), 0) > 0 else "-") if (i, j) in terminal
                              else arrows[a[idx[(i, j)]]] for j in range(cols)) for i in range(rows))
print("optimal policy (rows top to bottom):", policy_string(Q_star))

# 3. The living reward changes what is optimal.
for living in [-2.0, -0.4, -0.04, -0.01, 0.1]:
    Pl, Rl = mdp(living)
    _, V = all_values(Pl, Rl, gamma)
    Vb = V[np.flatnonzero(np.all(V >= V.max(0) - 1e-10, axis=1))[0]]   # the policy best everywhere
    print(f"living reward {living:+.2f}: V*(start) {Vb[start]:+7.3f}   policy {policy_string(q_from_v(Pl, Rl, Vb, gamma))}")

# 4. The discounted occupancy measure, and the objective written with it.
P_pi = np.einsum("sa,ast->st", np.eye(4)[Q_star.argmax(0)], P)
mu0 = np.eye(S)[start]
d = (1 - gamma) * mu0 @ np.linalg.inv(np.eye(S) - gamma * P_pi)
r_pi = R[Q_star.argmax(0), np.arange(S)]
print(f"occupancy sums to {d.sum():.6f}; d . r / (1 - gamma) = {d @ r_pi / (1 - gamma):+.4f} = V*(start) {V_star[start]:+.4f}")
N = np.linalg.inv(np.eye(END) - P_pi[:END, :END])     # fundamental matrix of the absorbing chain (lesson 31)
print(f"expected steps from the start under pi*, counting the exit: {N[start].sum():.2f}")

# V of the random policy (gamma = 0.99, living reward -0.04):
#   -0.977 -0.659 -0.206 +1.000
#   -1.172   wall  -0.806 -1.000
#   -1.254 -1.224 -1.083 -1.100
# from the start: linear solve -1.2539, Monte Carlo over 20,000 episodes -1.2501 +- 0.0063
# 262,144 deterministic policies evaluated
# best from the start: +0.6507; but at the bottom-right cell it gets -0.8751 while some policy gets +0.3380 there
# policies that are best in every state at once: 1; their value from the start +0.6507
# Bellman optimality residual max_s |V*(s) - max_a Q*(s, a)|: 1.11e-16
# V*:
#   +0.776 +0.844 +0.905 +1.000
#   +0.717   wall  +0.641 -1.000
#   +0.651 +0.593 +0.560 +0.338
# optimal policy (rows top to bottom): >>>+ | ^#^- | ^<^<
# living reward -2.00: V*(start) -10.564   policy >>>+ | ^#>- | >>>^
# living reward -0.40: V*(start)  -1.587   policy >>>+ | ^#^- | ^>^<
# living reward -0.04: V*(start)  +0.651   policy >>>+ | ^#^- | ^<^<
# living reward -0.01: V*(start)  +0.853   policy >>>+ | ^#<- | ^<<v
# living reward +0.10: V*(start) +10.000   policy >^<+ | ^#<- | ^>^v
# occupancy sums to 1.000000; d . r / (1 - gamma) = +0.6507 = V*(start) +0.6507
# expected steps from the start under pi*, counting the exit: 7.68
