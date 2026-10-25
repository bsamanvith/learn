import math
import numpy as np

# ---- Part 1: Jack's car rental (Sutton and Barto, example 4.2) ----
# Two locations hold at most 20 cars each. Each day, rental requests at the two locations are
# Poisson with means 3 and 4, and returns Poisson with means 3 and 2. A rental earns $10.
# Overnight, Jack can move up to 5 cars from one location to the other at $2 a car.
# The state is the number of cars at each location at the end of the day; gamma = 0.9.
MAX, MOVE, gamma = 20, 5, 0.9
n = MAX + 1
poisson = lambda lam, k: math.exp(-lam) * lam ** k / math.factorial(k)

def location(lam_req, lam_ret):
    """For a location starting the day with c cars: the distribution of the number at the end
    of the day, T[c, c'], and the expected number of rentals, E[c]. Exact: the tails of both
    Poisson distributions are lumped into the last possible value."""
    T, E = np.zeros((n, n)), np.zeros(n)
    for c in range(n):
        for req in range(c + 1):
            p_req = poisson(lam_req, req) if req < c else 1 - sum(poisson(lam_req, k) for k in range(c))
            E[c] += p_req * req                         # rentals = min(requests, cars)
            left = c - req
            for ret in range(MAX - left + 1):
                p_ret = poisson(lam_ret, ret) if ret < MAX - left else 1 - sum(poisson(lam_ret, k) for k in range(MAX - left))
                T[c, left + ret] += p_req * p_ret       # returns beyond 20 cars are lost
    return T, E

T1, E1 = location(3, 3)
T2, E2 = location(4, 2)
actions = np.arange(-MOVE, MOVE + 1)                    # cars moved from location 1 to location 2
S = n * n                                               # state index s = c1 * 21 + c2
c1, c2 = np.divmod(np.arange(S), n)
P = np.zeros((len(actions), S, S)); R = np.full((len(actions), S), -np.inf)
for k, a in enumerate(actions):
    ok = (a <= c1) & (-a <= c2)                         # cannot move cars that are not there
    m1, m2 = np.minimum(c1 - a, MAX), np.minimum(c2 + a, MAX)   # after the move; excess cars are lost
    R[k, ok] = 10 * (E1[m1[ok]] + E2[m2[ok]]) - 2 * abs(a)
    P[k, ok] = (T1[m1[ok]][:, :, None] * T2[m2[ok]][:, None, :]).reshape(ok.sum(), S)
print(f"{S} states, {len(actions)} actions; each row of P sums to 1: {np.allclose(P.sum(2)[np.isfinite(R)], 1)}")

def evaluate(pi):
    """Exact policy evaluation: one linear solve. pi holds an action index per state."""
    P_pi, r_pi = P[pi, np.arange(S)], R[pi, np.arange(S)]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)

def greedy(V):
    Q = R + gamma * P @ V                                # -inf for impossible moves
    return Q.argmax(0), Q.max(0)

# Policy iteration, starting from "never move a car".
pi = np.full(S, MOVE)                                    # index of action 0
for it in range(1, 50):
    V = evaluate(pi)
    new, _ = greedy(V)
    changed = int((new != pi).sum())
    print(f"policy iteration {it}: V(10, 10) = {V[10 * n + 10]:.2f}, states whose action changes: {changed}")
    if changed == 0:
        break
    pi = new
V_pi, pi_star = V, pi
print(f"policy iteration: {it} evaluations; V* ranges from {V_pi.min():.1f} to {V_pi.max():.1f}")

# Value iteration from zero, stopped when successive iterates differ by less than
# eps (1 - gamma) / (2 gamma) with eps = 1e-6, which guarantees ||V - V*|| < eps / 2 (the a posteriori bound).
V = np.zeros(S)
first_optimal, jack_errs = None, []
for k in range(1, 10_000):
    pol, V_new = greedy(V)
    jack_errs.append(np.abs(V_new - V_pi).max())
    if first_optimal is None and np.array_equal(pol, pi_star):
        first_optimal = k
    diff = np.abs(V_new - V).max(); V = V_new
    if diff < 1e-6 * (1 - gamma) / (2 * gamma):
        break
print(f"value iteration: {k} sweeps; max |V - V_PI| = {np.abs(V - V_pi).max():.2e};"
      f" its greedy policy was already optimal after sweep {first_optimal}")
jack_errs = np.array(jack_errs)
print(f"error ||V_k - V*|| after 10, 50, 100 sweeps: {jack_errs[9]:.3e} {jack_errs[49]:.3e} {jack_errs[99]:.3e};"
      f" the bound gamma^k ||V*||: {gamma ** 10 * V_pi.max():.3e} {gamma ** 50 * V_pi.max():.3e} {gamma ** 100 * V_pi.max():.3e}")
print(f"observed contraction per sweep, sweeps 50 to 100: {(jack_errs[99] / jack_errs[49]) ** (1 / 50):.4f}")

# Gauss-Seidel value iteration: update states in place, one at a time, using the newest values.
V = np.zeros(S)
for k_gs in range(1, 10_000):
    diff = 0.0
    for s_ in range(S):
        v = np.max(R[:, s_] + gamma * P[:, s_] @ V)
        diff = max(diff, abs(v - V[s_])); V[s_] = v
    if diff < 1e-6 * (1 - gamma) / (2 * gamma):
        break
print(f"Gauss-Seidel value iteration: {k_gs} sweeps; max |V - V_PI| = {np.abs(V - V_pi).max():.2e}")
print("optimal number of cars to move (rows: cars at location 1 = 20, 15, 10, 5, 0; columns: location 2 = 0, 5, ..., 20):")
for r in [20, 15, 10, 5, 0]:
    print("  ", " ".join(f"{actions[pi_star[r * n + c]]:+d}" for c in range(0, n, 5)))

# ---- Part 2: the contraction, measured. The gridworld of lesson 33, gamma = 0.9 and 0.99 ----
rows, cols, wall = 3, 4, (1, 1)
cells = [(i, j) for i in range(rows) for j in range(cols) if (i, j) != wall]
G = len(cells) + 1; END = G - 1
gidx = {c: k for k, c in enumerate(cells)}
term = {(0, 3): 1.0, (1, 3): -1.0}
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
Pg = np.zeros((4, G, G)); Rg = np.zeros((4, G))
for c in cells:
    s = gidx[c]
    for a in range(4):
        if c in term:
            Pg[a, s, END] = 1; Rg[a, s] = term[c]; continue
        for d, p in [(a, 0.8), ((a + 1) % 4, 0.1), ((a + 3) % 4, 0.1)]:
            nb = (c[0] + moves[d][0], c[1] + moves[d][1])
            Pg[a, s, gidx[nb] if nb in gidx else s] += p
        Rg[a, s] = -0.04
Pg[:, END, END] = 1

for g in [0.9, 0.99]:
    pol = np.zeros(G, dtype=int)                         # V* by policy iteration
    while True:
        Vp = np.linalg.solve(np.eye(G) - g * Pg[pol, np.arange(G)], Rg[pol, np.arange(G)])
        newp = (Rg + g * Pg @ Vp).argmax(0)
        if np.array_equal(newp, pol):
            break
        pol = newp
    V, errs = np.zeros(G), []
    for k in range(1, 61):
        V = (Rg + g * Pg @ V).max(0)
        errs.append(np.abs(V - Vp).max())
    # The error shrinks by gamma times the spectral radius of the optimal policy's chain on the
    # cells, because the exits drain probability out of them: the episode ends.
    rho = np.max(np.abs(np.linalg.eigvals(Pg[pol, np.arange(G)][:END, :END])))
    print(f"gridworld, gamma {g}: ||V_k - V*|| after 10, 20, 40 sweeps: {errs[9]:.2e} {errs[19]:.2e} {errs[39]:.2e};"
          f" observed rate {(errs[39] / errs[19]) ** (1 / 20):.3f}, gamma * rho(P_pi* on the cells) = {g * rho:.3f}")

# 441 states, 11 actions; each row of P sums to 1: True
# policy iteration 1: V(10, 10) = 550.75, states whose action changes: 318
# policy iteration 2: V(10, 10) = 566.10, states whose action changes: 272
# policy iteration 3: V(10, 10) = 574.82, states whose action changes: 79
# policy iteration 4: V(10, 10) = 574.95, states whose action changes: 8
# policy iteration 5: V(10, 10) = 574.95, states whose action changes: 0
# policy iteration: 5 evaluations; V* ranges from 421.4 to 637.0
# value iteration: 197 sweeps; max |V - V_PI| = 4.71e-07; its greedy policy was already optimal after sweep 32
# error ||V_k - V*|| after 10, 50, 100 sweeps: 1.903e+02 2.510e+00 1.294e-02; the bound gamma^k ||V*||: 2.221e+02 3.283e+00 1.692e-02
# observed contraction per sweep, sweeps 50 to 100: 0.9000
# Gauss-Seidel value iteration: 108 sweeps; max |V - V_PI| = 2.46e-07
# optimal number of cars to move (rows: cars at location 1 = 20, 15, 10, 5, 0; columns: location 2 = 0, 5, ..., 20):
#    +5 +4 +2 +1 +0
#    +5 +2 +0 +0 +0
#    +4 +1 +0 +0 +0
#    +1 +0 +0 +0 +0
#    +0 +0 -2 -3 -4
# gridworld, gamma 0.9: ||V_k - V*|| after 10, 20, 40 sweeps: 1.33e-02 2.06e-06 1.05e-13; observed rate 0.432, gamma * rho(P_pi* on the cells) = 0.438
# gridworld, gamma 0.99: ||V_k - V*|| after 10, 20, 40 sweeps: 9.57e-02 1.34e-04 2.21e-11; observed rate 0.458, gamma * rho(P_pi* on the cells) = 0.433
