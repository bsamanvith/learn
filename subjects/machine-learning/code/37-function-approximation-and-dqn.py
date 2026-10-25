import numpy as np

# ---- Part 1: linear TD and its fixed point. The 19-state random walk of lesson 35 with state
# aggregation: 19 states in 5 groups of about 4, one weight per group. gamma = 1, so the walk's
# stationary visiting distribution mu is used to weight errors. Compare semi-gradient TD(0) and
# gradient Monte Carlo with the best possible weights, the projection of v_pi.
n = 19
true_v = np.arange(-(n - 1), n, 2) / (n + 1)
groups = np.minimum(np.arange(n) * 5 // n, 4)
X = np.eye(5)[groups]                                              # one-hot group features, (19, 5)
P = np.zeros((n + 2, n + 2))                                       # states 0 and 20 are terminal
for s in range(1, n + 1):
    P[s, s - 1] = P[s, s + 1] = 0.5
r = np.zeros(n + 2); r[1] = -0.5; r[n] = 0.5                       # expected reward on leaving s
Pn = P[1:-1, 1:-1]                                                 # transitions among non-terminal states
visits = np.linalg.solve(np.eye(n) - Pn.T, np.eye(n)[n // 2])      # expected visits per episode from the middle
mu = visits / visits.sum()
D = np.diag(mu)
w_best = np.linalg.solve(X.T @ D @ X, X.T @ D @ true_v)            # minimises the mu-weighted value error
A = X.T @ D @ (X - Pn @ X); b = X.T @ D @ r[1:-1]
w_td = np.linalg.solve(A, b)                                       # the TD fixed point: E[delta x] = 0
ve = lambda w: np.sqrt(mu @ (X @ w - true_v) ** 2)
print(f"value error (mu-weighted RMS): best possible {ve(w_best):.4f}, TD fixed point {ve(w_td):.4f}")
rng = np.random.default_rng(0)
w_mc, w_t = np.zeros(5), np.zeros(5)
for e in range(1, 20_001):
    s, path = n // 2 + 1, []
    while 1 <= s <= n:
        path.append(s); s += rng.choice((-1, 1))
    G = 1.0 if s == n + 1 else -1.0
    alpha = 1e-4
    for t, st in enumerate(path):
        w_mc += alpha * (G - X[st - 1] @ w_mc) * X[st - 1]           # gradient MC: target is the return
        nxt = path[t + 1] if t + 1 < len(path) else s
        v_next = X[nxt - 1] @ w_t if 1 <= nxt <= n else 0.0
        rew = 0.0 if 1 <= nxt <= n else G
        w_t += alpha * (rew + v_next - X[st - 1] @ w_t) * X[st - 1]  # semi-gradient TD(0)
print(f"after 20,000 episodes: gradient MC {ve(w_mc):.4f} (|w - w_best| = {np.abs(w_mc - w_best).max():.3f}),"
      f" semi-gradient TD {ve(w_t):.4f} (|w - w_TD| = {np.abs(w_t - w_td).max():.3f})")

# ---- Part 2: the deadly triad. Baird's counterexample (Sutton and Barto, figure 11.1): seven
# states, eight weights, linear features, all rewards 0, so v = 0 is representable exactly.
# The behaviour policy goes to one of states 1-6 with probability 6/7 and to state 7 with 1/7;
# the target policy always goes to state 7. gamma = 0.99.
Xb = np.zeros((7, 8))
for i in range(6):
    Xb[i, i] = 2; Xb[i, 7] = 1
Xb[6, 6] = 1; Xb[6, 7] = 2
def baird(on_policy, steps=1000, alpha=0.01, seed=0):
    """Semi-gradient TD(0). Off-policy: learn about the target policy from behaviour data,
    weighting each update by rho = pi / b. On-policy: learn the behaviour policy's own values."""
    rng = np.random.default_rng(seed)
    w = np.array([1, 1, 1, 1, 1, 1, 10, 1], float)
    s = int(rng.integers(7))
    for _ in range(steps):
        solid = rng.random() < 1 / 7                               # the behaviour policy acts
        s2 = 6 if solid else int(rng.integers(6))
        rho = 1.0 if on_policy else (7.0 if solid else 0.0)       # importance ratio pi / b
        delta = 0 + 0.99 * Xb[s2] @ w - Xb[s] @ w
        w += alpha * rho * delta * Xb[s]
        s = s2
    return np.abs(Xb @ w).max()                                    # the largest |v(s)|; the truth is 0
print(f"Baird, largest |v(s)| (true value 0): off-policy after 1,000 steps {baird(False):.3g}, after 10,000 {baird(False, 10_000):.3g};"
      f" on-policy after 1,000 {baird(True):.3g}, after 10,000 {baird(True, 10_000):.3g} (it starts at 12)")

# ---- Part 3: DQN on CartPole, written from scratch: a NumPy MLP, a replay buffer, a target
# network (Polyak averaging), the Huber loss, epsilon-greedy exploration and Double DQN targets.
class CartPole:
    """The classic cart-pole (Barto, Sutton and Anderson, 1983), with the constants used by
    OpenAI Gym: push the cart left or right; reward 1 per step; the episode ends when the pole
    leans more than 12 degrees or the cart leaves [-2.4, 2.4], or after 500 steps."""
    g, mc, mp, l, F, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    def __init__(self, rng):
        self.rng = rng
    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4); self.t = 0
        return self.s.copy()
    def step(self, a):
        x, xd, th, thd = self.s
        f = self.F if a == 1 else -self.F
        ct, st = np.cos(th), np.sin(th)
        m = self.mc + self.mp
        tmp = (f + self.mp * self.l * thd ** 2 * st) / m
        thacc = (self.g * st - ct * tmp) / (self.l * (4 / 3 - self.mp * ct ** 2 / m))
        xacc = tmp - self.mp * self.l * thacc * ct / m
        x, xd, th, thd = x + self.dt * xd, xd + self.dt * xacc, th + self.dt * thd, thd + self.dt * thacc
        self.s = np.array([x, xd, th, thd]); self.t += 1
        fell = abs(x) > 2.4 or abs(th) > 12 * np.pi / 180
        return self.s.copy(), 1.0, fell, self.t >= 500           # next state, reward, terminal, truncated

class MLP:
    """A two-hidden-layer ReLU network with Adam, written out by hand."""
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.k = 0
    scale = None
    def forward(self, x):
        if self.scale is not None:
            x = x / self.scale
        self.h = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            self.h.append(x)
        return x
    def backward_and_step(self, g, lr=1e-3, b1=0.9, b2=0.999):
        grads_W, grads_b = [], []
        for i in range(len(self.W) - 1, -1, -1):
            grads_W.insert(0, self.h[i].T @ g); grads_b.insert(0, g.sum(0))
            if i:
                g = (g @ self.W[i].T) * (self.h[i] > 0)
        self.k += 1
        for j, (p, gr) in enumerate(zip(self.W + self.b, grads_W + grads_b)):
            self.m[j] = b1 * self.m[j] + (1 - b1) * gr; self.v[j] = b2 * self.v[j] + (1 - b2) * gr ** 2
            p -= lr * (self.m[j] / (1 - b1 ** self.k)) / (np.sqrt(self.v[j] / (1 - b2 ** self.k)) + 1e-8)
    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]; self.b = [b.copy() for b in other.b]

def dqn(seed, steps=100_000, target_every=1000, lr=5e-4, tau=0.005, replay=True, double=True, batch=64, gamma=0.99):
    rng = np.random.default_rng(seed)
    env = CartPole(rng)
    q = MLP([4, 64, 64, 2], rng); q_tgt = MLP([4, 64, 64, 2], rng); q_tgt.copy_from(q)
    q.scale = q_tgt.scale = np.array([2.4, 3.0, 0.21, 3.0])          # rough ranges of the four state variables
    cap = 50_000 if replay else batch                              # without replay: only the latest transitions
    S, A, R, S2, D = np.zeros((cap, 4)), np.zeros(cap, int), np.zeros(cap), np.zeros((cap, 4)), np.zeros(cap)
    n = 0
    s = env.reset(); ep_ret, returns = 0.0, []
    for t in range(steps):
        eps = max(0.05, 1 - t / 10_000)                            # explore a lot at first
        a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(q.forward(s[None])[0]))
        s2, r, term, trunc = env.step(a)
        i = n % cap; S[i], A[i], R[i], S2[i], D[i] = s, a, r, s2, term; n += 1
        ep_ret += r; s = s2
        if term or trunc:
            returns.append((t, ep_ret)); s = env.reset(); ep_ret = 0.0
        if n >= batch:
            idx = rng.integers(0, min(n, cap), batch) if replay else np.arange(batch)
            if target_every:
                nxt = q_tgt.forward(S2[idx])
                if double:                                         # choose with the online net, evaluate with the target net
                    a_star = np.argmax(q.forward(S2[idx]), 1); nxt_v = nxt[np.arange(batch), a_star]
                else:
                    nxt_v = nxt.max(1)
            else:                                                  # no target network: bootstrap from the net being trained
                nxt_v = q.forward(S2[idx]).max(1)
            y = R[idx] + gamma * (1 - D[idx]) * nxt_v
            out = q.forward(S[idx])
            q_max = np.abs(out).max()                                  # watch for divergence
            err = out[np.arange(batch), A[idx]] - y
            g = np.zeros_like(out); g[np.arange(batch), A[idx]] = np.clip(err, -1, 1) / batch   # Huber loss gradient
            q.backward_and_step(g, lr)
            if tau:                                                # Polyak averaging of the target network
                for P_t, P_o in zip(q_tgt.W + q_tgt.b, q.W + q.b):
                    P_t += tau * (P_o - P_t)
            elif target_every and t % target_every == 0:
                q_tgt.copy_from(q)
    # Evaluate the final greedy policy on 20 fresh episodes.
    ev = CartPole(np.random.default_rng(10_000 + seed)); tot = []
    for _ in range(20):
        s, done, G = ev.reset(), False, 0
        while not done:
            s, r, term, trunc = ev.step(int(np.argmax(q.forward(s[None])[0]))); G += r; done = term or trunc
        tot.append(G)
    return np.array(returns), np.mean(tot), q_max


results = {}
for name, kw in [("DQN", {}), ("no target network", {"tau": None, "target_every": 0}), ("no replay", {"replay": False})]:
    res = [dqn(seed, **kw) for seed in range(3)]
    results[name] = res
    last100 = [np.mean(r[-100:, 1]) for r, _, _ in res]
    print(f"{name:17s}: mean return of the last 100 training episodes, seeds 0-2: {' '.join(f'{x:5.1f}' for x in last100)};"
          f" greedy policy at the end: {' '.join(f'{ev:5.1f}' for _, ev, _ in res)}; largest |Q| in the last batch:"
          f" {' '.join(f'{qm:.3g}' for _, _, qm in res)}")

# value error (mu-weighted RMS): best possible 0.1068, TD fixed point 0.2691
# after 20,000 episodes: gradient MC 0.1120 (|w - w_best| = 0.050), semi-gradient TD 0.2791 (|w - w_TD| = 0.030)
# Baird, largest |v(s)| (true value 0): off-policy after 1,000 steps 696, after 10,000 1.5e+12; on-policy after 1,000 1.76, after 10,000 0.345 (it starts at 12)
# DQN              : mean return of the last 100 training episodes, seeds 0-2: 190.2 345.3 234.1; greedy policy at the end: 500.0 500.0  93.7; largest |Q| in the last batch: 138 124 121
# no target network: mean return of the last 100 training episodes, seeds 0-2:   9.8   9.5   9.6; greedy policy at the end:   9.7   9.2   9.6; largest |Q| in the last batch: 1.65e+09 1.72e+09 1.74e+09
# no replay        : mean return of the last 100 training episodes, seeds 0-2: 383.8 211.8 334.2; greedy policy at the end: 500.0 127.7 500.0; largest |Q| in the last batch: 98 99.7 96.8
