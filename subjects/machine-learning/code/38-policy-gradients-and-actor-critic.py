import numpy as np

# ---- Part 1: the policy gradient theorem, checked exactly. A random MDP with 6 states and 3
# actions, a tabular softmax policy pi(a|s) proportional to exp(theta[s, a]), gamma = 0.9.
rng = np.random.default_rng(0)
nS, nA, gamma = 6, 3, 0.9
P = rng.dirichlet(np.ones(nS), (nS, nA))                          # P[s, a, s']
R = rng.normal(0, 1, (nS, nA))
mu0 = np.full(nS, 1 / nS)

def policy(theta):
    e = np.exp(theta - theta.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def J_and_parts(theta):
    pi = policy(theta)
    P_pi = np.einsum("sa,sat->st", pi, P); r_pi = (pi * R).sum(1)
    M = np.linalg.inv(np.eye(nS) - gamma * P_pi)
    V = M @ r_pi
    Q = R + gamma * P @ V
    return mu0 @ V, pi, Q, V, mu0 @ M                               # J, pi, Q, V, discounted visits
theta = rng.normal(0, 1, (nS, nA))
J, pi, Q, V, visits = J_and_parts(theta)
# The theorem: dJ/dtheta[s, a] = sum_s visits(s) sum_b dpi(b|s)/dtheta[s, a] Q(s, b)
#            = visits(s) pi(a|s) (Q(s, a) - V(s))      for the softmax: the advantage appears.
grad_thm = visits[:, None] * pi * (Q - V[:, None])
grad_fd = np.zeros_like(theta); h = 1e-6
for s in range(nS):
    for a in range(nA):
        tp, tm = theta.copy(), theta.copy(); tp[s, a] += h; tm[s, a] -= h
        grad_fd[s, a] = (J_and_parts(tp)[0] - J_and_parts(tm)[0]) / (2 * h)
print(f"J = {J:.4f}; max |theorem - finite differences| = {np.abs(grad_thm - grad_fd).max():.2e}"
      f" (largest gradient entry {np.abs(grad_fd).max():.3f})")

# REINFORCE estimates the same gradient from sampled episodes. Sample truncated at 100 steps
# (gamma^100 = 3e-5), with and without a baseline b(s) = V(s).
def reinforce_sample(theta, rng, baseline):
    pi = policy(theta); g = np.zeros_like(theta)
    s = rng.choice(nS, p=mu0); traj = []
    for t in range(100):
        a = rng.choice(nA, p=pi[s]); traj.append((s, a, R[s, a])); s = rng.choice(nS, p=P[s, a])
    G = 0.0
    for t in range(99, -1, -1):
        s, a, r = traj[t]; G = r + gamma * G
        score = -pi[s].copy(); score[a] += 1                        # grad of log pi(a|s) w.r.t. theta[s, :]
        g[s] += gamma ** t * (G - (V[s] if baseline else 0.0)) * score
    return g
for baseline in [False, True]:
    samples = np.array([reinforce_sample(theta, np.random.default_rng(i), baseline) for i in range(4000)])
    mean, sd = samples.mean(0), samples.std(0)
    print(f"REINFORCE, {'baseline V(s)' if baseline else 'no baseline  '}: max |mean of 4,000 estimates - theorem| = "
          f"{np.abs(mean - grad_thm).max():.3f}; average standard deviation of one estimate {sd.mean():.3f}")

# ---- Part 2: the short corridor (Sutton and Barto, example 13.1), where the best policy is
# stochastic. Three states look identical to the agent; in the second, the actions are reversed.
# Reward -1 per step. The policy is one number: the probability p of choosing "right".
def corridor_value(p):
    # states 0, 1, 2 (then the goal): V = -1 + transitions; solve the 3 x 3 linear system
    T = np.array([[1 - p, p, 0], [p, 0, 1 - p], [0, 1 - p, 0]])     # from state 2, "right" reaches the goal
    return np.linalg.solve(np.eye(3) - T, -np.ones(3))[0]
ps = np.linspace(0.01, 0.99, 9801)
vals = np.array([corridor_value(p) for p in ps])
p_best = ps[np.argmax(vals)]
print(f"short corridor: best p(right) = {p_best:.3f}, value {vals.max():.2f};"
      f" an epsilon-greedy policy (epsilon = 0.1) gets {corridor_value(0.95):.1f} or {corridor_value(0.05):.1f}")

def corridor_run(alpha, baseline_alpha=None, episodes=1000, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.array([0.0, np.log(19)])                             # preferences for (right, left): p(right) = 0.05 at first
    w = 0.0                                                         # state-value baseline (all states look alike)
    returns = np.zeros(episodes)
    for e in range(episodes):
        s, traj = 0, []
        while s != 3 and len(traj) < 1000:
            pr = 1 / (1 + np.exp(theta[1] - theta[0]))              # softmax over two preferences
            right = rng.random() < pr
            traj.append((right, pr))
            if s == 1:
                s = 0 if right else 2
            else:
                s = s + 1 if right else max(s - 1, 0)
        G = 0.0
        for t in range(len(traj) - 1, -1, -1):
            right, pr = traj[t]; G -= 1.0
            score = np.array([1 - pr, -(1 - pr)]) if right else np.array([-pr, pr])
            delta = G - (w if baseline_alpha else 0.0)
            if baseline_alpha:
                w += baseline_alpha * delta
            theta += alpha * delta * score                          # gamma = 1
        returns[e] = -len(traj)
    return returns

corr = {}
for name, a, ab in [("REINFORCE, alpha = 2^-13", 2 ** -13, None), ("REINFORCE, alpha = 2^-12", 2 ** -12, None),
                    ("with baseline, alpha = 2^-9", 2 ** -9, 2 ** -6)]:
    corr[name] = np.mean([corridor_run(a, ab, seed=s) for s in range(100)], axis=0)
    print(f"{name:28s}: average return, episodes 1-100: {corr[name][:100].mean():6.1f}; episodes 901-1000: {corr[name][900:].mean():6.1f}")

# ---- Part 3: actor-critic on CartPole with a linear softmax policy and a small critic.
# The critic is linear in hand-made features; the advantage is estimated with GAE(gamma, lambda).
class CartPole:
    g, mc, mp, l, F, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    def __init__(self, rng): self.rng = rng
    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4); self.t = 0; return self.s.copy()
    def step(self, a):
        x, xd, th, thd = self.s
        f = self.F if a == 1 else -self.F
        ct, st = np.cos(th), np.sin(th); m = self.mc + self.mp
        tmp = (f + self.mp * self.l * thd ** 2 * st) / m
        thacc = (self.g * st - ct * tmp) / (self.l * (4 / 3 - self.mp * ct ** 2 / m))
        xacc = tmp - self.mp * self.l * thacc * ct / m
        self.s = np.array([x + self.dt * xd, xd + self.dt * xacc, th + self.dt * thd, thd + self.dt * thacc]); self.t += 1
        return self.s.copy(), 1.0, abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * np.pi / 180, self.t >= 500

scale = np.array([2.4, 3.0, 0.21, 3.0])
def phi(s):                                                         # critic features: quadratic in the scaled state
    z = s / scale
    return np.concatenate([[1.0], z, np.outer(z, z)[np.triu_indices(4)]])

def actor_critic(lam, seed, episodes=600, lr_pi=10.0, lr_v=1.0, gamma=0.99, batch=5, baseline=True):
    rng = np.random.default_rng(seed); env = CartPole(rng)
    theta = np.zeros(5); w = np.zeros(15)
    out = []
    for e in range(0, episodes, batch):
        g_theta, g_w = np.zeros(5), np.zeros(15)
        for _ in range(batch):
            s, done, xs, acts, rs, fs, terms = env.reset(), False, [], [], [], [], []
            while not done:
                x = np.append(s / scale, 1.0)
                p1 = 1 / (1 + np.exp(-theta @ x))                   # probability of pushing right
                a = int(rng.random() < p1)
                s2, r, term, trunc = env.step(a); done = term or trunc
                xs.append(x); acts.append((a, p1)); rs.append(r); fs.append(phi(s)); terms.append(term)
                s = s2
            fs.append(phi(s))
            v = np.array([f @ w for f in fs]); v[-1] = 0.0 if terms[-1] else v[-1]   # bootstrap only if truncated
            T = len(rs); adv = np.zeros(T); gae = 0.0
            for t in range(T - 1, -1, -1):                          # GAE: discounted sum of TD errors
                delta = rs[t] + gamma * v[t + 1] - v[t]
                gae = delta + gamma * lam * gae; adv[t] = gae
            targets = adv + v[:-1]
            for t in range(T):
                a, p1 = acts[t]
                score = (a - p1) * xs[t]                           # grad log pi for a Bernoulli-logistic policy
                g_theta += score * (adv[t] if baseline else (targets[t]))
                g_w += (targets[t] - v[t]) * fs[t]
            out.append(T)
        theta += lr_pi * g_theta / batch / 100                      # returns are O(100): scale the step
        w += lr_v * g_w / batch / 100
    return np.array(out)

ac = {}
for name, lam, base, lr in [("GAE lambda = 0 (one-step TD advantage)", 0.0, True, 10.0), ("GAE lambda = 0.95", 0.95, True, 10.0),
                            ("GAE lambda = 1 (return minus V)", 1.0, True, 10.0), ("no critic (REINFORCE, return-to-go)", 1.0, False, 10.0),
                            ("GAE lambda = 1, step size 1", 1.0, True, 1.0)]:
    runs = [actor_critic(lam, seed, baseline=base, lr_pi=lr, lr_v=lr / 10) for seed in range(3)]
    ac[name] = runs
    print(f"{name:38s}: mean episode length, episodes 1-100: {np.mean([r[:100].mean() for r in runs]):6.1f};"
          f" last 50 episodes, per seed: {' '.join(f'{r[-50:].mean():5.1f}' for r in runs)}")

# J = -0.5859; max |theorem - finite differences| = 4.03e-10 (largest gradient entry 0.625)
# REINFORCE, no baseline  : max |mean of 4,000 estimates - theorem| = 0.026; average standard deviation of one estimate 0.854
# REINFORCE, baseline V(s): max |mean of 4,000 estimates - theorem| = 0.030; average standard deviation of one estimate 0.783
# short corridor: best p(right) = 0.586, value -11.66; an epsilon-greedy policy (epsilon = 0.1) gets -44.2 or -82.1
# REINFORCE, alpha = 2^-13    : average return, episodes 1-100:  -50.2; episodes 901-1000:  -12.5
# REINFORCE, alpha = 2^-12    : average return, episodes 1-100:  -39.4; episodes 901-1000:  -11.7
# with baseline, alpha = 2^-9 : average return, episodes 1-100:  -17.6; episodes 901-1000:  -12.0
# GAE lambda = 0 (one-step TD advantage): mean episode length, episodes 1-100:   20.5; last 50 episodes, per seed:  10.5  62.4  48.8
# GAE lambda = 0.95                     : mean episode length, episodes 1-100:  111.9; last 50 episodes, per seed: 500.0 500.0 500.0
# GAE lambda = 1 (return minus V)       : mean episode length, episodes 1-100:   25.1; last 50 episodes, per seed:   9.3   9.6   9.3
# no critic (REINFORCE, return-to-go)   : mean episode length, episodes 1-100:   20.3; last 50 episodes, per seed:   9.4   9.3  17.9
# GAE lambda = 1, step size 1           : mean episode length, episodes 1-100:   30.9; last 50 episodes, per seed: 500.0 500.0 481.4
