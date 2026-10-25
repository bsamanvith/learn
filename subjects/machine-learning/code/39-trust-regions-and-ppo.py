import numpy as np

# ---- Part 1: the performance difference lemma, exactly, on a random MDP (8 states, 3 actions).
rng = np.random.default_rng(1)
nS, nA, gamma = 8, 3, 0.9
P = rng.dirichlet(np.ones(nS) * 0.5, (nS, nA))
R = rng.normal(0, 1, (nS, nA))
mu0 = np.full(nS, 1 / nS)

def softmax(theta):
    e = np.exp(theta - theta.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def evaluate(pi):
    """J, V, Q, A and the discounted visit counts eta = mu0 (I - gamma P_pi)^-1."""
    P_pi = np.einsum("sa,sat->st", pi, P); r_pi = (pi * R).sum(1)
    M = np.linalg.inv(np.eye(nS) - gamma * P_pi)
    V = M @ r_pi; Q = R + gamma * P @ V
    return mu0 @ V, V, Q, Q - V[:, None], mu0 @ M

pi = softmax(rng.normal(0, 1, (nS, nA))); pi2 = softmax(rng.normal(0, 1, (nS, nA)))
J1, V1, Q1, A1, eta1 = evaluate(pi)
J2, _, _, _, eta2 = evaluate(pi2)
lemma = eta2 @ (pi2 * A1).sum(1)                                   # visits under the NEW policy, advantages of the OLD
print(f"J(pi') - J(pi) = {J2 - J1:+.6f}; performance difference lemma: {lemma:+.6f};"
      f" the same sum with the old policy's visits (the surrogate): {eta1 @ (pi2 * A1).sum(1):+.6f}")

# ---- Part 2: the surrogate L_pi(pi') = J(pi) + sum_s eta_pi(s) sum_a pi'(a|s) A_pi(s, a) and the
# TRPO lower bound J(pi') >= L_pi(pi') - 4 eps gamma / (1 - gamma)^2 * alpha^2, alpha = max_s TV.
theta = rng.normal(0, 1, (nS, nA)); pi = softmax(theta)
J0, V0, Q0, A0, eta0 = evaluate(pi)
grad = eta0[:, None] * pi * A0                                     # the policy gradient (lesson 38)
eps = np.abs(A0).max()
print("  step   J(pi') - J(pi)   L(pi') - J(pi)   |J - L|   |J - L| / TV^2   max TV   bound - J(pi)")
for t in [0.01, 0.03, 0.1, 0.3, 1, 3]:
    pi_new = softmax(theta + t * grad / np.abs(grad).max())
    J_new = evaluate(pi_new)[0]
    L = J0 + eta0 @ (pi_new * A0).sum(1)
    tv = 0.5 * np.abs(pi_new - pi).sum(1).max()
    kl = (pi * np.log(pi / pi_new)).sum(1).max()
    bound = L - 4 * eps * gamma / (1 - gamma) ** 2 * tv ** 2
    print(f"{t:6.2f}   {J_new - J0:+.6f}        {L - J0:+.6f}       {abs(J_new - L):.2e}   {abs(J_new - L) / tv ** 2:8.3f}       {tv:.4f}   {bound - J0:+9.3f}")

# ---- Part 3: vanilla against natural policy gradient, exact gradients, tabular softmax.
# The natural gradient for a softmax policy is the advantage itself (Kakade 2001), so NPG is
# theta <- theta + step * A_pi / (1 - gamma): soft policy iteration.
Q = np.zeros((nS, nA))
for _ in range(2000):
    Q = R + gamma * P @ Q.max(1)
J_star = mu0 @ Q.max(1)
def iterations_to(update, step, tol=1e-3, max_it=20_000):
    th = np.zeros((nS, nA))
    for it in range(1, max_it + 1):
        J, V, Qp, A, eta = evaluate(softmax(th))
        if J_star - J < tol:
            return it
        th += step * update(softmax(th), A, eta)
    return None
pg = lambda pi, A, eta: eta[:, None] * pi * A
npg = lambda pi, A, eta: A / (1 - gamma)
for name, upd, steps in [("policy gradient", pg, [1, 10, 30, 100]), ("natural policy gradient", npg, [0.1, 1, 3, 10])]:
    res = {s: iterations_to(upd, s) for s in steps}
    print(f"{name:23s}: iterations to come within 0.001 of J* = {J_star:.4f}: " +
          ", ".join(f"step {s}: {res[s] if res[s] else '>20000'}" for s in steps))

# ---- Part 4: PPO on CartPole, from scratch: a tanh MLP actor and critic with Adam, GAE(0.99, 0.95),
# 2,048 steps per iteration, 10 epochs of minibatches of 64, the clipped surrogate.
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

class MLP:
    """tanh MLP with Adam; forward caches activations for one backward pass."""
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W = [rng.normal(0, 1, (a, b)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= out_scale
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.k = 0
    def forward(self, x):
        self.h = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            self.h.append(x)
        return x
    def step(self, g, lr):
        gW, gb = [], []
        for i in range(len(self.W) - 1, -1, -1):
            gW.insert(0, self.h[i].T @ g); gb.insert(0, g.sum(0))
            if i:
                g = (g @ self.W[i].T) * (1 - self.h[i] ** 2)
        norm = np.sqrt(sum((x ** 2).sum() for x in gW + gb))      # clip the global gradient norm at 0.5
        c = min(1.0, 0.5 / (norm + 1e-8))
        self.k += 1
        for j, (p, gr) in enumerate(zip(self.W + self.b, gW + gb)):
            gr = gr * c
            self.m[j] = 0.9 * self.m[j] + 0.1 * gr; self.v[j] = 0.999 * self.v[j] + 0.001 * gr ** 2
            p -= lr * (self.m[j] / (1 - 0.9 ** self.k)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.k)) + 1e-8)

SCALE = np.array([2.4, 3.0, 0.21, 3.0])

def ppo(seed, iters=50, n_steps=2048, epochs=10, mb=64, clip=0.2, lr=3e-4, gamma=0.99, lam=0.95):
    """clip=None runs the same loop on the unclipped surrogate (many epochs of plain importance-weighted policy gradient)."""
    rng = np.random.default_rng(seed); env = CartPole(rng)
    actor = MLP([4, 64, 64, 2], rng, out_scale=0.01); critic = MLP([4, 64, 64, 1], rng)
    s = env.reset(); ep_len, lengths, log = 0, [], []
    for it in range(iters):
        S, A, R, S2, D, DONE, LP = [], [], [], [], [], [], []
        for _ in range(n_steps):                                   # collect a batch with the current policy
            z = s / SCALE
            logits = actor.forward(z[None])[0]; p = np.exp(logits - logits.max()); p /= p.sum()
            a = int(rng.random() < p[1])
            s2, r, term, trunc = env.step(a); ep_len += 1
            S.append(z); A.append(a); R.append(r); S2.append(s2 / SCALE); D.append(term); DONE.append(term or trunc); LP.append(np.log(p[a]))
            s = s2
            if term or trunc:
                lengths.append(ep_len); ep_len = 0; s = env.reset()
        S, A, R, S2, D, DONE, LP = map(np.array, (S, A, R, S2, D, DONE, LP))
        V, V2 = critic.forward(S)[:, 0], critic.forward(S2)[:, 0]
        delta = R + gamma * (1 - D) * V2 - V                       # bootstrap unless the pole fell
        adv = np.zeros(n_steps); gae = 0.0
        for t in range(n_steps - 1, -1, -1):                       # GAE, cut at episode boundaries
            gae = delta[t] + gamma * lam * (1 - DONE[t]) * gae
            adv[t] = gae
        ret = adv + V
        kl_sum, clipped_frac, nmb = 0.0, 0.0, 0
        for ep in range(epochs):
            order = rng.permutation(n_steps)
            for k in range(0, n_steps, mb):
                i = order[k:k + mb]
                a_ = adv[i]; a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)  # normalise advantages per minibatch
                logits = actor.forward(S[i]); logits -= logits.max(1, keepdims=True)
                logp_all = logits - np.log(np.exp(logits).sum(1, keepdims=True))
                lp = logp_all[np.arange(len(i)), A[i]]
                ratio = np.exp(lp - LP[i])
                if clip is None:
                    coef = ratio * a_                              # d(ratio * A)/d(log pi) = ratio * A
                else:
                    unclipped = ratio * a_; clipped = np.clip(ratio, 1 - clip, 1 + clip) * a_
                    active = unclipped <= clipped                  # the min picks the unclipped term
                    coef = np.where(active, ratio * a_, 0.0)       # the clipped branch has zero gradient
                    clipped_frac += np.mean(~active); 
                pi = np.exp(logp_all)
                onehot = np.eye(2)[A[i]]
                g = -(coef[:, None] * (onehot - pi)) / len(i)       # minus: gradient ascent on the surrogate
                actor.step(g, lr)
                vpred = critic.forward(S[i])[:, 0]
                critic.step(((vpred - ret[i]) / len(i))[:, None], lr)
                kl_sum += np.mean(LP[i] - lp); nmb += 1
        log.append((it, np.mean(lengths[-10:]) if lengths else 0, kl_sum / nmb, clipped_frac / nmb))
    return lengths, log


ppo_runs = {}
for name, kw in [("PPO, clip 0.2, 10 epochs", {}), ("no clipping, 10 epochs", {"clip": None}),
                 ("no clipping, 1 epoch", {"clip": None, "epochs": 1})]:
    runs = [ppo(seed, **kw) for seed in range(3)]
    ppo_runs[name] = runs
    first = [next((it for it, m, _, _ in log if m >= 475), None) for _, log in runs]
    print(f"{name:25s}: first iteration with the last 10 episodes averaging 475+: {first};"
          f" final average {' '.join(f'{log[-1][1]:5.1f}' for _, log in runs)};"
          f" median KL(old || new) per update {np.median([kl for _, log in runs for _, _, kl, _ in log]):.4f}")

# J(pi') - J(pi) = +1.152474; performance difference lemma: +1.152474; the same sum with the old policy's visits (the surrogate): +1.120439
#   step   J(pi') - J(pi)   L(pi') - J(pi)   |J - L|   |J - L| / TV^2   max TV   bound - J(pi)
#   0.01   +0.023922        +0.023912       1.02e-05      0.493       0.0046      +0.013
#   0.03   +0.071728        +0.071636       9.18e-05      0.494       0.0136      -0.028
#   0.10   +0.238286        +0.237279       1.01e-03      0.494       0.0451      -0.851
#   0.30   +0.698749        +0.690321       8.43e-03      0.491       0.1310      -8.472
#   1.00   +1.915284        +1.863108       5.22e-02      0.441       0.3441     -61.388
#   3.00   +2.947035        +2.922682       2.44e-02      0.110       0.4700    -115.091
# policy gradient        : iterations to come within 0.001 of J* = 11.8436: step 1: 5345, step 10: 453, step 30: 49, step 100: 132
# natural policy gradient: iterations to come within 0.001 of J* = 11.8436: step 0.1: 29, step 1: 6, step 3: 5, step 10: 4
# PPO, clip 0.2, 10 epochs : first iteration with the last 10 episodes averaging 475+: [9, 14, 13]; final average 500.0 500.0 500.0; median KL(old || new) per update 0.0033
# no clipping, 10 epochs   : first iteration with the last 10 episodes averaging 475+: [4, 11, 10]; final average 208.7 500.0 500.0; median KL(old || new) per update 0.3804
# no clipping, 1 epoch     : first iteration with the last 10 episodes averaging 475+: [47, 47, 37]; final average 451.6 500.0 333.7; median KL(old || new) per update 0.0005
