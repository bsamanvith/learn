"""Assignment A10: deep RL from scratch in NumPy. DQN on CartPole, PPO on Pendulum.

Fill in every function and method marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

Given: the two environments, an MLP with a hand-written backward pass, and Adam. Yours: the
replay buffer, the DQN targets and loss, the target-network update and the DQN training loop;
the Gaussian policy's log-probability and its gradients, GAE, PPO's clipped objective and its
gradient, the PPO update and the PPO training loop.
"""
import numpy as np


# ---------------------------------------------------------------- given: environments

class CartPole:
    """Lesson 37's cart-pole. Two actions (push left, push right), reward 1 per step; the episode
    terminates when the pole leans past 12 degrees or the cart leaves [-2.4, 2.4], and is
    truncated after 500 steps. step returns (next observation, reward, terminated, truncated).
    Observations are scaled by the rough ranges of the four state variables."""
    g, mc, mp, l, F, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    scale = np.array([2.4, 3.0, 0.21, 3.0])
    obs_dim, n_actions = 4, 2

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4); self.t = 0
        return self.s / self.scale

    def step(self, a):
        x, xd, th, thd = self.s
        f = self.F if a == 1 else -self.F
        ct, st = np.cos(th), np.sin(th)
        m = self.mc + self.mp
        tmp = (f + self.mp * self.l * thd ** 2 * st) / m
        thacc = (self.g * st - ct * tmp) / (self.l * (4 / 3 - self.mp * ct ** 2 / m))
        xacc = tmp - self.mp * self.l * thacc * ct / m
        self.s = np.array([x + self.dt * xd, xd + self.dt * xacc, th + self.dt * thd, thd + self.dt * thacc]); self.t += 1
        fell = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * np.pi / 180
        return self.s / self.scale, 1.0, bool(fell), self.t >= 500


class Pendulum:
    """The pendulum swing-up of OpenAI Gym (Pendulum-v1). The action is a torque, clipped to
    [-2, 2]. The reward is -(theta^2 + 0.1 thetadot^2 + 0.001 u^2), with theta in [-pi, pi]
    measured from upright, so the best possible return is 0 and a random policy gets about
    -1,200. Episodes never terminate; they are truncated after 200 steps. Observations are
    (cos theta, sin theta, thetadot / 8)."""
    max_speed, max_torque, dt, g, m, l = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0
    obs_dim, act_dim = 3, 1

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.th, self.thd, self.t = self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-1, 1), 0
        return self._obs()

    def _obs(self):
        return np.array([np.cos(self.th), np.sin(self.th), self.thd / self.max_speed])

    def step(self, a):
        u = float(np.clip(np.asarray(a).reshape(-1)[0], -self.max_torque, self.max_torque))
        th_n = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_n ** 2 + 0.1 * self.thd ** 2 + 0.001 * u ** 2
        thd = self.thd + (3 * self.g / (2 * self.l) * np.sin(self.th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        self.thd = float(np.clip(thd, -self.max_speed, self.max_speed))
        self.th = self.th + self.thd * self.dt
        self.t += 1
        return self._obs(), -cost, False, self.t >= 200


# ---------------------------------------------------------------- given: a network and Adam

class MLP:
    """Fully connected network with tanh (or relu) hidden layers and a linear output.
    forward(x) caches what backward needs; backward(g) takes dL/d(output), stores the parameter
    gradients in self.grads (same order as self.params) and returns dL/d(input)."""
    def __init__(self, sizes, rng, act="tanh", out_scale=1.0):
        self.act = act
        self.params = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            W = rng.normal(0, 1, (a, b)) * (np.sqrt(2 / a) if act == "relu" else 1 / np.sqrt(a))
            if i == len(sizes) - 2:
                W *= out_scale
            self.params += [W, np.zeros(b)]

    def forward(self, x):
        self.h = [x]
        n = len(self.params) // 2
        for i in range(n):
            x = x @ self.params[2 * i] + self.params[2 * i + 1]
            if i < n - 1:
                x = np.tanh(x) if self.act == "tanh" else np.maximum(x, 0)
            self.h.append(x)
        return x

    def backward(self, g):
        n = len(self.params) // 2
        self.grads = [None] * len(self.params)
        for i in range(n - 1, -1, -1):
            self.grads[2 * i] = self.h[i].T @ g; self.grads[2 * i + 1] = g.sum(0)
            g = g @ self.params[2 * i].T
            if i:
                g = g * ((1 - self.h[i] ** 2) if self.act == "tanh" else (self.h[i] > 0))
        return g

    def copy(self):
        new = MLP.__new__(MLP); new.act = self.act; new.params = [p.copy() for p in self.params]
        return new


class Adam:
    """Adam over a list of parameter arrays, updated in place. step(grads) descends; pass
    max_norm to clip the global gradient norm first."""
    def __init__(self, params, lr, max_norm=None):
        self.params, self.lr, self.max_norm, self.t = params, lr, max_norm, 0
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        if self.max_norm is not None:
            norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            grads = [g * min(1.0, self.max_norm / (norm + 1e-8)) for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


# ---------------------------------------------------------------- Part 1: DQN

class ReplayBuffer:
    """A ring buffer of transitions (s, a, r, s2, terminated). Once full, each new transition
    overwrites the oldest."""
    def __init__(self, capacity, obs_dim):
        self.capacity, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim)); self.a = np.zeros(capacity, dtype=int); self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, obs_dim)); self.d = np.zeros(capacity)

    def add(self, s, a, r, s2, terminated):
        """Store one transition. self.n is the number of stored transitions (at most capacity)."""
        raise NotImplementedError  # TODO

    def sample(self, batch, rng):
        """batch transitions drawn uniformly with replacement (rng.integers) from those stored.
        Returns (s, a, r, s2, terminated) as arrays."""
        raise NotImplementedError  # TODO


def dqn_targets(r, terminated, q_next_target, gamma, q_next_online=None):
    """y = r + gamma (1 - terminated) * value of the next state. The value is
    max_a q_next_target[:, a] for DQN; for Double DQN (q_next_online given) it is
    q_next_target[:, argmax_a q_next_online[:, a]]."""
    raise NotImplementedError  # TODO


def huber_loss_grad(q, actions, y):
    """Mean over the batch of the Huber loss (delta = 1) between q[i, actions[i]] and y[i].
    Returns (loss, dloss/dq), where dloss/dq has the shape of q and is zero for actions not taken."""
    raise NotImplementedError  # TODO


def polyak_update(target, online, tau):
    """target <- (1 - tau) target + tau online, for every parameter array, in place."""
    raise NotImplementedError  # TODO


def train_dqn(seed, steps=60_000, lr=5e-4, tau=0.005, gamma=0.99, batch=64, capacity=50_000,
              eps_steps=10_000, double=True):
    """DQN on CartPole: an MLP [4, 64, 64, 2] with relu units (MLP(..., act="relu")), a copy of it
    as target network updated by polyak_update after every gradient step, epsilon decaying
    linearly from 1 to 0.05 over eps_steps, a gradient step on one sampled batch per environment
    step once the buffer holds `batch` transitions, Huber loss, Adam(net.params, lr).
    Use rng = np.random.default_rng(seed) for everything, including CartPole(rng).
    Returns (list of (step, episode return) for every finished episode, the trained network)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: PPO

def gaussian_logprob(a, mu, log_std):
    """log N(a; mu, diag(exp(log_std))^2) summed over action dimensions. a, mu: (B, d);
    log_std: (d,). Returns (B,)."""
    raise NotImplementedError  # TODO


def gaussian_logprob_grads(a, mu, log_std):
    """Gradients of gaussian_logprob with respect to mu, (B, d), and log_std, (B, d) (one row
    per sample; sum the rows to get the gradient of a sum)."""
    raise NotImplementedError  # TODO


def gae(rewards, values, next_values, terminated, ended, gamma, lam):
    """Generalised advantage estimates for a batch of T consecutive steps that may span several
    episodes. values[t] = V(s_t); next_values[t] = V(s_{t+1}), the value of the observation that
    followed step t (before any reset). terminated[t]: the episode ended at a true terminal
    state, so the next state's value is 0. ended[t]: the episode ended for any reason
    (terminated or truncated), so the advantage must not accumulate across it.
        delta_t = r_t + gamma (1 - terminated_t) next_values[t] - values[t]
        A_t     = delta_t + gamma lam (1 - ended_t) A_{t+1}
    Returns (advantages, returns = advantages + values)."""
    raise NotImplementedError  # TODO


def ppo_clip_objective(logp_new, logp_old, adv, eps):
    """PPO's clipped surrogate, averaged over the batch: mean of min(r A, clip(r, 1-eps, 1+eps) A)
    with r = exp(logp_new - logp_old). Returns (objective, d objective / d logp_new)."""
    raise NotImplementedError  # TODO


class GaussianPolicy:
    """mu = MLP(obs); a ~ N(mu, exp(log_std)^2), with a state-independent log_std. (given)"""
    def __init__(self, obs_dim, act_dim, rng, hidden=64, log_std=-0.5):
        self.net = MLP([obs_dim, hidden, hidden, act_dim], rng, out_scale=0.01)
        self.log_std = np.full(act_dim, float(log_std))

    def params(self):
        return self.net.params + [self.log_std]


def ppo_update(policy, critic, batch, rng, opt_pi, opt_v, epochs=10, minibatch=64, eps=0.2):
    """epochs passes over the batch in random minibatches (rng.permutation). For each minibatch:
    normalise the advantages within it; ascend the clipped objective with respect to the policy
    (the network's output mu and log_std, through gaussian_logprob_grads); descend the mean
    squared error between critic(obs) and the returns. batch is a dict with obs, act, logp
    (old log-probabilities), adv and ret. Returns the mean of (logp_old - logp_new) over the
    last epoch, an estimate of KL(old || new)."""
    raise NotImplementedError  # TODO


def train_ppo(seed, env_cls=Pendulum, total_steps=200_000, n_steps=2048, gamma=0.9, lam=0.95,
              lr=1e-3, epochs=10, minibatch=64, eps=0.2):
    """PPO with a GaussianPolicy and a critic MLP([obs_dim, 64, 64, 1]), both trained by Adam at
    lr with gradient norm clipped at 0.5. Each iteration collects n_steps steps with the current
    policy (store the sampled, unclipped action and its log-probability; the environment clips),
    computes GAE, and calls ppo_update. Use rng = np.random.default_rng(seed) for everything.
    Returns (list of (step, episode return) for every finished episode, the policy)."""
    raise NotImplementedError  # TODO
