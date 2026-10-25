"""Assignment A9: tabular reinforcement learning, from Markov chains to Q-learning, in NumPy.

Fill in every function marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

The environments are given: gridworlds built from a text layout (including the 4 x 3 world of
lesson 33 and the cliff of lesson 36), the random walk of lesson 35 and the maximisation-bias
MDP of lesson 36. Each exposes its full model, P[a, s, s'] and R[a, s], for the planning parts,
and reset() and step() for the learning parts. Everything else is yours.

Conventions used throughout:
    P has shape (A, S, S), R has shape (A, S): R[a, s] is the expected reward of action a in s.
    V has shape (S,), Q has shape (S, A), a policy pi has shape (S, A) with rows summing to 1.
    Terminal states are absorbing with reward 0, so the Bellman equations need no special case.
"""
import numpy as np


# ---------------------------------------------------------------- given: environments

class GridWorld:
    """A grid from a list of strings. Characters:
        '.'  an ordinary cell              '#'  a wall
        'S'  the start (an ordinary cell)  'G'  a terminal cell: entering it pays `goal`
        'P'  a terminal cell: entering it pays `pit`
        'C'  a cliff: entering it pays `cliff` and sends the agent back to the start
    Actions 0..3 are up, right, down, left. With probability 1 - slip the move goes the intended
    way, otherwise one of the two perpendicular ways (slip / 2 each). Moving off the grid or into
    a wall leaves the agent where it is. Every move that enters an ordinary cell pays `living`.
    States are the non-wall cells in row-major order."""
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, layout, slip=0.0, living=-1.0, goal=1.0, pit=-1.0, cliff=-100.0):
        self.layout, self.slip = layout, slip
        self.pay = {"G": goal, "P": pit, "C": cliff}
        self.living = living
        self.cells = [(i, j) for i, row in enumerate(layout) for j, ch in enumerate(row) if ch != "#"]
        self.index = {c: k for k, c in enumerate(self.cells)}
        self.n_states, self.n_actions = len(self.cells), 4
        self.start = next(k for k, (i, j) in enumerate(self.cells) if layout[i][j] == "S")
        self.terminal = np.array([layout[i][j] in "GP" for i, j in self.cells])
        self.P = np.zeros((4, self.n_states, self.n_states)); self.R = np.zeros((4, self.n_states))
        for s in range(self.n_states):
            if self.terminal[s]:
                self.P[:, s, s] = 1
                continue
            for a in range(4):
                for d, p in self._directions(a):
                    s2, r = self._move(s, d)
                    self.P[a, s, s2] += p; self.R[a, s] += p * r

    def _directions(self, a):
        return [(d, p) for d, p in [(a, 1 - self.slip), ((a + 1) % 4, self.slip / 2), ((a + 3) % 4, self.slip / 2)] if p > 0]

    def _move(self, s, d):
        i, j = self.cells[s]
        n = (i + self.moves[d][0], j + self.moves[d][1])
        if n not in self.index:
            n = (i, j)
        ch = self.layout[n[0]][n[1]]
        if ch == "C":
            return self.start, self.pay["C"]
        return self.index[n], self.pay.get(ch, self.living)

    def reset(self):
        return self.start

    def step(self, s, a, rng):
        """Sample one transition from non-terminal s: returns (next state, reward, done)."""
        dirs = self._directions(a)
        d = dirs[0][0] if len(dirs) == 1 else dirs[rng.choice(len(dirs), p=[p for _, p in dirs])][0]
        s2, r = self._move(s, d)
        return s2, r, bool(self.terminal[s2])


GRID_4x3 = ["...G",
            ".#.P",
            "S..."]
CLIFF = ["............",
         "............",
         "............",
         "SCCCCCCCCCCG"]


class RandomWalk:
    """Lesson 35's random walk: states 0..n-1 in a row (plus terminal states n and n+1 at the
    left and right ends), start in the middle, step left or right with probability 1/2 whatever
    the action; reward 1 for leaving on the right, else 0. True values are (k + 1) / (n + 1)."""
    def __init__(self, n=5):
        self.n, self.n_states, self.n_actions = n, n + 2, 1
        self.true_values = np.arange(1, n + 1) / (n + 1)

    def reset(self):
        return self.n // 2

    def step(self, s, a, rng):
        s2 = s + (1 if rng.random() < 0.5 else -1)
        if s2 < 0:
            return self.n, 0.0, True
        if s2 >= self.n:
            return self.n + 1, 1.0, True
        return s2, 0.0, False


class MaxBias:
    """Lesson 36's maximisation-bias MDP. State 0 (A): action 0 ("left") goes to state 1 (B) with
    reward 0, action 1 ("right") ends the episode with reward 0. State 1 (B): any of the 10
    actions ends the episode with reward drawn from N(-0.1, 1). State 2 is terminal. Actions
    that do not exist in a state are never chosen: use MaxBias.valid(s)."""
    n_states, n_actions = 3, 10

    @staticmethod
    def valid(s):
        return [0, 1] if s == 0 else list(range(10))

    def reset(self):
        return 0

    def step(self, s, a, rng):
        if s == 0:
            return (1, 0.0, False) if a == 0 else (2, 0.0, True)
        return 2, float(rng.normal(-0.1, 1.0)), True


# ---------------------------------------------------------------- Part 1: Markov chains and bandits

def stationary_distribution(P):
    """The stationary distribution of an irreducible transition matrix P (rows sum to 1):
    the probability vector pi with pi P = pi. Returns shape (n,)."""
    raise NotImplementedError  # TODO


def ucb1_choose(t, counts, sums):
    """UCB1's arm at step t = 1, 2, ...: any arm not yet pulled (lowest index first), else
    argmax of mean + sqrt(2 ln t / count)."""
    raise NotImplementedError  # TODO


def thompson_choose(counts, successes, rng):
    """Thompson sampling for Bernoulli arms with Beta(1, 1) priors: draw one sample from each
    arm's posterior with rng.beta and return the argmax."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: dynamic programming

def policy_evaluation(P, R, pi, gamma):
    """V^pi exactly, by solving the Bellman expectation equation (I - gamma P_pi) V = r_pi with
    np.linalg.solve. Assumes gamma < 1."""
    raise NotImplementedError  # TODO


def q_from_v(P, R, V, gamma):
    """Q(s, a) = R(s, a) + gamma sum_s' P(s' | s, a) V(s'). Returns shape (S, A)."""
    raise NotImplementedError  # TODO


def greedy_policy(Q):
    """The deterministic greedy policy as an (S, A) one-hot matrix; ties go to the lowest index."""
    raise NotImplementedError  # TODO


def value_iteration(P, R, gamma, tol=1e-10):
    """Iterate V <- max_a [R + gamma P V] from V = 0 until the largest change in a sweep is below
    tol. Returns (V, greedy policy, number of sweeps)."""
    raise NotImplementedError  # TODO


def policy_iteration(P, R, gamma, pi=None):
    """Policy iteration from pi (default: always action 0): evaluate exactly, improve greedily,
    stop when the policy is stable. Returns (V, pi, number of improvement steps)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 3: learning to predict

def td0_prediction(env, alpha, gamma, episodes, rng, V0=0.5):
    """TD(0) for the policy built into env (RandomWalk ignores its action; pass a = 0).
    V starts at V0 for non-terminal states and 0 for terminal ones (index >= env.n).
    Returns the (n_states,) array of estimates after the given number of episodes."""
    raise NotImplementedError  # TODO


def mc_prediction(env, alpha, gamma, episodes, rng, V0=0.5):
    """Constant-alpha every-visit Monte Carlo: after each episode, move V(S_t) towards the
    return G_t for every step t. Same interface as td0_prediction."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 4: control

def epsilon_greedy(q, eps, rng, valid=None):
    """With probability eps a uniformly random action (from `valid` if given, else all), otherwise
    a greedy one, breaking ties uniformly at random. Use rng.random() once to decide, then
    rng.integers / rng.choice."""
    raise NotImplementedError  # TODO


def sarsa_target(Q, r, s2, a2, done, gamma):
    """r + gamma Q(s2, a2), or just r if the transition ended the episode."""
    raise NotImplementedError  # TODO


def q_learning_target(Q, r, s2, done, gamma, valid=None):
    """r + gamma max_a Q(s2, a) over the valid actions (all if valid is None), or r if done."""
    raise NotImplementedError  # TODO


def expected_sarsa_target(Q, r, s2, done, gamma, eps):
    """The expectation of Q(s2, .) under the eps-greedy policy, with the greedy probability
    1 - eps split equally among tied greedy actions."""
    raise NotImplementedError  # TODO


def double_q_update(Q1, Q2, s, a, r, s2, done, alpha, gamma, update_first, valid=None):
    """One double Q-learning update, in place. If update_first, Q1 is updated with the action
    chosen by Q1 and evaluated by Q2; otherwise the roles swap. Ties in the argmax go to the
    lowest index (np.argmax)."""
    raise NotImplementedError  # TODO


def run_control(env, method, episodes, alpha, eps, gamma, rng, valid=None):
    """Train with method in {"sarsa", "q_learning", "expected_sarsa", "double_q"} for the given
    number of episodes, with an eps-greedy behaviour policy (for double_q, eps-greedy on
    Q1 + Q2, and update_first = rng.random() < 0.5). `valid(s)`, if given, lists the legal
    actions in s. Returns (Q, total reward of each episode, first action of each episode);
    for double_q, Q is Q1 + Q2."""
    raise NotImplementedError  # TODO
