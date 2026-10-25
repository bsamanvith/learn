"""Assignment A8: preference fine-tuning (reward models, RLHF, DPO) and LoRA, in NumPy.

Fill in every function and method marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

The setting is a small language model over a vocabulary of 6 tokens. A prompt is one digit,
x in 0..9, and a response is 3 tokens, so each prompt has 6^3 = 216 possible responses, few
enough to enumerate. The policy is autoregressive: an MLP reads a one-hot code of the prompt
and of the tokens generated so far and outputs logits for the next token. The MLP, Adam and
the data are given; the probability calculations, the losses and the training steps are yours.
"""
import itertools
import numpy as np

V, L, N_PROMPTS = 6, 3, 10                                 # vocabulary, response length, prompts
RESPONSES = np.array(list(itertools.product(range(V), repeat=L)))   # (216, 3), every response


# ---------------------------------------------------------------- given: layers and the optimiser

class Linear:
    def __init__(self, n_in, n_out, rng):
        self.params = {"W": rng.standard_normal((n_in, n_out)) * np.sqrt(1 / n_in), "b": np.zeros(n_out)}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, g):
        self.grads = {"W": self.x.T @ g, "b": g.sum(0)}
        return g @ self.params["W"].T


class Tanh:
    params = {}

    def forward(self, x):
        self.y = np.tanh(x); return self.y

    def backward(self, g):
        self.grads = {}; return g * (1 - self.y ** 2)


class Adam:
    """Adam over the params/grads dictionaries of a list of layers. step(sign=-1) ascends."""
    def __init__(self, layers, lr):
        self.layers, self.lr, self.t = layers, lr, 0
        self.m = [{k: np.zeros_like(p) for k, p in l.params.items()} for l in layers]
        self.v = [{k: np.zeros_like(p) for k, p in l.params.items()} for l in layers]

    def step(self, sign=1.0):
        self.t += 1
        for l, ms, vs in zip(self.layers, self.m, self.v):
            for k, p in l.params.items():
                g = sign * l.grads[k]
                ms[k] = 0.9 * ms[k] + 0.1 * g; vs[k] = 0.999 * vs[k] + 0.001 * g * g
                p -= self.lr * (ms[k] / (1 - 0.9 ** self.t)) / (np.sqrt(vs[k] / (1 - 0.999 ** self.t)) + 1e-8)


def features(x, prefix):
    """Input to the policy network for predicting token len(prefix) of the response: one-hot
    prompt (10), one-hot position (3), and one-hot codes of the previous tokens, zero where
    absent (3 x 6 = 18). x is (n,), prefix is (n, i) with i < L. Returns (n, 31)."""
    n, i = prefix.shape
    f = np.zeros((n, N_PROMPTS + L + L * V))
    f[np.arange(n), x] = 1
    f[:, N_PROMPTS + i] = 1
    for j in range(i):
        f[np.arange(n), N_PROMPTS + L + j * V + prefix[:, j]] = 1
    return f


# ---------------------------------------------------------------- Part 1: LoRA

class LoRALinear:
    """A frozen Linear layer W0, b0 plus a trainable low-rank update:
        y = x W0 + b0 + (alpha / r) * x A B,   A: (n_in, r), B: (r, n_out).
    A starts random (std 1 / sqrt(n_in)) and B at zero. Only A and B are in self.params, so an
    optimiser built on this layer trains only them."""
    def __init__(self, base, r, alpha, rng):
        self.W0, self.b0 = base.params["W"].copy(), base.params["b"].copy()
        n_in, n_out = self.W0.shape
        self.scale = alpha / r
        self.params = {"A": rng.standard_normal((n_in, r)) / np.sqrt(n_in), "B": np.zeros((r, n_out))}

    def forward(self, x):
        raise NotImplementedError  # TODO

    def backward(self, g):
        raise NotImplementedError  # TODO

    def merged(self):
        """A plain Linear with W = W0 + (alpha / r) A B and b = b0, computing the same function."""
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: the policy's probabilities

class Policy:
    """Autoregressive policy: logits for token i = net(features(x, y[:, :i])), with
    net = Linear(31, H) -> Tanh -> Linear(H, V). (given)"""
    def __init__(self, H, rng):
        self.layers = [Linear(N_PROMPTS + L + L * V, H, rng), Tanh(), Linear(H, V, rng)]

    def logits(self, f):
        for l in self.layers:
            f = l.forward(f)
        return f

    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)

    def trainable(self):
        return [l for l in self.layers if l.params]


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def seq_logprob(policy, x, y):
    """log pi(y | x) for each row: the sum over the L positions of log softmax(logits)[y_i].
    x (n,), y (n, L). Returns (n,)."""
    raise NotImplementedError  # TODO


def seq_logprob_backward(policy, x, y, w):
    """Accumulate into the policy's trainable layers the gradient of sum_n w[n] log pi(y_n | x_n)
    (w is (n,)). Run one forward and backward pass per position and ADD the gradients of the
    positions: when this returns, each trainable layer's grads hold the total."""
    raise NotImplementedError  # TODO


def all_logprobs(policy):
    """log pi(y | x) for every prompt and every response: an array (N_PROMPTS, 216). (given)"""
    x = np.repeat(np.arange(N_PROMPTS), len(RESPONSES))
    y = np.tile(RESPONSES, (N_PROMPTS, 1))
    return seq_logprob(policy, x, y).reshape(N_PROMPTS, len(RESPONSES))


def sample(policy, x, rng):
    """Sample one response for each prompt in x, token by token. Returns (n, L) ints."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 3: reward models

def bradley_terry_loss(r_w, r_l):
    """Mean of -log sigmoid(r_w - r_l) over pairs, computed stably. Returns (loss, dloss/dr_w,
    dloss/dr_l), the gradients as arrays like r_w."""
    raise NotImplementedError  # TODO


def fit_reward_model(phi_w, phi_l, steps=2000, lr=0.5, l2=1e-3):
    """A linear reward model r(y) = phi(y) . w fitted to preference pairs by gradient descent on
    the Bradley-Terry loss plus (l2 / 2) |w|^2, starting from w = 0. phi_w, phi_l are the feature
    vectors (n, k) of the winners and losers. Returns w (k,)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 4: the KL-regularised optimum and RLHF

def kl_optimal_policy(logp_ref, r, beta):
    """The maximiser of E_pi[r] - beta KL(pi || pi_ref), separately for each prompt (row):
    pi(y) proportional to pi_ref(y) exp(r(y) / beta). logp_ref and r are (N_PROMPTS, 216).
    Returns probabilities (N_PROMPTS, 216); stable for small beta."""
    raise NotImplementedError  # TODO


def kl_divergence(p, logp_ref):
    """KL(p || pi_ref) for each row, with 0 log 0 = 0. Returns (N_PROMPTS,)."""
    raise NotImplementedError  # TODO


def rlhf_step(policy, ref_policy, reward_fn, opt, beta, group, rng):
    """One policy-gradient step on E[r] - beta KL(pi || pi_ref), GRPO-style: for every prompt,
    sample `group` responses from the policy, score each with
        R = reward_fn(x, y) - beta * (log pi(y|x) - log pi_ref(y|x)),
    and use as its advantage R minus the mean of R over that prompt's group. Accumulate the
    gradient of mean(advantage * log pi(y|x)) with seq_logprob_backward, and take one ASCENT
    step with opt.step(sign=-1). Returns the mean of reward_fn over the samples."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 5: DPO

def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta):
    """Mean DPO loss -log sigmoid(beta [(lp_w - ref_w) - (lp_l - ref_l)]) over pairs, where lp_*
    are the policy's log-probabilities of the winners and losers and ref_* the reference's.
    Returns (loss, dloss/dlp_w, dloss/dlp_l)."""
    raise NotImplementedError  # TODO


def ipo_loss(lp_w, lp_l, ref_w, ref_l, tau):
    """Mean IPO loss (Azar et al., 2023): (Delta - 1 / (2 tau))^2 with
    Delta = (lp_w - ref_w) - (lp_l - ref_l). Returns (loss, dloss/dlp_w, dloss/dlp_l)."""
    raise NotImplementedError  # TODO


def preference_step(policy, x, y_w, y_l, ref_w, ref_l, opt, loss_fn, coef):
    """One gradient-DESCENT step of a pairwise loss (dpo_loss or ipo_loss, with coef = beta or
    tau) on a batch of pairs: compute the policy's log-probabilities of winners and losers,
    the loss and its gradients with respect to them, accumulate the parameter gradient of
    sum(g_w * lp_w + g_l * lp_l) with seq_logprob_backward, and call opt.step(). Note that
    seq_logprob_backward replaces the grads, so combine the winners and losers in one call.
    Returns the loss."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- given: the task

def true_reward(x, y):
    """Hidden human taste (the checker and your experiments may use it; your methods may not).
    People like responses whose tokens sum to the prompt (mod 6), like token 0 at the start,
    and dislike a token repeated twice in a row."""
    y = np.atleast_2d(y)
    return (2.0 * (y.sum(1) % V == np.asarray(x) % V) + 1.0 * (y[:, 0] == 0)
            - 1.5 * (y[:, 1:] == y[:, :-1]).sum(1))


def make_reference(rng, H=64, scale=3.0):
    """A reference ("SFT") policy: the MLP with random weights, its output layer scaled up so
    that each prompt has its own uneven distribution over responses. (given)"""
    pol = Policy(H, rng)
    pol.layers[0].params["W"] *= 2.0
    pol.layers[2].params["W"] *= scale
    return pol


def response_features(x, y):
    """Features for a linear reward model: which token is at which position (18), and whether
    the tokens sum to the prompt mod 6 (1). (given) Deliberately blind to repeated tokens."""
    y = np.atleast_2d(y)
    f = np.zeros((len(y), L * V + 1))
    for i in range(L):
        f[np.arange(len(y)), i * V + y[:, i]] = 1
    f[:, L * V] = y.sum(1) % V == np.asarray(x) % V
    return f
