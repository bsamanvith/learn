import itertools
import numpy as np

rng = np.random.default_rng(0)

# 1. A tiny "language model": responses are 4 tokens from {a, b, c, d}, 256 possible responses,
# generated autoregressively. The policy has one vector of 4 logits for every prefix
# (1 + 4 + 16 + 64 = 85 prefixes), so it can represent any distribution over responses.
TOK, L = "abcd", 4
Y = ["".join(t) for t in itertools.product(TOK, repeat=L)]
prefixes = ["".join(t) for n in range(L) for t in itertools.product(TOK, repeat=n)]
pid = {p: i for i, p in enumerate(prefixes)}
steps_of = np.array([[pid[y[:i]] * 4 + TOK.index(y[i]) for i in range(L)] for y in Y])   # (256, 4)

def seq_logprob(theta):
    """log pi(y) for all 256 responses: the sum of the 4 token log-probabilities."""
    lp = theta - np.log(np.exp(theta).sum(1, keepdims=True))                      # (85, 4)
    return lp.reshape(-1)[steps_of].sum(1)

def grad_theta(theta, w):
    """Gradient of sum_y w(y) log pi(y) with respect to the logits theta."""
    p = np.exp(theta - np.log(np.exp(theta).sum(1, keepdims=True)))
    g = np.zeros(theta.size)
    np.add.at(g, steps_of.reshape(-1), np.repeat(w, L))                            # d/d logit of the token taken
    counts = np.zeros(len(prefixes)); np.add.at(counts, steps_of.reshape(-1) // 4, np.repeat(w, L))
    return g.reshape(theta.shape) - counts[:, None] * p                             # minus the softmax expectation

# The reference ("SFT") model: random logits, so some responses are much likelier than others.
theta_ref = rng.normal(0, 1.0, (len(prefixes), 4))
lp_ref = seq_logprob(theta_ref); p_ref = np.exp(lp_ref)

# 2. Hidden human taste: a reward the model never sees. People like 'a's, dislike a token
# repeated twice in a row, and like responses that end in 'd'.
def true_reward(y):
    return y.count("a") - 1.5 * sum(y[i] == y[i + 1] for i in range(L - 1)) + 1.0 * (y[-1] == "d")
r_true = np.array([true_reward(y) for y in Y])
print(f"reference model: E[true reward] = {p_ref @ r_true:.3f}; best possible response "
      f"{Y[r_true.argmax()]!r} with reward {r_true.max():.1f}")

# 3. Preference data: pairs of responses sampled from the reference model, labelled by people
# who follow the Bradley-Terry model, P(y1 preferred to y2) = sigmoid(r(y1) - r(y2)).
n_pairs = 2000
i1, i2 = rng.choice(256, n_pairs, p=p_ref), rng.choice(256, n_pairs, p=p_ref)
keep = i1 != i2; i1, i2 = i1[keep], i2[keep]
pref = rng.random(len(i1)) < 1 / (1 + np.exp(-(r_true[i1] - r_true[i2])))
win, lose = np.where(pref, i1, i2), np.where(pref, i2, i1)
print(f"{len(win)} preference pairs; the true reward agrees with {np.mean(r_true[win] > r_true[lose]):.3f} of the labels"
      f" ({np.mean(r_true[win] == r_true[lose]):.3f} are ties)")

# 4. A reward model: linear in which token sits at which position (16 features). It cannot
# express "repeated token", so it is wrong in a systematic way, like any real reward model.
phi = np.array([[float(y[i] == t) for i in range(L) for t in TOK] for y in Y])
w = np.zeros(16)
for _ in range(3000):                                           # logistic regression on differences
    d = phi[win] - phi[lose]
    s = 1 / (1 + np.exp(-(d @ w)))
    w += 0.5 * (d.T @ (1 - s) / len(d) - 1e-3 * w)
r_hat = phi @ w
print(f"reward model: train accuracy {np.mean(r_hat[win] > r_hat[lose]):.3f},"
      f" correlation with the true reward under the reference model "
      f"{np.cov(r_hat, r_true, aweights=p_ref)[0, 1] / np.sqrt(np.cov(r_hat, aweights=p_ref) * np.cov(r_true, aweights=p_ref)):.3f}")

# 5. The KL-regularised objective  max_pi E_pi[r_hat] - beta KL(pi || pi_ref)  has the
# closed-form solution pi_beta(y) = pi_ref(y) exp(r_hat(y) / beta) / Z(beta). Sweep beta.
def tilt(r, beta):
    lp = lp_ref + r / beta
    p = np.exp(lp - lp.max()); return p / p.sum()
kl = lambda p: np.sum(p * (np.log(p + 1e-300) - lp_ref))
print("   beta   KL(nats)  E[r_hat]  E[r_true]")
for beta in [10, 2, 1, 0.5, 0.25, 0.1, 0.03, 0.01]:
    p = tilt(r_hat, beta)
    print(f"{beta:7.2f} {kl(p):9.3f} {p @ r_hat:9.3f} {p @ r_true:10.3f}")
print(f"the reward model's favourite response: {Y[r_hat.argmax()]!r}, r_hat {r_hat.max():.2f}, true reward {r_true[r_hat.argmax()]:.1f}")

# 6. RLHF by policy gradient: optimise the same objective with samples, as PPO does, using the
# score-function gradient with the KL folded into the reward, r_hat(y) - beta log(pi/pi_ref).
beta = 0.25
theta = theta_ref.copy()
for it in range(6000):
    lp = seq_logprob(theta); p = np.exp(lp)
    s = rng.choice(256, 256, p=p)
    reward = r_hat[s] - beta * (lp[s] - lp_ref[s])
    adv = reward - reward.mean()                                # a baseline: subtract the batch mean
    wts = np.bincount(s, weights=adv, minlength=256) / len(s)
    theta += 1.0 * grad_theta(theta, wts)
p_rl = np.exp(seq_logprob(theta)); p_star = tilt(r_hat, beta)
print(f"policy gradient (beta = {beta}): KL to pi_ref {kl(p_rl):.3f}, E[r_hat] {p_rl @ r_hat:.3f},"
      f" E[r_true] {p_rl @ r_true:.3f}; KL(pi_rl || pi_beta) = {np.sum(p_rl * np.log(p_rl / p_star)):.4f}")

# 7. DPO: skip the reward model and fit the policy directly to the preferences, with
# loss = -log sigmoid(beta [log pi(yw)/pi_ref(yw) - log pi(yl)/pi_ref(yl)]).
beta, theta, dpo_track = 0.25, theta_ref.copy(), []
for epoch in range(1, 32001):
    ratio = seq_logprob(theta) - lp_ref
    h = beta * (ratio[win] - ratio[lose])
    g = beta * (1 - 1 / (1 + np.exp(-h))) / len(win)          # d(-loss)/dh, times dh/d(ratio) = beta
    wts = np.bincount(win, weights=g, minlength=256) - np.bincount(lose, weights=g, minlength=256)
    theta += 10.0 * grad_theta(theta, wts)                       # full-batch gradient ascent on -loss
    if epoch in (30, 100, 300, 1000, 3000, 10000, 32000):
        p = np.exp(seq_logprob(theta)); dpo_track.append((kl(p), p @ r_true))
    if epoch in (100, 1000, 10000, 32000):
        print(f"DPO (beta = {beta}), {epoch:5d} epochs: loss {np.mean(np.logaddexp(0, -h)):.4f},"
              f" KL to pi_ref {kl(p):.3f}, E[r_true] {p @ r_true:.3f}")

# reference model: E[true reward] = -0.088; best possible response 'abad' with reward 3.0
# 1973 preference pairs; the true reward agrees with 0.730 of the labels (0.117 are ties)
# reward model: train accuracy 0.695, correlation with the true reward under the reference model 0.725
#    beta   KL(nats)  E[r_hat]  E[r_true]
#   10.00     0.003    -0.126      0.006
#    2.00     0.084     0.141      0.366
#    1.00     0.332     0.473      0.759
#    0.50     1.210     1.065      1.277
#    0.25     3.420     1.830      1.395
#    0.10     6.681     2.408      0.894
#    0.03     6.841     2.421      0.840
#    0.01     6.953     2.423      0.661
# the reward model's favourite response: 'aaaa', r_hat 2.43, true reward -0.5
# policy gradient (beta = 0.25): KL to pi_ref 3.475, E[r_hat] 1.840, E[r_true] 1.396; KL(pi_rl || pi_beta) = 0.0156
# DPO (beta = 0.25),   100 epochs: loss 0.5127, KL to pi_ref 1.036, E[r_true] 1.514
# DPO (beta = 0.25),  1000 epochs: loss 0.4220, KL to pi_ref 2.612, E[r_true] 2.305
# DPO (beta = 0.25), 10000 epochs: loss 0.3890, KL to pi_ref 4.875, E[r_true] 2.891
# DPO (beta = 0.25), 32000 epochs: loss 0.3848, KL to pi_ref 4.554, E[r_true] 2.624
