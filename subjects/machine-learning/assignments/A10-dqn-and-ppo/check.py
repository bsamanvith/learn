"""Autograder for assignment A10. Usage: python3 check.py [your_file.py]

The pieces are checked against hand-computed values and finite differences, GAE against a
brute-force sum, the PPO update on a synthetic batch, and the two training loops by whether
they learn: DQN on CartPole and PPO on a one-step continuous task. Takes about two minutes."""
import importlib.util
import sys
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

results = []


def test(name, hint=""):
    def wrap(fn):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                fn()
            results.append((name, True, ""))
        except NotImplementedError:
            results.append((name, None, "not implemented yet"))
        except AssertionError as e:
            results.append((name, False, (str(e) or "wrong answer") + (f"\n        hint: {hint}" if hint else "")))
        except Exception:
            results.append((name, False, "crashed:\n" + "".join(traceback.format_exc(limit=-2))))
        return fn
    return wrap


def close(a, b, tol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: shape {a.shape}, expected {b.shape}"
    err = np.max(np.abs(a - b)) if a.size else 0
    assert err <= tol, f"{what}: off by {err:.3g} (tolerance {tol:g})"


def numgrad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h; up = f()
        x[i] = old - h; dn = f()
        x[i] = old
        g[i] = (up - dn) / (2 * h)
    return g


# ---------------------------------------------------------------- Part 1: DQN

@test("ReplayBuffer", "keep a write position that wraps around at capacity; n counts stored transitions up to capacity")
def _():
    buf = S.ReplayBuffer(5, 2)
    for k in range(7):
        buf.add(np.array([k, k]), k % 2, float(k), np.array([k + 1, k + 1]), k == 6)
    assert buf.n == 5, f"after 7 adds to a buffer of capacity 5, n should be 5, got {buf.n}"
    s, a, r, s2, d = buf.sample(400, np.random.default_rng(0))
    assert s.shape == (400, 2) and s2.shape == (400, 2) and a.shape == r.shape == d.shape == (400,), "sample shapes"
    assert set(np.unique(r)) == {2.0, 3.0, 4.0, 5.0, 6.0}, f"the stored rewards should be 2..6 (0 and 1 overwritten), sampled {sorted(set(np.unique(r)))}"
    close(s[:, 0], r, 0, "s matches r for each sampled transition")
    close(d, (r == 6).astype(float), 0, "terminated flags")


@test("dqn_targets", "y = r + gamma (1 - terminated) * next value; Double DQN picks the action with the online net")
def _():
    r = np.array([1.0, 1.0, 0.5]); d = np.array([0.0, 1.0, 0.0])
    qt = np.array([[1.0, 3.0], [5.0, 6.0], [2.0, 0.0]]); qo = np.array([[4.0, 0.0], [0.0, 1.0], [0.0, 9.0]])
    close(S.dqn_targets(r, d, qt, 0.9), [1 + 0.9 * 3, 1, 0.5 + 0.9 * 2], 1e-12, "DQN targets")
    close(S.dqn_targets(r, d, qt, 0.9, qo), [1 + 0.9 * 1, 1, 0.5 + 0.9 * 0], 1e-12, "Double DQN targets")


@test("huber_loss_grad", "the gradient of the Huber loss is the error clipped to [-1, 1]; divide by the batch size")
def _():
    rng = np.random.default_rng(1)
    q = rng.normal(0, 2, (6, 3)); a = rng.integers(0, 3, 6); y = rng.normal(0, 2, 6)
    loss, g = S.huber_loss_grad(q, a, y)
    err = q[np.arange(6), a] - y
    close(loss, np.mean(np.where(np.abs(err) <= 1, 0.5 * err ** 2, np.abs(err) - 0.5)), 1e-12, "loss")
    close(g, numgrad(lambda: S.huber_loss_grad(q, a, y)[0], q), 1e-6, "gradient")


@test("polyak_update", "update the target's arrays in place: pt += tau * (po - pt)")
def _():
    rng = np.random.default_rng(2)
    on = S.MLP([3, 4, 2], rng); tg = S.MLP([3, 4, 2], rng)
    before = [p.copy() for p in tg.params]; ids = [id(p) for p in tg.params]
    S.polyak_update(tg, on, 0.1)
    for p, b, o in zip(tg.params, before, on.params):
        close(p, 0.9 * b + 0.1 * o, 1e-12, "target parameters")
    assert [id(p) for p in tg.params] == ids, "update the arrays in place (an Adam built on them must see the change)"


@test("train_dqn learns CartPole", "if the pieces pass: one gradient step per environment step, polyak after each, epsilon from 1 to 0.05")
def _():
    rets = [np.array(S.train_dqn(seed, steps=30_000)[0]) for seed in (0, 1)]
    late = [r[r[:, 0] >= 10_000, 1].mean() for r in rets]
    assert min(late) > 60, f"mean episode return over steps 10,000-30,000, seeds 0 and 1: {np.round(late, 1)}; a working DQN gets about 90 to 140 (random play gets about 22)"


# ---------------------------------------------------------------- Part 2: PPO

@test("gaussian_logprob and its gradients", "log N = sum over dimensions of -z^2/2 - log_std - log(2 pi)/2, z = (a - mu)/std")
def _():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (5, 2)); mu = rng.normal(0, 1, (5, 2)); ls = np.array([-0.3, 0.4])
    lp = S.gaussian_logprob(a, mu, ls)
    std = np.exp(ls)
    close(lp, np.sum(-0.5 * ((a - mu) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi)), 1), 1e-12, "log-probabilities")
    dmu, dls = S.gaussian_logprob_grads(a, mu, ls)
    close(dmu, numgrad(lambda: S.gaussian_logprob(a, mu, ls).sum(), mu), 1e-6, "gradient with respect to mu")
    close(dls.sum(0), numgrad(lambda: S.gaussian_logprob(a, mu, ls).sum(), ls), 1e-6, "gradient with respect to log_std (summed over the batch)")


@test("gae", "go backwards; cut the recursion where ended[t] = 1; bootstrap from next_values unless terminated[t] = 1")
def _():
    rng = np.random.default_rng(4)
    T, gamma, lam = 12, 0.9, 0.8
    r = rng.normal(0, 1, T); v = rng.normal(0, 1, T); nv = rng.normal(0, 1, T)
    term = np.zeros(T); end = np.zeros(T)
    term[3] = end[3] = 1; end[7] = 1                              # a termination at 3, a truncation at 7
    adv, ret = S.gae(r, v, nv, term, end, gamma, lam)
    expect = np.zeros(T)
    for t in range(T):                                            # brute force: sum of discounted deltas until the episode ends
        k, w = t, 1.0
        while True:
            expect[t] += w * (r[k] + gamma * (1 - term[k]) * nv[k] - v[k])
            if end[k] or k == T - 1:
                break
            w *= gamma * lam; k += 1
    close(adv, expect, 1e-12, "advantages")
    close(ret, expect + v, 1e-12, "returns (advantages + values)")


@test("ppo_clip_objective", "the gradient is ratio * A / B where the min picks the unclipped term, and 0 where it picks the clipped one")
def _():
    lo = np.log(np.array([0.5, 0.5, 0.5, 0.5, 0.2]))
    ln = np.log(np.array([0.75, 0.75, 0.3, 0.3, 0.21]))          # ratios 1.5, 1.5, 0.6, 0.6, 1.05
    A = np.array([2.0, -2.0, 2.0, -2.0, 1.0])
    obj, g = S.ppo_clip_objective(ln, lo, A, 0.2)
    close(obj, np.mean([2.4, -3.0, 1.2, -1.6, 1.05]), 1e-12, "objective")
    close(g, numgrad(lambda: S.ppo_clip_objective(ln, lo, A, 0.2)[0], ln), 1e-6, "gradient with respect to logp_new")
    assert g[0] == 0 and g[3] == 0, "the gradient must be exactly zero where the clipped term is chosen"


def toy_batch(rng, n=2048):
    pol = S.GaussianPolicy(2, 1, rng)
    obs = rng.uniform(-1, 1, (n, 2))
    mu = pol.net.forward(obs); act = mu + np.exp(pol.log_std) * rng.standard_normal(mu.shape)
    logp = S.gaussian_logprob(act, mu, pol.log_std)
    target = 0.8 * obs[:, :1] - 0.3 * obs[:, 1:]
    adv = -((act - target) ** 2)[:, 0]
    return pol, {"obs": obs, "act": act, "logp": logp, "adv": adv - adv.mean(), "ret": adv}


@test("ppo_update", "minibatch advantages normalised; ascend the objective for the policy (mu and log_std), descend the MSE for the critic")
def _():
    rng = np.random.default_rng(5)
    pol, batch = toy_batch(rng)
    critic = S.MLP([2, 32, 32, 1], rng)
    opt_pi = S.Adam(pol.params(), 3e-4, max_norm=0.5); opt_v = S.Adam(critic.params, 1e-3, max_norm=0.5)
    mse0 = np.mean((critic.forward(batch["obs"])[:, 0] - batch["ret"]) ** 2)
    kl = S.ppo_update(pol, critic, batch, rng, opt_pi, opt_v, epochs=4, minibatch=64, eps=0.2)
    lp = S.gaussian_logprob(batch["act"], pol.net.forward(batch["obs"]), pol.log_std)
    gain = np.mean(np.exp(lp - batch["logp"]) * batch["adv"])
    mse1 = np.mean((critic.forward(batch["obs"])[:, 0] - batch["ret"]) ** 2)
    assert gain > 0, f"the surrogate E[r A] should rise above 0 after the update, got {gain:.4f}"
    assert 0 < kl < 0.1, f"KL estimate {kl:.4f}: the policy should move, but not far"
    assert pol.log_std[0] < -0.5, "log_std should shrink here: every advantage is a negative squared distance from the target"
    assert mse1 < mse0, f"the critic's squared error should fall, went from {mse0:.4f} to {mse1:.4f}"


class Reach:
    """A one-step task: observe x in [-1, 1]^2, choose a, receive -(a - 0.8 x_1 + 0.3 x_2)^2."""
    obs_dim, act_dim = 2, 1

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.x = self.rng.uniform(-1, 1, 2); return self.x.copy()

    def step(self, a):
        a = float(np.asarray(a).reshape(-1)[0])
        r = -(a - 0.8 * self.x[0] + 0.3 * self.x[1]) ** 2
        return self.reset(), r, True, False


@test("train_ppo learns a continuous task", "if ppo_update passes: collect n_steps with the current policy, GAE with term and end flags, then update")
def _():
    rets = np.array(S.train_ppo(0, env_cls=Reach, total_steps=40_960, n_steps=1024, gamma=0.9, lam=0.95, lr=1e-3)[0])
    first, last = rets[:1024, 1].mean(), rets[-1024:, 1].mean()
    assert last > -0.05, f"mean reward in the first batch {first:.3f}, in the last {last:.3f}; a working PPO gets above -0.01 (the start is about -0.6)"


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
