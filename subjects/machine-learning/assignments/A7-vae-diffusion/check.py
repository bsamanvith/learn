"""Autograder for assignment A7. Usage: python3 check.py [your_file.py]

Values are compared with direct computations (numerical integration, Bayes' rule on a grid,
simulation of the forward chain), gradients with central finite differences, and the
samplers with an oracle noise predictor whose answers are exact."""
import importlib.util
import sys
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
rng = np.random.default_rng(7)

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


def numgrad(f, a, h=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old = a[idx]
        a[idx] = old + h; up = f()
        a[idx] = old - h; dn = f()
        a[idx] = old
        g[idx] = (up - dn) / (2 * h)
    return g


class Oracle:
    """A stand-in for the network when the data is the single point a: it returns the exact
    noise eps = (x_t - sqrt(abar_t) a) / sqrt(1 - abar_t), reading t from self.t."""
    def __init__(self, a, abar):
        self.a, self.abar, self.t, self.calls = a, abar, None, 0
    def forward(self, inp):
        self.calls += 1
        d = len(self.a)
        x = inp[:, :d]
        ab = self.abar[self.t - 1]
        return (x - np.sqrt(ab) * self.a) / np.sqrt(1 - ab)


# ---------------------------------------------------------------- Part 1

@test("gaussian_kl", "KL = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar); its gradients are mu and 0.5 * (exp(logvar) - 1)")
def _():
    mu, lv = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
    kl, gm, gl = S.gaussian_kl(mu, lv)
    assert np.shape(kl) == (4,), f"kl should have one value per row, got shape {np.shape(kl)}"
    z = mu[0] + np.exp(lv[0] / 2) * np.random.default_rng(0).standard_normal((400000, 3))
    mc = np.mean(np.sum(-0.5 * (z - mu[0]) ** 2 / np.exp(lv[0]) - lv[0] / 2 + 0.5 * z ** 2, 1))
    close(kl[0], mc, 0.02, "KL of row 0 against a Monte Carlo estimate")
    close(gm, numgrad(lambda: S.gaussian_kl(mu, lv)[0].sum(), mu), 1e-6, "dKL/dmu")
    close(gl, numgrad(lambda: S.gaussian_kl(mu, lv)[0].sum(), lv), 1e-6, "dKL/dlogvar")


@test("bernoulli_loglik", "x * logits - log(1 + exp(logits)), with np.logaddexp for stability; the gradient is x - sigmoid(logits)")
def _():
    x = rng.random((3, 5)); L = 3 * rng.standard_normal((3, 5))
    ll, g = S.bernoulli_loglik(x, L)
    p = 1 / (1 + np.exp(-L))
    close(ll, np.sum(x * np.log(p) + (1 - x) * np.log(1 - p), 1), 1e-10, "log-likelihood")
    close(g, numgrad(lambda: S.bernoulli_loglik(x, L)[0].sum(), L), 1e-6, "gradient")
    ll, g = S.bernoulli_loglik(np.array([[1.0, 0.0]]), np.array([[-1000.0, 1000.0]]))
    close(ll, [-2000.0], 1e-9, "log-likelihood at logits of size 1000")


@test("reparameterise", "z = mu + exp(logvar / 2) * eps")
def _():
    mu, lv, e = rng.standard_normal((3, 2)), rng.standard_normal((3, 2)), rng.standard_normal((3, 2))
    z, dmu, dlv = S.reparameterise(mu, lv, e)
    close(z, mu + np.exp(lv / 2) * e, 1e-12, "z")
    close(dmu, numgrad(lambda: S.reparameterise(mu, lv, e)[0].sum(), mu), 1e-6, "dz/dmu")
    close(dlv, numgrad(lambda: S.reparameterise(mu, lv, e)[0].sum(), lv), 1e-6, "dz/dlogvar")


@test("vae_elbo: value and gradients", "the decoder gets drec / N; the encoder gets the reparameterised gradient minus the KL gradient, both / N")
def _():
    r = np.random.default_rng(1)
    enc, dec = S.MLP([6, 8, 4], r), S.MLP([2, 8, 6], r)
    x = (r.random((5, 6)) < 0.5).astype(float); e = r.standard_normal((5, 2))
    elbo, rec, kl = S.vae_elbo(enc, dec, x, e)
    h = enc.forward(x); z = h[:, :2] + np.exp(h[:, 2:] / 2) * e
    L = dec.forward(z); p = 1 / (1 + np.exp(-L))
    want_rec = np.mean(np.sum(x * np.log(p) + (1 - x) * np.log(1 - p), 1))
    want_kl = np.mean(0.5 * np.sum(h[:, :2] ** 2 + np.exp(h[:, 2:]) - 1 - h[:, 2:], 1))
    close([elbo, rec, kl], [want_rec - want_kl, want_rec, want_kl], 1e-10, "(elbo, reconstruction, KL)")
    S.vae_elbo(enc, dec, x, e)
    grads = [g.copy() for g in enc.grads + dec.grads]
    for k, (p_, g) in enumerate(zip(enc.params + dec.params, grads)):
        for idx in [tuple(r.integers(0, s) for s in p_.shape) for _ in range(4)]:
            old = p_[idx]
            p_[idx] = old + 1e-6; up = S.vae_elbo(enc, dec, x, e)[0]
            p_[idx] = old - 1e-6; dn = S.vae_elbo(enc, dec, x, e)[0]
            p_[idx] = old
            num = (up - dn) / 2e-6
            assert abs(num - g[idx]) < 1e-5 * max(1, abs(num)), f"gradient of parameter {k} at {idx}: {g[idx]:.6g}, finite differences give {num:.6g}"


@test("iwae_bound approaches log p(x)", "average the weights in log space: max + log(mean(exp(lw - max)))")
def _():
    r = np.random.default_rng(2)
    enc, dec = S.MLP([4, 16, 2], r), S.MLP([1, 16, 4], r)
    enc.params[-2][:] = 0; enc.params[-1][:] = [0.3, np.log(3)]        # q(z | x) = N(0.3, 3): a broad proposal
    x = (r.random((6, 4)) < 0.5).astype(float)
    zs = np.linspace(-10, 10, 4001)
    ll = np.array([S.bernoulli_loglik(np.repeat(xi[None], len(zs), 0), dec.forward(zs[:, None]))[0] for xi in x])
    logp = np.log(np.sum(np.exp(ll - 0.5 * zs ** 2) / np.sqrt(2 * np.pi), 1) * (zs[1] - zs[0]))
    L1 = np.mean([S.iwae_bound(enc, dec, x, 1, r) for _ in range(300)], 0)
    L1000 = S.iwae_bound(enc, dec, x, 3000, r)
    assert np.shape(L1000) == (6,), f"expected one value per example, got shape {np.shape(L1000)}"
    assert np.all(L1 < logp + 0.02), f"the K = 1 bound {L1.round(3)} must lie below log p(x) {logp.round(3)}"
    close(L1000, logp, 0.03, "the K = 3000 estimate against log p(x) computed by integration")
    big = S.iwae_bound(enc, S.MLP([1, 16, 4], np.random.default_rng(3)), np.ones((1, 4)) * 1, 5, r)
    assert np.all(np.isfinite(big)), "iwae_bound returned a non-finite value"


# ---------------------------------------------------------------- Part 2

@test("make_schedule", "abar is the cumulative product of alpha = 1 - beta")
def _():
    beta, alpha, abar = S.make_schedule(5, 0.1, 0.5)
    close(beta, [0.1, 0.2, 0.3, 0.4, 0.5], 1e-12, "beta")
    close(alpha, [0.9, 0.8, 0.7, 0.6, 0.5], 1e-12, "alpha")
    close(abar, np.cumprod([0.9, 0.8, 0.7, 0.6, 0.5]), 1e-12, "abar")


@test("q_sample matches the forward chain", "x_t = sqrt(abar[t - 1]) x0 + sqrt(1 - abar[t - 1]) eps: step t is at index t - 1")
def _():
    beta, alpha, abar = S.make_schedule(20, 0.01, 0.3)
    n = 200000
    x = np.full((n, 1), 1.5)
    r = np.random.default_rng(4)
    for t in range(1, 21):
        x = np.sqrt(alpha[t - 1]) * x + np.sqrt(beta[t - 1]) * r.standard_normal(x.shape)
        if t in (1, 7, 20):
            q = S.q_sample(np.full((n, 1), 1.5), np.full(n, t), r.standard_normal((n, 1)), abar)
            close([q.mean(), q.var()], [x.mean(), x.var()], 0.01, f"mean and variance at step {t}")


@test("q_posterior is Bayes' rule", "precision alpha_t / beta_t + 1 / (1 - abar_{t-1}); see lesson 24")
def _():
    beta, alpha, abar = S.make_schedule(30, 0.01, 0.2)
    for t, x0, xt in [(2, 1.0, 0.5), (15, -0.7, 0.9), (30, 2.0, -1.0)]:
        g = np.linspace(-8, 8, 200001)
        logw = -(xt - np.sqrt(alpha[t - 1]) * g) ** 2 / (2 * beta[t - 1]) - (g - np.sqrt(abar[t - 2]) * x0) ** 2 / (2 * (1 - abar[t - 2]))
        w = np.exp(logw - logw.max()); w /= w.sum()
        m, v = np.sum(w * g), np.sum(w * (g - np.sum(w * g)) ** 2)
        mean, var = S.q_posterior(np.array([[x0]]), np.array([[xt]]), np.array([t]), beta, abar)
        close([np.ravel(mean)[0], np.ravel(var)[0]], [m, v], 1e-6, f"posterior mean and variance at t = {t}")


@test("ddpm_loss: value and gradients", "mix with q_sample, predict with predict_eps, and backpropagate 2 (pred - eps) / N")
def _():
    r = np.random.default_rng(5)
    _, _, abar = S.make_schedule(10, 0.01, 0.5)
    net = S.MLP([2 + 16, 8, 2], r)
    x0, t, e = r.standard_normal((6, 2)), r.integers(1, 11, 6), r.standard_normal((6, 2))
    loss = S.ddpm_loss(net, x0, t, e, abar, 10)
    xt = np.sqrt(abar[t - 1])[:, None] * x0 + np.sqrt(1 - abar[t - 1])[:, None] * e
    pred = net.forward(np.concatenate([xt, S.time_features(t, 10)], 1))
    close(loss, np.mean(np.sum((pred - e) ** 2, 1)), 1e-12, "loss")
    S.ddpm_loss(net, x0, t, e, abar, 10)
    for k, p_ in enumerate(net.params):
        g = net.grads[k].copy()
        num = numgrad(lambda: S.ddpm_loss(net, x0, t, e, abar, 10), p_)
        close(g, num, 1e-5 * max(1, np.abs(num).max()), f"gradient of parameter {k}")


@test("p_sample_step with an exact noise predictor", "mean = (x_t - beta_t / sqrt(1 - abar_t) eps) / sqrt(alpha_t); add sqrt(beta_t) z except at t = 1")
def _():
    beta, alpha, abar = S.make_schedule(20, 0.01, 0.3)
    a = np.array([0.7, -1.2])
    orc = Oracle(a, abar)
    r = np.random.default_rng(6)
    for t in [20, 9, 2]:
        xt = r.standard_normal((4, 2)); z = r.standard_normal((4, 2)); orc.t = t
        out = S.p_sample_step(orc, xt, t, z, beta, alpha, abar, 20)
        mean, _ = S.q_posterior(np.repeat(a[None], 4, 0), xt, np.full(4, t), beta, abar)
        close(out, np.asarray(mean) + np.sqrt(beta[t - 1]) * z, 1e-10, f"step from t = {t}")
    orc.t = 1
    close(S.p_sample_step(orc, r.standard_normal((3, 2)), 1, r.standard_normal((3, 2)), beta, alpha, abar, 20),
          np.repeat(a[None], 3, 0), 1e-10, "the last step (t = 1) with exact noise must return the data point, with no noise added")


@test("ddim_step re-noises its estimate of x_0", "x0_hat = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t); then sqrt(abar_s) x0_hat + sqrt(1 - abar_s) eps")
def _():
    _, _, abar = S.make_schedule(50, 0.001, 0.2)
    a = np.array([1.5, 0.3])
    orc = Oracle(a, abar)
    r = np.random.default_rng(8)
    e = r.standard_normal((5, 2))
    xt = np.sqrt(abar[39]) * a + np.sqrt(1 - abar[39]) * e
    orc.t = 40
    close(S.ddim_step(orc, xt, 40, 12, abar, 50), np.sqrt(abar[11]) * a + np.sqrt(1 - abar[11]) * e, 1e-10, "step from 40 to 12")
    close(S.ddim_step(orc, xt, 40, 0, abar, 50), np.repeat(a[None], 5, 0), 1e-10, "step from 40 to 0")


@test("sample, ancestral and DDIM, with an exact noise predictor", "loop over t = T, ..., 1; for DDIM use the pairs of consecutive entries of np.linspace(T, 0, ddim_steps + 1)")
def _():
    sched = S.make_schedule(30, 0.01, 0.4)
    a = np.array([-0.4, 0.9])

    class Tracking(Oracle):
        """Reads t from the time features, by matching them against every step's."""
        def forward(self, inp):
            feats = inp[:, len(self.a):]
            table = S.time_features(np.arange(1, 31), 30)
            self.t = int(np.argmin(np.sum((table - feats[0]) ** 2, 1))) + 1
            return super().forward(inp)

    orc = Tracking(a, sched[2])
    out = S.sample(orc, 7, 2, sched, np.random.default_rng(9))
    close(out, np.repeat(a[None], 7, 0), 1e-8, "ancestral samples of a single data point")
    assert orc.calls == 30, f"ancestral sampling with T = 30 should call the network 30 times, it called it {orc.calls} times"
    orc.calls = 0
    out = S.sample(orc, 7, 2, sched, np.random.default_rng(9), ddim_steps=6)
    close(out, np.repeat(a[None], 7, 0), 1e-8, "DDIM samples of a single data point")
    assert orc.calls == 6, f"DDIM with 6 steps should call the network 6 times, it called it {orc.calls} times"


@test("end to end: a diffusion model learns two moons", "if the other checks pass, look at how you index the schedule (step t is index t - 1)")
def _():
    r = np.random.default_rng(10)
    sched = S.make_schedule(50, 1e-4, 0.25)
    net = S.MLP([2 + 16, 128, 128, 2], r)
    opt = S.Adam([net], 2e-3)
    for _ in range(2500):
        x0 = S.two_moons(256, r)
        S.ddpm_loss(net, x0, r.integers(1, 51, 256), r.standard_normal((256, 2)), sched[2], 50)
        opt.step()
    X = S.sample(net, 1000, 2, sched, r)
    ref = S.two_moons(5000, r)
    d = np.sqrt(np.min(np.sum((X[:, None] - ref[None]) ** 2, 2), 1))
    assert np.median(d) < 0.06, f"median distance of samples to the data is {np.median(d):.3f}; expected below 0.06"


@test("end to end: a VAE's ELBO improves", "train with Adam.step(sign=-1) to ascend the ELBO")
def _():
    r = np.random.default_rng(11)
    protos = (r.random((4, 20)) < 0.5).astype(float)
    def data(n):
        X = protos[r.integers(0, 4, n)]
        flip = r.random(X.shape) < 0.05
        return np.abs(X - flip)
    enc, dec = S.MLP([20, 32, 4], r), S.MLP([2, 32, 20], r)
    opt = S.Adam([enc, dec], 3e-3)
    for _ in range(800):
        S.vae_elbo(enc, dec, data(64), r.standard_normal((64, 2))); opt.step(sign=-1)
    elbo = np.mean([S.vae_elbo(enc, dec, data(500), r.standard_normal((500, 2)))[0] for _ in range(3)])
    assert elbo > -8.5, f"ELBO after training is {elbo:.2f} nats; a working VAE reaches about -7 (4 prototypes with 5% flipped bits)"


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
