import numpy as np

rng = np.random.default_rng(0)

# 1. Two facts the lesson derives, checked numerically.
# (a) KL(N(mu, s^2) || N(0, 1)) = (mu^2 + s^2 - 1 - log s^2) / 2, against a Monte Carlo average of log q - log p
mu, s = 1.5, 0.5
z = mu + s * rng.standard_normal(1_000_000)
mc = np.mean(-0.5 * ((z - mu) / s) ** 2 - np.log(s) + 0.5 * z ** 2)
print(f"KL closed form {(mu**2 + s**2 - 1 - np.log(s**2)) / 2:.4f}   Monte Carlo {mc:.4f}")

# (b) Two unbiased estimators of d/dmu E_{z ~ N(mu, 1)}[z^2] (true value 2 mu): the score-function
# estimator f(z) (z - mu), and the reparameterised one f'(mu + eps) = 2 (mu + eps)
eps = rng.standard_normal(100_000); z = mu + eps
score, reparam = z ** 2 * (z - mu), 2 * z
print(f"gradient estimates (true {2 * mu}): score function {score.mean():.3f} (std {score.std():.2f}),"
      f" reparameterised {reparam.mean():.3f} (std {reparam.std():.2f})")

# 2. Data: 16x16 images of a soft bar with a random angle (0 to 180 degrees) and length (3 to 7 pixels).
# Each pixel is black or white, on with probability given by the bar's ink. Two numbers determine
# the bar, so a 2-dimensional latent space is enough in principle.
yy, xx = np.mgrid[0:16, 0:16] - 7.5
def bars(n):
    th, ln = rng.uniform(0, np.pi, n), rng.uniform(3, 7, n)
    u = np.cos(th)[:, None, None] * xx + np.sin(th)[:, None, None] * yy        # along the bar
    v = -np.sin(th)[:, None, None] * xx + np.cos(th)[:, None, None] * yy       # across it
    ink = 0.02 + 0.96 * np.exp(-v ** 2 / 0.8) / (1 + np.exp(6 * (np.abs(u) - ln[:, None, None])))
    return (rng.random((n, 256)) < ink.reshape(n, 256)).astype(float), th, ln   # binary pixels

# 3. The VAE: encoder 256 -> 128 -> (mu, log var) in 2-D; decoder 2 -> 128 -> 256 Bernoulli means.
def init(a, b): return [rng.standard_normal((a, b)) * np.sqrt(2 / a), np.zeros(b)]
P = {}
P["W1"], P["b1"] = init(256, 128); P["Wm"], P["bm"] = init(128, 2); P["Wv"], P["bv"] = init(128, 2)
P["W2"], P["b2"] = init(2, 128); P["W3"], P["b3"] = init(128, 256)

def encode(X):
    H = np.maximum(X @ P["W1"] + P["b1"], 0)
    return H, H @ P["Wm"] + P["bm"], H @ P["Wv"] + P["bv"]

def decode(Z):
    G = np.maximum(Z @ P["W2"] + P["b2"], 0)
    return G, G @ P["W3"] + P["b3"]                                  # logits of the pixel means

def log_bernoulli(X, L):
    """log p(x | z) summed over pixels, for Bernoulli means sigmoid(L), written stably."""
    return np.sum(X * L - np.logaddexp(0, L), 1)

def elbo_and_grads(X):
    """Average ELBO over the batch, with one sample of z per example, and its gradients."""
    N = len(X)
    H, m, lv = encode(X)
    e = rng.standard_normal(m.shape); sd = np.exp(lv / 2); Z = m + sd * e     # reparameterisation
    G, L = decode(Z)
    rec = log_bernoulli(X, L); kl = 0.5 * np.sum(m ** 2 + sd ** 2 - 1 - lv, 1)
    # backward pass of -ELBO / N
    dL = (1 / (1 + np.exp(-L)) - X) / N
    g = {"W3": G.T @ dL, "b3": dL.sum(0)}
    dG = (dL @ P["W3"].T) * (G > 0)
    g["W2"], g["b2"] = Z.T @ dG, dG.sum(0)
    dZ = dG @ P["W2"].T
    dm = dZ + m / N                                                    # through z = m + sd e, and the KL
    dlv = dZ * e * sd / 2 + 0.5 * (sd ** 2 - 1) / N
    g["Wm"], g["bm"], g["Wv"], g["bv"] = H.T @ dm, dm.sum(0), H.T @ dlv, dlv.sum(0)
    dH = (dm @ P["Wm"].T + dlv @ P["Wv"].T) * (H > 0)
    g["W1"], g["b1"] = X.T @ dH, dH.sum(0)
    return np.mean(rec - kl), np.mean(rec), np.mean(kl), g

# gradient check: fix the noise by reseeding before each evaluation
X, _, _ = bars(8); worst = 0
state = rng.bit_generator.state
_, _, _, g = elbo_and_grads(X)
for k in P:
    for i in [tuple(rng.integers(0, s) for s in P[k].shape) for _ in range(3)]:
        old = P[k][i]
        P[k][i] = old + 1e-6; rng.bit_generator.state = state; up = elbo_and_grads(X)[0]
        P[k][i] = old - 1e-6; rng.bit_generator.state = state; dn = elbo_and_grads(X)[0]
        P[k][i] = old
        worst = max(worst, abs(-(up - dn) / 2e-6 - g[k][i]))
print(f"gradient check: max |backprop - finite difference| {worst:.1e}")

# train with Adam on fresh data each step
M = {k: np.zeros_like(v) for k, v in P.items()}; V = {k: np.zeros_like(v) for k, v in P.items()}
for t in range(1, 8001):
    X, _, _ = bars(128)
    elbo, rec, kl, g = elbo_and_grads(X)
    for k in P:
        M[k] = 0.9 * M[k] + 0.1 * g[k]; V[k] = 0.999 * V[k] + 0.001 * g[k] ** 2
        P[k] -= 1e-3 * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
    if t in (1, 500, 2000, 8000):
        print(f"step {t:4d}: ELBO {elbo:8.2f}  = reconstruction {rec:8.2f} - KL {kl:5.2f}")

# 4. How tight is the bound? Importance-weighted estimates of log p(x) with K samples from q(z|x):
# log (1/K) sum_k p(x, z_k) / q(z_k | x) is a lower bound for every K and tends to log p(x) as K grows.
Xte, th_te, ln_te = bars(500)
_, m, lv = encode(Xte); sd = np.exp(lv / 2)
for K in [1, 10, 100, 1000]:
    lw = []
    for _ in range(K):
        e = rng.standard_normal(m.shape); Z = m + sd * e
        log_q = np.sum(-0.5 * e ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi), 1)
        log_prior = np.sum(-0.5 * Z ** 2 - 0.5 * np.log(2 * np.pi), 1)
        lw.append(log_bernoulli(Xte, decode(Z)[1]) + log_prior - log_q)
    lw = np.array(lw)
    est = lw.max(0) + np.log(np.mean(np.exp(lw - lw.max(0)), 0))
    print(f"K = {K:4d}: estimate of log p(x) averaged over the test set {est.mean():8.2f}")
# 5. What did the two latent dimensions learn?
r, phi = np.linalg.norm(m, axis=1), np.arctan2(m[:, 1], m[:, 0])
print(f"distance of mu from 0 against bar length: correlation {np.corrcoef(r, ln_te)[0, 1]:.3f}")
c = np.abs(np.mean(np.exp(1j * (phi - 2 * th_te))))                  # 1 if phi = 2 theta + constant
c2 = np.abs(np.mean(np.exp(1j * (phi + 2 * th_te))))
print(f"direction of mu against bar angle: |mean exp(i(phi -+ 2 theta))| = {max(c, c2):.3f}  (1 = phi is 2 theta, up to rotation and reflection)")

# KL closed form 1.4431   Monte Carlo 1.4434
# gradient estimates (true 3.0): score function 3.004 (std 7.25), reparameterised 3.000 (std 2.00)
# gradient check: max |backprop - finite difference| 2.3e-08
# step    1: ELBO  -222.80  = reconstruction  -222.61 - KL  0.19
# step  500: ELBO   -50.16  = reconstruction   -46.34 - KL  3.82
# step 2000: ELBO   -45.17  = reconstruction   -41.87 - KL  3.31
# step 8000: ELBO   -43.62  = reconstruction   -39.99 - KL  3.63
# K =    1: estimate of log p(x) averaged over the test set   -43.78
# K =   10: estimate of log p(x) averaged over the test set   -43.60
# K =  100: estimate of log p(x) averaged over the test set   -43.57
# K = 1000: estimate of log p(x) averaged over the test set   -43.57
# distance of mu from 0 against bar length: correlation 0.903
# direction of mu against bar angle: |mean exp(i(phi -+ 2 theta))| = 0.987  (1 = phi is 2 theta, up to rotation and reflection)
