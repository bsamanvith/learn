import numpy as np

rng = np.random.default_rng(0)
d = 40
w_true = np.zeros(d); w_true[:5] = [1.5, -1.0, 1.0, 0.8, -0.6]  # only 5 of 40 features matter

def make(n):
    X = rng.standard_normal((n, d))
    p = 1 / (1 + np.exp(-(X @ w_true - 4.0)))      # intercept -4: positives are rare
    return X, (rng.random(n) < p).astype(int)

Xtr, ytr = make(400)                               # small training set: the model will overfit
Xva, yva = make(2000)                              # validation: for the threshold and temperature
Xte, yte = make(20000)                             # test: report numbers here, once
print(f"positive rate: train {ytr.mean():.3f}, test {yte.mean():.3f}")

def fit_logreg(X, y, steps=5000, lr=0.5):          # plain gradient descent on the log-loss
    Xb = np.c_[np.ones(len(X)), X]; w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        w -= lr * Xb.T @ (1 / (1 + np.exp(-Xb @ w)) - y) / len(y)
    return w

w = fit_logreg(Xtr, ytr)
logit = lambda X: np.c_[np.ones(len(X)), X] @ w
s_va, s_te = logit(Xva), logit(Xte)               # scores = logits

def confusion(y, pred):
    tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
    return tp, fp, fn, tn

def report(y, pred, name):
    tp, fp, fn, tn = confusion(y, pred)
    prec = tp / max(tp + fp, 1); rec = tp / (tp + fn)
    print(f"{name:26s} acc {(tp + tn) / len(y):.3f}  precision {prec:.3f}  recall {rec:.3f}  F1 {2 * prec * rec / max(prec + rec, 1e-12):.3f}")

report(yte, np.zeros_like(yte), "always 'negative'")
report(yte, (s_te > 0).astype(int), "threshold p > 0.5")
report(yte, (s_te > np.log(0.1 / 0.9)).astype(int), "threshold p > 0.1")

def auc(y, s):                                     # P(score of a random positive > random negative)
    r = np.argsort(np.argsort(s)) + 1              # ranks 1..n (no ties with continuous scores)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def average_precision(y, s):                       # area under the precision-recall curve
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); prec = tp / np.arange(1, len(y) + 1)
    return np.sum(prec * yy) / yy.sum()

print(f"ROC AUC {auc(yte, s_te):.3f}   average precision {average_precision(yte, s_te):.3f}   (a random scorer gets 0.5 and {yte.mean():.3f})")

def ece(y, p, bins=10):                            # expected calibration error
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum(np.mean(b == k) * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if np.any(b == k))

sig = lambda z: 1 / (1 + np.exp(-z))
nll = lambda y, z: np.mean(np.logaddexp(0, z) - y * z)
b, a = fit_logreg(s_va[:, None], yva, steps=20000, lr=0.1)   # Platt scaling: logistic regression on the score
print(f"Platt scaling on validation: p = sigmoid({a:.2f} * score {b:+.2f})")
print(f"test log-loss {nll(yte, s_te):.3f} -> {nll(yte, a * s_te + b):.3f}")
print(f"test ECE      {ece(yte, sig(s_te)):.3f} -> {ece(yte, sig(a * s_te + b)):.3f}")
print(f"ROC AUC after scaling: {auc(yte, a * s_te + b):.3f} (unchanged: same ranking)")

# positive rate: train 0.083, test 0.085
# always 'negative'          acc 0.915  precision 0.000  recall 0.000  F1 0.000
# threshold p > 0.5          acc 0.885  precision 0.343  recall 0.391  F1 0.366
# threshold p > 0.1          acc 0.821  precision 0.257  recall 0.593  F1 0.359
# ROC AUC 0.817   average precision 0.331   (a random scorer gets 0.5 and 0.085)
# Platt scaling on validation: p = sigmoid(0.28 * score -1.20)
# test log-loss 0.379 -> 0.233
# test ECE      0.082 -> 0.004
# ROC AUC after scaling: 0.817 (unchanged: same ranking)
