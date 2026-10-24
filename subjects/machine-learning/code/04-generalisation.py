import numpy as np

rng = np.random.default_rng(0)
n, sigma = 15, 0.3
x = (np.arange(n) + 0.5) / n
y = np.sin(2 * np.pi * x) + sigma * rng.standard_normal(n)

def design(x, d):                      # columns 1, t, t^2, ..., t^d with t in [-1, 1]
    return np.vander(2 * x - 1, d + 1, increasing=True)

def fit(x, y, d):
    return np.linalg.lstsq(design(x, d), y, rcond=None)[0]

def kfold_cv(x, y, d, k=5):
    idx = rng.permutation(len(x))
    folds = np.array_split(idx, k)
    errs = []
    for f in folds:
        train = np.setdiff1d(idx, f)
        w = fit(x[train], y[train], d)
        errs.append(np.mean((design(x[f], d) @ w - y[f]) ** 2))
    return np.mean(errs)

def loocv(x, y, d):                    # the closed form for linear smoothers
    X = design(x, d)
    H = X @ np.linalg.pinv(X)
    r = y - H @ y
    return np.mean((r / (1 - np.diag(H))) ** 2)

for d in range(0, 10):
    w = fit(x, y, d)
    train = np.mean((design(x, d) @ w - y) ** 2)
    print(f"degree {d}: train {train:.3f}   5-fold CV {kfold_cv(x, y, d):.3f}   LOOCV {loocv(x, y, d):.3f}")

# degree 0: train 0.776   5-fold CV 0.856   LOOCV 0.891
# degree 1: train 0.287   5-fold CV 0.341   LOOCV 0.396
# degree 2: train 0.283   5-fold CV 0.503   LOOCV 0.540
# degree 3: train 0.054   5-fold CV 0.125   LOOCV 0.102   <- LOOCV's choice
# degree 4: train 0.052   5-fold CV 0.165   LOOCV 0.173
# degree 5: train 0.043   5-fold CV 0.114   LOOCV 0.153   <- 5-fold's choice
# degree 6: train 0.040   5-fold CV 0.281   LOOCV 0.148
# degree 7: train 0.039   5-fold CV 3.501   LOOCV 1.470
# degree 8: train 0.038   5-fold CV 20.681  LOOCV 20.932
# degree 9: train 0.011   5-fold CV 27.911  LOOCV 34.547
