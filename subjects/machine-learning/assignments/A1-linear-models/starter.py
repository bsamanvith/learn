"""Assignment A1: linear models from scratch.

Fill in every function marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same functions

The checker tests each function on its own and tells you what is wrong, so you can work
through them in any order. The lesson each one builds on is noted in its docstring.
Conventions: X is an (n, d) array of features, y an (n,) array of targets, and weights
come back as a (d,) array plus a separate scalar intercept b.
"""
import numpy as np


# ---------------------------------------------------------------- Part 1: least squares (lesson 01)

def fit_least_squares(X, y):
    """Return (w, b) minimising sum_i (y_i - w.x_i - b)^2.

    Don't invert X^T X: centre the data so the intercept drops out, then solve with
    np.linalg.lstsq (which is stable even when X^T X is badly conditioned)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: gradient descent (lesson 02)

def gd_least_squares(X, y, lr, steps):
    """Gradient descent on L(w, b) = (1/2n) sum_i (y_i - w.x_i - b)^2, starting from zero.

    Return (w, b, losses) where losses[t] is L before step t (length `steps`)."""
    raise NotImplementedError  # TODO


def max_stable_lr(X):
    """The largest learning rate for which gd_least_squares converges, i.e. 2 / lambda_max of
    the Hessian of L with respect to (w, b). Hint: append a column of ones to X."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 3: logistic regression (lesson 03)

def sigmoid(z):
    """1 / (1 + exp(-z)), computed without overflow warnings for large |z|."""
    raise NotImplementedError  # TODO


def fit_logistic(X, y, lam=0.0, lr=0.5, steps=3000):
    """Gradient descent on the mean log-loss plus (lam/2)||w||^2 (the intercept is not
    penalised), starting from zero. y contains 0s and 1s. Return (w, b)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 4: regularisation (lesson 05)

def fit_ridge(X, y, lam):
    """Return (w, b) minimising sum_i (y_i - w.x_i - b)^2 + lam ||w||^2, intercept unpenalised."""
    raise NotImplementedError  # TODO


def ridge_df(X, lam):
    """Effective degrees of freedom of ridge on the centred X: sum_j d_j^2 / (d_j^2 + lam)."""
    raise NotImplementedError  # TODO


def soft_threshold(z, t):
    """sign(z) * max(|z| - t, 0), elementwise."""
    raise NotImplementedError  # TODO


def fit_lasso(X, y, lam, iters=500):
    """Coordinate descent for (1/2n) sum_i (y_i - w.x_i - b)^2 + lam ||w||_1, intercept
    unpenalised (centre X and y first). Return (w, b)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 5: model selection (lesson 04)

def kfold_indices(n, k, seed=0):
    """Shuffle range(n) with np.random.default_rng(seed) and split it into k folds whose
    sizes differ by at most one. Return a list of k integer arrays."""
    raise NotImplementedError  # TODO


def cv_mse(X, y, lam, k=5, seed=0):
    """Mean squared error of fit_ridge(., ., lam) estimated by k-fold cross-validation:
    the average over all n points of the squared error made by the model that did not
    see that point."""
    raise NotImplementedError  # TODO


def choose_lambda(X, y, lams, k=5, seed=0):
    """Return the value in `lams` with the smallest cv_mse."""
    raise NotImplementedError  # TODO
