"""Problem book assignment — fit a line three ways, then break it.

Real data: scikit-learn's diabetes set (442 patients), body-mass index against disease
progression one year later. Loss is the mean squared error L = (1/n) Σ (ŷ - y)², as in
lessons 01 and 02, so the Hessian is (2/n) XᵀX and gradient descent needs η < 2/λ_max.

Run: .venv/bin/python subjects/machine-learning/sims/fit_three_ways.py [--plot]
"""

import sys
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

data = load_diabetes(scaled=False)
x, y = data.data[:, 2], data.target          # BMI in kg/m², progression score
n = len(y)
X = np.c_[x, np.ones(n)]                       # columns: slope, intercept
print(f"Step 0 · data: n = {n}, BMI from {x.min():.1f} to {x.max():.1f} (mean {x.mean():.2f}), y mean {y.mean():.2f}")


# Step 1 · normal equations.
w, b = np.linalg.solve(X.T @ X, X.T @ y)
mse = np.mean((X @ [w, b] - y) ** 2)
print(f"\nStep 1 · normal equations: w = {w:.6f}, b = {b:.6f}")
print(f"         MSE = {mse:.2f}, RMSE = {mse ** 0.5:.2f}, R² = {1 - mse / y.var():.4f}")

# Step 2 · scikit-learn.
lr = LinearRegression().fit(x[:, None], y)
print(f"\nStep 2 · scikit-learn:     w = {lr.coef_[0]:.6f}, b = {lr.intercept_:.6f}")
assert np.allclose([w, b], [lr.coef_[0], lr.intercept_], atol=1e-6)


# Step 3 · gradient descent from scratch, on the raw feature.
def gd(X, y, eta, max_steps, tol=1e-3):
    """Run GD from zero until the loss is within tol·MSE* of the best. Returns (theta, steps, losses)."""
    theta, losses = np.zeros(X.shape[1]), []
    for t in range(1, max_steps + 1):
        theta = theta - eta * (2 / len(y)) * X.T @ (X @ theta - y)
        loss = np.mean((X @ theta - y) ** 2)
        losses.append(loss)
        if not np.isfinite(loss) or loss - mse < tol * mse:
            break
    return theta, t, np.array(losses)


H = (2 / n) * X.T @ X
lam = np.linalg.eigvalsh(H)
eta_max = 2 / lam[-1]
print(f"\nStep 3 · Hessian eigenvalues {lam[0]:.4f} and {lam[-1]:.1f}: condition number κ = {lam[-1] / lam[0]:,.0f}")
print(f"         largest safe step η_max = 2/λ_max = {eta_max:.6f}")
theta, steps, raw_losses = gd(X, y, 0.9 * eta_max, 2_000_000)
print(f"         η = 0.9·η_max: {steps:,} steps to get within 0.1% of the best MSE  (w = {theta[0]:.3f}, b = {theta[1]:.2f})")


# Step 4 · break it: step just past the cliff.
_, _, blown = gd(X, y, 1.01 * eta_max, 60)
print(f"\nStep 4 · η = 1.01·η_max: loss after 60 steps = {blown[-1]:,.0f}  (it started at {np.mean(y ** 2):,.0f} and climbs every step)")


# Step 5 · standardise and run again.
z = (x - x.mean()) / x.std()
Z = np.c_[z, np.ones(n)]
lam_z = np.linalg.eigvalsh((2 / n) * Z.T @ Z)
print(f"\nStep 5 · standardised BMI: Hessian eigenvalues {lam_z[0]:.3f} and {lam_z[1]:.3f} (κ = 1), η_max = {2 / lam_z[1]:.3f}")
for eta in (0.1, 0.5, 0.9, 1.0, 1.05):
    th, t, ls = gd(Z, y, eta, 200)
    end = f"{ls[-1]:,.1f}" if np.isfinite(ls[-1]) else "overflow"
    verdict = f"converged in {t} step{'s' * (t > 1)}" if ls[-1] - mse < 1e-3 * mse else f"did not converge in {t} steps"
    print(f"         η = {eta:<4}: {verdict} (last loss {end})")
wz, bz = np.linalg.solve(Z.T @ Z, Z.T @ y)
print(f"         back to BMI units: w = {wz / x.std():.6f}, b = {bz - wz * x.mean() / x.std():.6f}  (same line as Step 1)")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].scatter(x, y, s=8, alpha=0.6)
    xs = np.linspace(x.min(), x.max(), 2)
    axs[0].plot(xs, w * xs + b, "k")
    axs[0].set(xlabel="BMI (kg/m²)", ylabel="progression", title="Data and the least-squares line")
    axs[1].scatter(X @ [w, b], y - X @ [w, b], s=8, alpha=0.6)
    axs[1].axhline(0, color="k")
    axs[1].set(xlabel="fitted value", ylabel="residual", title="Residuals")
    axs[2].semilogy(raw_losses - mse + 1e-9, label="raw BMI, η = 0.9·η_max")
    _, _, std_losses = gd(Z, y, 0.1, 200)
    axs[2].semilogy(std_losses - mse + 1e-9, label="standardised, η = 0.1")
    axs[2].set(xscale="log", xlabel="step", ylabel="loss − best loss", title="Gradient descent")
    axs[2].legend()
    plt.tight_layout()
    plt.show()
