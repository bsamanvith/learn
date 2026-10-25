---
title: "Fit a line three ways, then break it"
kind: assignment
time: "90–120 min"
after: "02"
---

# Fit a line three ways, then break it

You've derived least squares ([lesson 01](../lessons/01-linear-regression.html)) and
gradient descent ([lesson 02](../lessons/02-gradient-descent.html)). Tonight you
**run** them on real medical data, three different ways, and check they agree to six
decimals. Then you break gradient descent twice: once with a step that's too big, and
once with a feature that isn't scaled. You'll feel a condition number of 26,000 as
33,000 steps of waiting, and then watch the same problem converge in **one** step.

**Deliverable:** one script, `fit.py`, and three plots: the data with your fitted
line, the residuals, and gradient descent's loss curves on log axes.

## Before you start

- Lessons 01 and 02. Loss is the mean squared error
  $L(w, b) = \frac{1}{n}\sum_i (w x_i + b - y_i)^2$.
- From the repo root: `.venv/bin/python -c "import sklearn"` should print nothing.
  If it fails, follow `tools/README.md` (scikit-learn is in `tools/requirements.txt`).
- Data: scikit-learn ships the **diabetes** dataset (442 patients) inside the package,
  so nothing is downloaded. Use body-mass index in kg/m² as $x$ and disease progression
  one year later as $y$:

```python
from sklearn.datasets import load_diabetes
d = load_diabetes(scaled=False)
x, y = d.data[:, 2], d.target
```

**You should see:** 442 patients, BMI from 18.0 to 42.2 with mean 26.38, and mean
$y$ of 152.13.

## Step 1 · Normal equations by hand

Build $X$ with a column of BMI and a column of ones. Solve
$X^\top X\,\theta = X^\top y$ with `np.linalg.solve`. Don't use `inv`, and ask
yourself why.

**You should see:** $w = 10.233128$, $b = -117.773367$, MSE $= 3890.46$, $R^2 = 0.3439$.
Each extra unit of BMI goes with about 10 more points of progression. BMI explains
about a third of the variation.

**Plot 1:** the scatter and your line. **Plot 2:** residuals $y - \hat y$ against
$\hat y$. Is there a pattern, or a fan shape? What would each tell you?

## Step 2 · The library

Fit `sklearn.linear_model.LinearRegression` on `x[:, None]`.

**You should see:** the same $w$ and $b$ to six decimals. Assert it in code
(`np.allclose(..., atol=1e-6)`), so the script fails loudly if you ever break Step 1.

## Step 3 · Gradient descent from scratch

Start at $\theta = 0$ and step
$\theta \leftarrow \theta - \eta \cdot \frac{2}{n} X^\top (X\theta - y)$. Before
choosing $\eta$, compute the Hessian $\frac{2}{n} X^\top X$ and its eigenvalues
(`np.linalg.eigvalsh`). Lesson 02 says $\eta$ must stay below $2/\lambda_{\max}$.

Run with $\eta = 0.9 \cdot 2/\lambda_{\max}$ until the loss is within 0.1% of the
Step 1 MSE. Count the steps.

**Predict first:** a few hundred steps? A few thousand?

**You should see:** eigenvalues $0.0544$ and $1432.3$, so the condition number is
$\kappa \approx 26{,}333$ and $\eta_{\max} \approx 0.001396$. It takes **33,495 steps**.
Even then $w = 9.79$ and $b = -105.8$: still visibly off the true values. A flat,
narrow valley lets the loss look nearly done long before the knobs are.

## Step 4 · Break it: over the cliff

Set $\eta = 1.01 \cdot \eta_{\max}$ and run 60 steps.

**Warm up in the lab first:** open the [gradient descent lab](../widgets/gradient-descent-lab.html),
push the learning rate past the edge, and watch the walk zig-zag outwards. Then do it
for real.

**You should see:** the loss rises at **every** step, from 29,074 at the start to about
271,295 after 60 steps. The component along the steep eigen-direction is multiplied by
$|1 - \eta\lambda_{\max}| = 1.02$ each step, so it grows exponentially, slowly at first.

**Plot 3 (start it now):** loss minus the best loss, against step, both axes
logarithmic.

## Step 5 · Fix it: standardise

Replace BMI with $z = (x - \bar x)/s_x$ and repeat Steps 3 and 4. First compute the new
Hessian.

**Predict first:** what happens with $\eta = 0.5$?

**You should see:** the Hessian is exactly $2I$ (both eigenvalues $2.000$, $\kappa = 1$),
so $\eta_{\max} = 1$. Then:

| $\eta$ | 0.1 | 0.5 | 0.9 | 1.0 | 1.05 |
|---|---|---|---|---|---|
| result | 20 steps | **1 step** | 20 steps | bounces forever | blows up |

With $\eta = 0.5$, one step lands exactly on the minimum: on a perfectly round bowl the
gradient points straight at the bottom, and $\eta = 1/\lambda$ is precisely the right
distance (this is Newton's method). $\eta = 0.1$ and $\eta = 0.9$ take the same number
of steps: one approaches from one side, the other overshoots and comes back, both
shrinking the error by $|1 - 2\eta| = 0.8$ each step.

Finally, convert the standardised fit back to BMI units and check that it's the
Step 1 line again. Add the standardised loss curve to Plot 3, next to the raw one.

## Deliverable

`fit.py` that prints every checkpoint above, and three plots:

1. Scatter with the fitted line.
2. Residuals against fitted values.
3. Loss gap on log–log axes: raw BMI crawling for tens of thousands of steps, and
   standardised BMI dropping to the floor in about 20.

If your numbers match the checkpoints, you're done. Add a one-line comment under
Plot 3: in your own words, why scaling made such a difference.

## Stretch

- **Add a second feature** (`d.data[:, 8]`, a blood serum measurement). Now the Hessian
  is $3 \times 3$. Does standardising still give $\kappa = 1$? (It won't. Why not? Look at
  the correlation between the two features.)
- **Momentum.** Add heavy-ball momentum to the raw-BMI run and find a setting that
  beats 33,495 steps by a factor of 100.
- **Mini-batches.** Use 32 random patients per step. The loss curve gets noisy: where
  does it level off, and how does that floor change with $\eta$? This is lesson 08's
  question.

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/fit_three_ways.py`](../sims/fit_three_ways.py) prints every checkpoint on this
page. Run it with `--plot` to draw the three plots. Its output, shortened:

```text
Step 1 · normal equations: w = 10.233128, b = -117.773367
         MSE = 3890.46, RMSE = 62.37, R² = 0.3439
Step 2 · scikit-learn:     w = 10.233128, b = -117.773367
Step 3 · Hessian eigenvalues 0.0544 and 1432.3: condition number κ = 26,333
         largest safe step η_max = 2/λ_max = 0.001396
         η = 0.9·η_max: 33,495 steps to get within 0.1% of the best MSE  (w = 9.792, b = -105.82)
Step 4 · η = 1.01·η_max: loss after 60 steps = 271,295  (it started at 29,074 and climbs every step)
Step 5 · standardised BMI: Hessian eigenvalues 2.000 and 2.000 (κ = 1), η_max = 1.000
         η = 0.1 : converged in 20 steps
         η = 0.5 : converged in 1 step
         η = 1.0 : did not converge in 200 steps
```

Why `solve` and not `inv`: solving is cheaper and more accurate. Forming the inverse
explicitly loses digits when $X^\top X$ is badly conditioned, and here $\kappa$ is
26,000.

</details>
