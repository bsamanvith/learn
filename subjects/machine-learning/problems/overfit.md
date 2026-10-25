---
title: "Overfit on purpose"
kind: assignment
time: "90 min"
after: "05"
---

# Overfit on purpose

Overfitting usually sneaks up on you. Tonight you do it deliberately and as badly as
possible: twenty noisy points, polynomials up to degree 19, a training error of exactly
zero, and a test error around $10^{14}$. Then you try the two cures from lessons 04 and 05,
a ridge penalty and choosing the model by cross-validation, and you find out which one
actually works here, and why more data beats both.

**Deliverable:** one script, `overfit.py`, and three plots: the fitted curves, training
and test error against degree, and error against $\lambda$.

## Before you start

- Lessons 04 (generalisation) and 05 (regularisation). The leave-one-out shortcut from
  [problem set 2](../lessons/P2-problems-stage-2.html), problem 5, saves you all the refitting.
- Data: `rng = np.random.default_rng(1)`, then `x = np.sort(rng.uniform(0, 1, 20))`,
  `y = np.sin(2*np.pi*x) + rng.normal(0, 0.3, 20)`. For testing, 2000 fresh points the
  same way: `xt = rng.uniform(0, 1, 2000)` and `yt` with fresh noise. The noise has
  variance 0.09, so no model can get test MSE below 0.09.
- Polynomial features: for Step 1 use Chebyshev polynomials,
  `np.polynomial.chebyshev.chebvander(2*x - 1, d)`. They span the same functions as
  $1, x, \dots, x^d$ but keep the least-squares problem solvable at degree 19.

## Step 1 · Climb the degrees

Fit least squares (`np.linalg.lstsq`) for degrees 0, 1, 3, 5, 7, 9, 12, 15 and 19, and
print training MSE, test MSE and the size of the coefficient vector.

**Predict first:** which degree has the lowest test error? What is the training error
at degree 19, with 20 coefficients and 20 points?

**You should see:**

| degree | 0 | 1 | 3 | 5 | 7 | 9 | 15 | 19 |
|---|---|---|---|---|---|---|---|---|
| train MSE | 0.513 | 0.233 | 0.112 | 0.087 | 0.080 | 0.061 | 0.023 | 0.000 |
| test MSE | 0.643 | 0.302 | **0.099** | 0.169 | 0.405 | 9.5 | 4.4e5 | 2.6e14 |
| $\lVert w\rVert$ | 0.2 | 1.0 | 0.9 | 1.0 | 1.8 | 10.5 | 2.6e3 | 6.2e7 |

Training error only goes down. Test error is best at degree 3, within 0.01 of the noise
floor, and then explodes. Degree 19 passes through every point exactly, and between and
beyond them it swings to values in the millions. The coefficients grow with it: large
weights are the signature of overfitting, which is what ridge will attack.

**Plot 1:** the data, the truth $\sin2\pi x$, and the fits of degree 1, 3 and 19 (clip
the axis to $[-2, 2]$). **Plot 2:** train and test MSE against degree, log scale.

## Step 2 · Cure one: keep degree 19, penalise the weights

Use the plain power basis $X = [1, x, \dots, x^{19}]$ and ridge regression,
$w_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$, for $\lambda = 10^{-14}, 10^{-13}, \dots, 1$.
This $X$ is terribly conditioned, so compute everything from one SVD $X = USV^\top$:
$w_\lambda = V\operatorname{diag}\big(\frac{s_j}{s_j^2 + \lambda}\big)U^\top y$, and the
hat matrix's diagonal is $H_{ii} = \sum_jU_{ij}^2\frac{s_j^2}{s_j^2 + \lambda}$. Score each
$\lambda$ by leave-one-out, $\frac1n\sum_i\big(\frac{y_i - \hat y_i}{1 - H_{ii}}\big)^2$, and
by test error. Also print the effective number of parameters, $\operatorname{tr}H$.

**You should see:**

| $\lambda$ | $10^{-14}$ | $10^{-10}$ | $10^{-6}$ | $10^{-2}$ | $1$ |
|---|---|---|---|---|---|
| leave-one-out | 158 | 14.3 | 0.238 | **0.156** | 0.268 |
| test | 33.0 | 0.244 | 0.805 | 0.227 | 0.272 |
| $\operatorname{tr}H$ | 12.1 | 10.1 | 7.3 | 4.2 | 2.4 |

Leave-one-out picks $\lambda = 10^{-2}$, with about 4 effective parameters and test MSE
0.227. That's a cure for the explosion ($10^{14}$ down to 0.23), but it is still more
than twice degree 3's error. **Plot 3:** train, leave-one-out and test error against
$\lambda$ on log–log axes, with the chosen $\lambda$ marked.

## Step 3 · Cure two: let leave-one-out choose the degree

Compute the leave-one-out error of plain least squares for each degree 0 to 15 (the
same formula, with $H = X(X^\top X)^{-1}X^\top$, or `F @ np.linalg.pinv(F)`).

**You should see:** it picks degree 3 (leave-one-out 0.153), whose test MSE is 0.099.
Choosing the right model class beat penalising the wrong one. The penalty treats all 20
coefficients alike, but the truth needs only a few smooth ones, and a penalty on the
coefficients of $x^{19}$ is a strange way to say "be smooth".

## Step 4 · Break the trade-off: more data

Draw 200 new training points and refit degree 19 with **no** penalty.

**You should see:** train 0.071, test 0.097. With ten times more points than
coefficients, the noise averages out and the wild model is fine. Overfitting is a
property of a model *and* a dataset size, not of the model alone.

## Deliverable

`overfit.py` printing the tables above, and the three plots. Add one sentence: given
twenty points, which cure would you reach for first, and what would change your mind?

## Stretch

- **Ridge depends on the basis.** Penalise the Chebyshev coefficients instead of the
  power-basis ones, again choosing $\lambda$ by leave-one-out. Is the result better or
  worse? Ridge is not invariant to a change of basis, because $\lVert w\rVert^2$ isn't.
- **A smoothness penalty.** Replace $\lambda\lVert w\rVert^2$ by $\lambda\int_0^1 f''(x)^2dx$,
  which is a quadratic form $w^\top Pw$ in the coefficients. This is a smoothing spline in
  disguise. How close to degree 3 does it get?
- **Repeat over seeds.** With 50 different datasets, how often does leave-one-out pick
  degree 3? How variable is its choice?

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/overfit.py`](../sims/overfit.py) prints every checkpoint on this page; run it with
`--plot` for the three plots. Its output, shortened:

```text
Step 1 · least squares by degree
         degree  3: train MSE =   0.1116   test MSE =       0.0991   ‖w‖ =        0.9
         degree  9: train MSE =   0.0614   test MSE =       9.5190   ‖w‖ =       10.5
         degree 19: train MSE =   0.0000   test MSE = 258260858738522.9375   ‖w‖ = 62447232.1
Step 2 · ridge on degree 19 (power basis)
         λ =  1e-02: train 0.1022   LOO   0.1563   test   0.2273   effective parameters tr(H) =  4.22
         LOO picks λ = 1e-02: test MSE 0.2273 with 4.2 effective parameters
         (the best λ in hindsight, 1e-03, gets 0.2142)
Step 3 · LOO over degrees 0–15 picks degree 3 (LOO 0.1530): test MSE 0.0991
Step 4 · degree 19, no penalty, 200 points: train 0.0708   test 0.0970
```

</details>
