---
title: "Watch logistic regression run away"
kind: assignment
time: "90 min"
after: "03"
---

# Watch logistic regression run away

Lesson 03 minimised the log-loss and stopped when the gradient was small. Tonight you
find a dataset where that never happens: the loss keeps falling towards zero, the weights
keep growing, and gradient descent never arrives anywhere. Then you discover that it is
quietly doing something useful while it runs away: its direction is turning, very slowly,
towards the **maximum-margin** line, the one a support vector machine would pick.

**Deliverable:** one script, `runaway.py`, and two plots: the data with the decision
line drawn at several times, and $\|w\|$ against the step count on a log axis.

## Before you start

- Lesson 03 (logistic regression and log-loss). Problem 5 of the
  [hard ML volume](../lessons/H2-hard-problems-ml.html) proves what you'll see, if you
  want the theory first.
- Fold the bias into the weights: give every point a third coordinate equal to 1, so
  the model is $p(y = 1 \mid x) = \sigma(w^\top x)$ with $w \in \mathbb R^3$. Use labels
  $y = \pm1$, so the loss is $\frac1n\sum_i\log(1 + e^{-y_iw^\top x_i})$.
- Data: `rng = np.random.default_rng(0)`, then 40 points from $N((2, 2), I)$ labelled
  $+1$ and 40 from $N((-2, -2), I)$ labelled $-1$. Draw the positives first, then the
  negatives, each as `rng.normal(0, 1, (40, 2)) ± 2`.

## Step 1 · Gradient descent on separable data

Write the loss and its gradient, $-\frac1n\sum_iy_ix_i\,\sigma(-y_iw^\top x_i)$. Use
`np.logaddexp(0, -m)` for $\log(1 + e^{-m})$, so huge margins don't overflow. The
gradient is Lipschitz with constant $L = \lambda_{\max}(X^\top X)/(4n)$: compute it and
take $\eta = 1/L$, rounded down. Run 100,000 steps from $w = 0$.

**Predict first:** after 100,000 steps, is the loss zero? Has $w$ stopped changing?

**You should see:** $L = 2.47$, so $\eta = 0.4$. Then:

| step $t$ | 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|
| loss | 5.0e-2 | 8.6e-3 | 1.2e-3 | 1.5e-4 | 1.6e-5 |
| $\lVert w\rVert$ | 1.25 | 2.27 | 3.63 | 5.23 | 6.98 |

The loss falls by about 10 for every factor of 10 in $t$, so it behaves like $1/t$, and
it never reaches 0. $\lVert w\rVert$ grows by an almost constant amount per factor of
10: that is $\lVert w\rVert \propto \log t$, growth that never stops.

## Step 2 · Where is it heading?

Compute the maximum-margin separator: the smallest $\hat w$ with $y_i\hat w^\top x_i \ge 1$
for every $i$ (the margin is then $1/\lVert\hat w\rVert$). A few lines do it: coordinate
ascent on the dual, $\alpha_i \leftarrow \max\big(0,\ \alpha_i + (1 - (K\alpha)_i)/K_{ii}\big)$
with $K = ZZ^\top$, $Z_i = y_ix_i$, and $\hat w = Z^\top\alpha$. Then print the angle between
your $w$ and $\hat w$ at each checkpoint.

**You should see:** margin $\gamma = 1.159$ and $\lVert\hat w\rVert = 0.863$. The angle
shrinks from 14.5° at $t = 10$ to 13.6°, 11.1°, 8.3° and 6.2° at $t = 100{,}000$:
heading for zero, but only like $1/\log t$. In the last decade $\lVert w\rVert$ grew by
1.74, close to $\lVert\hat w\rVert\ln10 = 1.99$, as the theory says it should
($w_t \approx \hat w\log t$).

**Plot 1:** the data with the decision line $w^\top x = 0$ at each checkpoint, and the
maximum-margin line dashed. Watch the line rotate towards it.

## Step 3 · Break the pattern: overlapping data

Draw a second pair of blobs from the same generator, centred at $\pm(0.6, 0.6)$ so the classes overlap, and rerun.

**You should see:** the loss stops at 0.4680 and $\lVert w\rVert$ stops at 1.3545 within a
few hundred steps and never moves again. With overlap, every direction eventually pays for
misclassified points, so a finite minimiser exists. **Plot 2:** $\lVert w\rVert$ against
$t$ on a log axis for both datasets: a rising straight line, and a flat one.

## Step 4 · Fix it with weight decay

Back on the separable data, add $\frac\lambda2\lVert w\rVert^2$ to the loss.

**You should see** after 100,000 steps:

| $\lambda$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
|---|---|---|---|
| $\lVert w\rVert$ | 2.01 | 3.17 | 4.55 |
| angle to max margin | 12.4° | 9.3° | 7.0° |

The weights now converge, and smaller $\lambda$ lands closer to the maximum-margin
direction. As $\lambda \to 0$ the regularised solution's direction tends to $\hat w$:
that limit is the theorem behind Step 2.

## Deliverable

`runaway.py`, printing the tables above, and the two plots. Under Plot 2 write one
sentence: why does an unregularised logistic regression on separable data report
predicted probabilities of 0.99999 and above, and why should you not believe them?

## Stretch

- **Momentum or Adam.** Do they reach the max-margin direction faster? (Adam does not
  converge to the same direction in general. Check it.)
- **Early stopping as regularisation.** Plot the angle against $t$ next to the angle
  against $\lambda$ from Step 4. Which step count matches which $\lambda$?
- **Remove the bias** and redo Step 2. The answer changes. Why?

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/runaway_logistic.py`](../sims/runaway_logistic.py) prints every checkpoint on
this page; run it with `--plot` for both plots. Its output, shortened:

```text
Step 1 · separable blobs: 80 points; max-margin separator has margin γ = 1.159, ‖ŵ‖ = 0.863
         logistic smoothness bound L = λ_max(XᵀX)/(4n) = 2.468, so take η = 1/L ≈ 0.4
         t =      10: loss = 5.00e-02   ‖w‖ =   1.25   angle to max-margin = 14.52°
         t =     100: loss = 8.55e-03   ‖w‖ =   2.27   angle to max-margin = 13.56°
         t =   1,000: loss = 1.22e-03   ‖w‖ =   3.63   angle to max-margin = 11.06°
         t =  10,000: loss = 1.45e-04   ‖w‖ =   5.23   angle to max-margin =  8.34°
         t = 100,000: loss = 1.59e-05   ‖w‖ =   6.98   angle to max-margin =  6.24°
         growth per factor of 10 in t, last decade: 1.74   (‖ŵ‖·ln 10 = 1.99)
Step 2 · overlapping blobs: loss 0.4680 and ‖w‖ = 1.3545 from step 1,000 on
Step 3 · weight decay: λ = 1e-2 → ‖w‖ = 2.01, 12.35°;  1e-3 → 3.17, 9.25°;  1e-4 → 4.55, 6.98°
```

</details>
