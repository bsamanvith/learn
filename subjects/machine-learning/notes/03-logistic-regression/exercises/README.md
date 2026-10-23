# 03 · Logistic regression: exercises

Data: hours studied and whether the student passed.

| hours $x$ | 0 | 1 | 2 | 3 |
|-----------|---|---|---|---|
| passed $y$ | 0 | 0 | 1 | 1 |

## Pen and paper

1. Compute $\sigma(-2), \sigma(0), \sigma(2)$ (use $e^2 \approx 7.39$).
2. For $w = 2, b = -3$, compute $p$ for each of the four students. Where is the
   decision boundary?
3. Compute the log-loss of that model on the four points.
4. Compute $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ at $w = 2, b = -3$ using the
   "prediction minus truth" formula. Which way should each knob move?
5. Show that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. (Hint: write
   $\sigma = (1 + e^{-z})^{-1}$ and use the chain rule.)
6. This dataset is perfectly separable (every fail is below every pass).
   What happens to $w$ if you run gradient descent forever? Why? (This is why
   lesson 05's regularisation matters.)

## Code

7. Write `sigmoid`, `log_loss(w, b)` and `gradient(w, b)` in NumPy for the
   data above, reusing the structure of `sims/gradient_descent.py`.
   Check your gradient against a numerical one.
8. Train with $\eta = 0.5$ for 1,000 steps and print $w$, $b$ and the loss every
   100 steps. Confirm what you predicted in (6).
9. Try replacing log-loss with squared error $(p - y)^2$ and start at
   $w = -5, b = 5$ (confidently wrong). Compare how fast each loss starts to
   improve.

---

## Answers

1. $\sigma(-2) \approx 0.119$, $\sigma(0) = 0.5$, $\sigma(2) \approx 0.881$.
2. $z = -3, -1, 1, 3$, so $p \approx 0.047, 0.269, 0.731, 0.953$. The boundary is at $x = 1.5$.
3. Costs: $-\log(0.953) \approx 0.049$, $-\log(0.731) \approx 0.313$,
   $-\log(1 - 0.269) \approx 0.313$, $-\log(1 - 0.047) \approx 0.049$. Mean $\approx 0.181$.
4. $p - y \approx 0.047, 0.269, -0.269, -0.047$.
   $\frac{\partial L}{\partial b} = \frac14(0) = 0$ and
   $\frac{\partial L}{\partial w} = \frac14(0 + 0.269 - 0.538 - 0.142) \approx -0.103$.
   So $w$ should grow (sharper S) and $b$ is already balanced for this $w$.
6. $w$ keeps growing without limit. Making the S steeper always pushes every
   $p$ closer to the correct 0 or 1, so the loss keeps shrinking toward 0 but
   never reaches it. The weights blow up; regularisation stops this.
