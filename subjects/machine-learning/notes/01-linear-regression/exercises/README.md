# 01 · Linear regression: exercises

Use this tiny dataset for everything below (small enough to do by hand):

| $x$ | 1 | 2 | 3 |
|-----|---|---|---|
| $y$ | 2 | 4 | 5 |

## Pen and paper

1. Sketch the three points. Guess a line by eye and write down your $w$ and $b$.
2. Compute the MSE of the line $\hat y = 1\cdot x + 1$.
3. Compute the MSE of your own guessed line. Is it better or worse than (2)?
4. Use the exact formulas $w^* = \text{Cov}(x,y)/\text{Var}(x)$ and
   $b^* = \bar y - w^*\bar x$ (with the $\frac1n$ convention) to find the best line.
5. Compute the MSE of the best line, then $R^2 = 1 - \text{MSE}/\text{Var}(y)$.
6. Explain in one sentence why no other line can have a lower MSE than the
   one in (4), using the "bowl" picture.

## Code

7. In plain Python (no NumPy), write `mse(w, b, xs, ys)` and check it against
   your answers to (2) and (5).
8. Loop over a grid of $w \in [0, 3]$ and $b \in [-1, 3]$ in steps of 0.1,
   compute the MSE at each point, and print the $(w, b)$ with the lowest
   value. How close does the grid get to the exact answer?
9. Why does (8) stop being practical when the model has 1,000 knobs instead
   of 2? (Count how many grid points you would need.)

---

## Answers

2. Predictions are 2, 3, 4, so residuals are 0, −1, −1. MSE = (0 + 1 + 1)/3 = **2/3 ≈ 0.667**.
4. $\bar x = 2$, $\bar y = 11/3$. $\text{Cov}(x,y) = \frac13[(-1)(-\frac53) + 0 + (1)(\frac43)] = 1$.
   $\text{Var}(x) = \frac13(1 + 0 + 1) = \frac23$. So $w^* = 1.5$ and
   $b^* = \frac{11}{3} - 1.5\cdot 2 = \frac23 \approx 0.667$.
5. Residuals are $\frac16, -\frac13, \frac16$, so MSE = $\frac13\cdot\frac{6}{36} = \frac1{18} \approx 0.056$.
   $\text{Var}(y) = \frac{14}{9} \approx 1.556$, so $R^2 = 1 - \frac{1}{28} \approx 0.964$.
6. MSE as a function of $(w, b)$ is a bowl with a single lowest point; the
   formula finds the point where the bowl is flat in every direction, which
   for a bowl can only be the bottom.
9. At 10 values per knob, 1,000 knobs need $10^{1000}$ grid points. Gradient
   descent needs one gradient per step, no matter how many knobs there are.
