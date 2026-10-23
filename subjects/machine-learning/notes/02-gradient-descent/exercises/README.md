# 02 · Gradient descent: exercises

Same data as lesson 01: $x = (1, 2, 3)$, $y = (2, 4, 5)$.

## Pen and paper

1. For $L(w) = (w - 3)^2$, write $\frac{dL}{dw}$. Starting at $w = 0$ with
   $\eta = 0.25$, do three steps by hand. What fraction of the remaining
   distance to 3 does each step close?
2. Same function, $\eta = 1.1$. Do three steps. What happens, and why? (Hint:
   the curvature of $(w-3)^2$ is $c = 2$, so the limit is $\eta < 2/c = 1$.)
3. Redo the worked example's step 1 yourself (start at $w = 0, b = 0$,
   $\eta = 0.1$) and confirm $w = 1.667$, $b = 0.733$.
4. In your own words, explain why the delta rule multiplies the error by the
   input $x_i$, and not by something else.
5. At the exact answer $w = 1.5$, $b = 2/3$, compute both partial
   derivatives. What do you get, and what does that mean geometrically?

## Lab

6. In the lab, find (by trying) the largest learning rate that still
   converges on the centred data. Compare it with the "max stable rate" the
   lab shows.
7. Switch to raw x. Set the learning rate just under the max stable rate and
   count roughly how many steps it takes to converge. Then switch back to
   centred data and do the same. Explain the difference using the shape of
   the contours.
8. Click a starting point far from the bottom along the long axis of the raw-x valley,
   and another far along the short axis. Which one converges faster?

## Code

9. In `sims/gradient_descent.py`, add a stopping rule: stop when both
   partial derivatives are smaller than $10^{-6}$ in size. Print how many steps
   it took for $\eta = 0.01, 0.05, 0.1, 0.17$.
10. Replace the full-batch gradient with the gradient from **one random
    point** per step (`i = np.random.randint(3)`). This is stochastic gradient
    descent. Plot the loss. How does the curve look different?

---

## Answers

1. $\frac{dL}{dw} = 2(w - 3)$. Steps: $0 \to 1.5 \to 2.25 \to 2.625$. Each step
   multiplies the distance to 3 by $1 - \eta c = 1 - 0.25 \cdot 2 = 0.5$, so it
   closes half the remaining distance.
2. $0 \to 6.6 \to -1.32 \to 8.18$. The factor is $1 - 1.1\cdot 2 = -1.2$: the sign
   flips (you jump across the bottom) and the size grows by 20% per step,
   so it diverges.
5. Both are 0 (residuals are $\frac16, -\frac13, \frac16$; they sum to 0 and
   $\frac16\cdot1 - \frac13\cdot2 + \frac16\cdot3 = 0$). The surface is flat
   there: it is the bottom of the bowl.
7. With raw x the bowl is a long thin valley (curvatures roughly 20.6 and
   0.07 for the lab's data, a ratio near 300), so the safe rate is set by the steep direction
   while the shallow direction needs hundreds of times more steps. With centred x
   the curvatures are close to each other and one rate suits both.
8. The start along the short (steep) axis, because that direction converges
   fast; the long axis is where gradient descent crawls.
