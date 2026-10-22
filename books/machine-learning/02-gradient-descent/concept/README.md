# 02 · Gradient descent and the delta rule

> **Open the lab first:** [gradient descent lab](https://bsamanvith.github.io/learn/ml/gradient-descent-lab.html)
> (source: [`docs/ml/gradient-descent-lab.html`](../../../../docs/ml/gradient-descent-lab.html)).
> For a hands-off version, watch the [animations](https://bsamanvith.github.io/learn/ml/watch-it-learn.html). Press **Run**, then play with the learning rate.
> Come back here to understand what you saw.

## The picture to keep in your head

You are standing on a hillside in thick fog. You want to reach the lowest
point in the valley. You cannot see the valley, but you can feel the slope
under your feet. So you:

1. Feel which way the ground tilts.
2. Take a small step in the downhill direction.
3. Repeat until the ground is flat.

That is gradient descent. The hillside is the **loss surface** from lesson
01: the height at each point $(w, b)$ is the loss of that line. "Feeling the
slope" is computing the **gradient**. The step size is the **learning rate**.

## Step 1: the slope in one dimension

Freeze $b$ for a moment and look at the loss as a function of $w$ only. It is
a U-shaped curve.

```
 L(w)
  │ \                     /
  │  \                   /
  │   \       ●  slope here is negative (going down to the right)
  │    \     ╱ ╲
  │     \             /
  │      \_         _/
  │        ‾‾‾‾‾‾‾‾‾      ← minimum: slope is zero
  └──────────────────────────── w
```

> **Math detour: the derivative is a slope.** The derivative $\frac{dL}{dw}$
> answers one question: *if I nudge $w$ up by a tiny amount, how much does
> $L$ change, per unit of nudge?*
> - Negative derivative: increasing $w$ makes the loss go **down**. Move right.
> - Positive derivative: increasing $w$ makes the loss go **up**. Move left.
> - Zero: you are at a flat spot (the bottom, for a bowl).
>
> You can always approximate it numerically:
> $\frac{dL}{dw} \approx \frac{L(w + h) - L(w)}{h}$ for a tiny $h$ like $10^{-5}$.

In every case, **move opposite to the sign of the slope**. And move further
when the slope is steep (you are far from the bottom), less when it is
shallow (you are close). One rule does both:

$$w_{\text{new}} = w - \eta \, \frac{dL}{dw}$$

$\eta$ ("eta") is the **learning rate**: how big a step to take per unit of
slope. The amount you change by is often written with a delta:

$$\Delta w = -\eta\,\frac{dL}{dw}$$

That minus sign is the "descent" in gradient descent.

## Step 2: more than one knob, partial derivatives

The loss depends on both $w$ and $b$. Ask the same question for each knob
separately, holding the other one fixed:

- $\frac{\partial L}{\partial w}$: nudge only $w$, how fast does the loss change?
- $\frac{\partial L}{\partial b}$: nudge only $b$, how fast does the loss change?

The curly $\partial$ just means "the other knobs are frozen". These are
**partial derivatives**. Stack them into a vector and you get the
**gradient**:

$$\nabla L = \begin{bmatrix} \partial L / \partial w \\ \partial L / \partial b \end{bmatrix}$$

Two facts about the gradient make everything work:

1. It points in the direction of **steepest uphill**. So $-\nabla L$ is
   steepest downhill.
2. It is at right angles to the contour lines of the loss map (contour
   lines are paths of equal loss; walking along one doesn't change the loss).

In the lab, the arrow at the current point is the step $-\eta\nabla L$, and
the bottom panel shows the 1D slice through the bowl along $w$, with the
tangent line whose slope is $\partial L/\partial w$.

The update now happens to every knob at once:

$$w \leftarrow w - \eta\,\frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta\,\frac{\partial L}{\partial b}$$

## Step 3: computing the gradient for linear regression

Recall $L = \frac1n\sum_i (w x_i + b - y_i)^2$. Call the residual
$r_i = \hat y_i - y_i = w x_i + b - y_i$.

> **Math detour: the chain rule.** If $L$ depends on $r$ and $r$ depends on
> $w$, then $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial r}\cdot\frac{\partial r}{\partial w}$.
> "How much L moves per unit of r" times "how much r moves per unit of w".
> This one rule, applied over and over, is backpropagation (lesson 11).

For one point: $\frac{\partial (r_i^2)}{\partial r_i} = 2r_i$, and
$\frac{\partial r_i}{\partial w} = x_i$, $\frac{\partial r_i}{\partial b} = 1$. So:

$$\frac{\partial L}{\partial w} = \frac{2}{n}\sum_i r_i\,x_i, \qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum_i r_i$$

## Step 4: the delta rule, and why it makes sense

Plug that into the update and flip the sign of the residual so it reads
"truth minus guess", $\delta_i = y_i - \hat y_i$ (this is the **delta**):

$$\boxed{\;\Delta w = \eta\,\frac{2}{n}\sum_i \delta_i\,x_i, \qquad \Delta b = \eta\,\frac{2}{n}\sum_i \delta_i\;}$$

This is the **delta rule** (also called the Widrow–Hoff or LMS rule, from
1960). Read it in words: *change each weight in proportion to the error,
times the input that weight is attached to.*

It matches common sense:

| Situation | $\delta$ | $x$ | $\Delta w$ | What happens |
|-----------|----------|-----|------------|--------------|
| Guess too low, positive input | + | + | + | Raise $w$ so the guess rises |
| Guess too low, negative input | + | − | − | Lower $w$ (because $w\cdot x$ rises when $w$ falls) |
| Guess too high, positive input | − | + | − | Lower $w$ |
| Guess exactly right | 0 | any | 0 | Leave it alone |
| Input is zero | any | 0 | 0 | This weight didn't cause the error, so don't blame it |

That last row is the deep idea: **each weight is blamed in proportion to how
much it contributed to the output.** Keep it in mind. In lesson 03 you will
see logistic regression produce *exactly the same rule*, and in neural nets
the "delta" of each neuron is passed backwards through the layers. That is
backpropagation in one sentence.

## Step 5: a worked example by hand

Data from lesson 01: $x = (1, 2, 3)$, $y = (2, 4, 5)$. Start at $w = 0$, $b = 0$,
learning rate $\eta = 0.1$.

| | $w$ | $b$ | guesses $\hat y$ | deltas $\delta = y - \hat y$ | $\partial L/\partial w$ | $\partial L/\partial b$ | loss |
|---|---|---|---|---|---|---|---|
| step 0 | 0 | 0 | 0, 0, 0 | 2, 4, 5 | $-\frac23(2+8+15) = -16.67$ | $-\frac23(11) = -7.33$ | 15.00 |
| step 1 | 1.667 | 0.733 | 2.40, 4.07, 5.73 | −0.40, −0.07, −0.73 | | | 0.234 |

One step took the loss from 15 to 0.23. The update was
$\Delta w = -0.1 \times (-16.67) = +1.667$ and $\Delta b = +0.733$. Keep going
and it settles at $w = 1.5$, $b = 0.667$, the exact answer from lesson 01.
Run [`../code/gradient_descent.py`](../code/gradient_descent.py) to watch
every step.

## Step 6: the learning rate, the most important knob you set yourself

In the lab, try the three presets.

- **Too small:** every step is tiny. You get there eventually, but it takes
  thousands of steps.
- **Just right:** fast, smooth convergence.
- **Too big:** you jump over the bottom to the other side of the valley,
  further up than you started. Each jump gets bigger. The loss explodes.
  This is **divergence**.

Why is there a hard limit? Near the bottom, a bowl looks like
$L \approx \frac{c}{2}(w - w^*)^2$, where $c$ is the **curvature** (how sharply the
bowl bends). One step multiplies your distance from the bottom by
$(1 - \eta c)$. If $\eta c > 2$, that factor is bigger than 1 in size, and every
step takes you further away. So:

$$\eta < \frac{2}{c_{\max}}$$

where $c_{\max}$ is the curvature in the steepest direction. The lab shows
this number live.

## Step 7: why feature scaling matters (switch the lab to "raw x")

Switch the lab's data to **raw x (far from 0)**. The inputs are now 2 to 4 instead of −1 to 1.
The bowl turns into a long, narrow valley. The steep direction caps the
learning rate (you must keep $\eta$ small or you diverge across the valley),
but the shallow direction then needs thousands of those small steps. You see
gradient descent zig-zag and crawl.

The fix is to **centre and scale** your inputs (subtract the mean, divide by
the standard deviation) before training. That makes the bowl round, and one
learning rate works well in every direction. This is why every ML pipeline
normalises its inputs, and why neural nets use tricks like batch norm.
(Lesson 06 gives the matrix version: the "curvatures" are the eigenvalues of
the Hessian.)

## Step 8: when does it stop?

Stop when the gradient is close to zero (the ground is flat), when the loss
stops improving, or after a fixed number of steps. For linear regression the
loss is a perfect bowl (**convex**), so there is exactly one bottom and
gradient descent always finds it with a safe learning rate. Neural nets are
not convex: they have many valleys, saddle points and plateaus. Surprisingly,
plain gradient descent (with noise from mini-batches, lesson 08) still works
very well on them.

## Summary

- The gradient is the vector of partial derivatives; it points uphill.
- Gradient descent: $\theta \leftarrow \theta - \eta\nabla L(\theta)$, for every knob $\theta$.
- For squared error that becomes the delta rule: change = rate × error × input.
- The learning rate must be below $2/c_{\max}$; scaling inputs makes the bowl
  round so one rate suits every direction.

Then try the [exercises](../exercises/README.md).
