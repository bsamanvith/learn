# 01 · Linear regression

> **Full version on the site:** [https://bsamanvith.github.io/learn/ml/01-linear-regression.html](https://bsamanvith.github.io/learn/ml/01-linear-regression.html) has the complete derivations, interactive figures and exercises. This file is the short summary.

## The question

You have points. Each point is an input $x$ (say, house size in 100 m²) and an
output $y$ (price in lakhs). You want a rule that, given a new $x$, guesses $y$.

The simplest rule that can possibly work is a straight line:

$$\hat{y} = w\,x + b$$

- $w$ (the **weight**) is the slope: how much $\hat y$ rises per unit of $x$.
- $b$ (the **bias**) is the intercept: what $\hat y$ is when $x = 0$.
- $\hat y$ ("y-hat") is the model's *guess*; $y$ is the true answer.

That is the whole model. It has two knobs, $w$ and $b$. **Learning means
finding the knob settings that make the guesses good.**

```
 y
 │            •    /
 │        •     •/
 │          •  / •      ← the line ŷ = w·x + b
 │      •    /
 │    •    /  •
 │      /  •
 │    /
 └──────────────────── x
```

## What does "good" mean? The loss

For one point, the error is the gap between guess and truth, called the
**residual**: $\hat y_i - y_i$.

We want one number that says how bad the whole line is. Add up the errors,
but square them first:

$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n} \left(\hat y_i - y_i\right)^2 = \frac{1}{n}\sum_{i=1}^{n} \left(w x_i + b - y_i\right)^2$$

This is the **mean squared error (MSE)**. Why square?

1. Positive and negative errors would cancel if we just added them. Squaring
   makes every error count.
2. Big errors get punished much more than small ones (an error of 3 costs 9,
   an error of 1 costs 1).
3. The result is a smooth bowl shape, which makes it easy to find the
   bottom. That is lesson 02.

> **Math detour: mean and variance.** The "mean" of numbers is their average,
> $\bar{x} = \frac{1}{n}\sum x_i$. The variance is the mean of squared
> distances from the mean, $\frac{1}{n}\sum (x_i - \bar x)^2$. Notice MSE has
> the same shape: it is the average squared distance between the guesses and
> the truth. If your model always guessed the mean $\bar y$, its MSE would be
> exactly the variance of $y$. So a useful model must beat "the variance of
> $y$". That is what $R^2$ measures: $R^2 = 1 - \text{MSE}/\text{Var}(y)$.

## The key picture: the loss is a landscape over the knobs

Here is the shift in thinking that the rest of ML depends on.

Each choice of $(w, b)$ gives one line, and one loss number. So you can draw
the loss as a **surface** over a 2D map whose axes are $w$ and $b$. For MSE
that surface is always a smooth bowl. The best line is the lowest point of
the bowl.

```
   data space (x, y)                knob space (w, b)
   one line  ─────────────────────▶  one point on the map
   all lines ─────────────────────▶  the whole map, with height = loss
```

Open the [gradient descent lab](https://bsamanvith.github.io/learn/ml/gradient-descent-lab.html)
now and click around the right-hand map. Every click is a different line on
the left. Watch the line get better as you click lower in the bowl.

## Two ways to find the bottom

1. **Solve it exactly.** For this one model there is a formula (the
   "normal equations"). For one input:

   $$w^* = \frac{\text{Cov}(x, y)}{\text{Var}(x)}, \qquad b^* = \bar y - w^* \bar x$$

   Read it as: the slope is "how much x and y move together" divided by "how
   much x moves on its own".
2. **Walk downhill.** Start anywhere, look at which way is downhill, take a
   step, repeat. This is **gradient descent**, and it is the one that
   matters, because it works for models that have no formula: logistic
   regression, neural networks, and LLMs with billions of knobs.

## More than one input

With several inputs (size, rooms, age) the model becomes a weighted sum:

$$\hat y = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b = \mathbf{w}\cdot\mathbf{x} + b$$

> **Math detour: the dot product.** $\mathbf{w}\cdot\mathbf{x}$ means
> "multiply matching entries and add them up". It is the single most common
> operation in all of deep learning. A neuron is a dot product plus a bias,
> passed through a squashing function (lesson 03).

Nothing else changes. The loss is still MSE, still a bowl, just in more
dimensions than we can draw.

## Sidebar: why squared error, really?

If you assume the true data is "a line plus random Gaussian noise", then the
line that makes the observed data *most probable* is exactly the line with
the smallest squared error. So MSE is not an arbitrary choice; it falls out
of an assumption about the noise. Lesson 03 does the same trick with coin
flips instead of Gaussian noise and gets log-loss. This idea is called
**maximum likelihood**, and it is how almost every loss in ML is chosen.

## Check yourself

- What are the two knobs of a 1-input linear model, and what does each do to
  the line?
- Why can't we just add up the residuals without squaring?
- What is the difference between "data space" and "knob space"?

Then try the [exercises](../exercises/README.md).
