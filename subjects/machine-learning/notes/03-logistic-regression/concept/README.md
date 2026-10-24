# 03 · Logistic regression and log-loss

> **Full version on the site:** [https://bsamanvith.github.io/learn/machine-learning/03-logistic-regression/](https://bsamanvith.github.io/learn/machine-learning/03-logistic-regression/) has the complete derivations, interactive figures and exercises. This file is the short summary.

> **Name check.** This is the "logarithmic" model you may have heard of. Its
> real name is **logistic** regression, after the logistic (sigmoid) curve.
> Logarithms show up twice inside it, in the *log-odds* and in the *log-loss*,
> which is why the two names get mixed up.

## The question

Now the output is a yes/no. Did the student pass ($y = 1$) or fail ($y = 0$),
given hours studied $x$? We want the model to output a **probability** of
passing, a number between 0 and 1.

A straight line $wx + b$ can't do that: it runs off to $+\infty$ and $-\infty$.
So we keep the line and **squash** it.

## The sigmoid squash

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
 σ(z)
 1 ┤                    ___________
   │                 _-‾
 .5┤ - - - - - - - -●             ← σ(0) = 0.5
   │             _-‾
 0 ┤___________-‾
   └────────────┼──────────────── z
                0
```

- Very negative $z$ gives almost 0. Very positive $z$ gives almost 1.
- $z = 0$ gives exactly 0.5.

The model is:

$$p = \sigma(wx + b) = P(y = 1 \mid x)$$

The line $z = wx + b$ is still there underneath. The **decision boundary**
(where $p = 0.5$) is where the line crosses zero: $x = -b/w$. The weight $w$
controls how sharp the S is; the bias $b$ slides it left and right.

## Where the first logarithm hides: log-odds

Solve $p = \sigma(z)$ for $z$ and you get:

$$z = wx + b = \log\frac{p}{1 - p}$$

$\frac{p}{1-p}$ is the **odds** ($p = 0.8$ means odds of 4 to 1). So
logistic regression says: *the log of the odds is a straight line in x.*
Each extra unit of $x$ multiplies the odds by $e^{w}$.

> **Math detour: logarithms in two lines.** $\log a$ answers "$e$ to what
> power gives $a$?". The two facts ML uses constantly:
> 1. $\log(ab) = \log a + \log b$. Products become sums.
> 2. $\log$ of a number near 0 is a large negative number
>    ($\log 0.01 \approx -4.6$), and $\log 1 = 0$.

## Scoring a probability: the second logarithm, log-loss

How do we measure how good the predicted probabilities are? Ask: *how
probable does the model say the data we actually saw is?*

> **Math detour: Bernoulli and likelihood.** A yes/no outcome with
> probability $p$ of "yes" is a **Bernoulli** variable. The probability the
> model assigns to the outcome that actually happened is
> $p$ if $y = 1$ and $1 - p$ if $y = 0$; written in one line,
> $p^{y}(1-p)^{1-y}$. For independent points, the probability of the whole
> dataset is the product of these. That product is the **likelihood**. A good
> model makes the likelihood large.

Products of many small numbers are awkward, so take the log (fact 1: the
product becomes a sum), average, and put a minus sign in front so that
smaller is better:

$$L(w, b) = -\frac1n \sum_{i=1}^n \Big[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \,\Big]$$

This is **log-loss**, also called **binary cross-entropy**. For each point
only one of the two terms is active. Here is what one point costs when the
truth is $y = 1$:

| Model says $p$ | Cost $-\log p$ | Verdict |
|----------------|----------------|---------|
| 0.99 | 0.01 | confident and right: almost free |
| 0.9 | 0.11 | |
| 0.5 | 0.69 | a shrug |
| 0.1 | 2.30 | |
| 0.01 | 4.61 | confident and wrong: very expensive |

Log-loss punishes *confident mistakes* brutally. That is what you want from
a model that reports probabilities.

### Why not just use squared error?

Squared error on top of a sigmoid makes a loss surface with flat plateaus:
when the model is confidently wrong, the sigmoid is flat, so the gradient is
tiny and learning stalls exactly when it should be fastest. Log-loss's
logarithm cancels the sigmoid's flatness. It also makes the surface a
convex bowl again, so gradient descent has one bottom to find.

## The gradient: the delta rule comes back

We need $\frac{\partial L}{\partial w}$. Use the chain rule through
$z = wx + b$ and $p = \sigma(z)$.

1. The sigmoid has a neat derivative: $\frac{d\sigma}{dz} = \sigma(z)\big(1 - \sigma(z)\big) = p(1-p)$.
2. For one point, $\frac{\partial L_i}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{p - y}{p(1-p)}$.
3. Multiply: $\frac{\partial L_i}{\partial z} = \frac{p - y}{p(1-p)}\cdot p(1-p) = p - y$.

Everything messy cancels. Then $\frac{\partial z}{\partial w} = x$ and
$\frac{\partial z}{\partial b} = 1$, so:

$$\frac{\partial L}{\partial w} = \frac1n\sum_i (p_i - y_i)\,x_i, \qquad \frac{\partial L}{\partial b} = \frac1n\sum_i (p_i - y_i)$$

Compare with linear regression in lesson 02: it is **the same delta rule**,
"error times input", with the guess now being a probability. (The factor of
2 disappears only because of how each loss is defined.) This is not a
coincidence: whenever the loss is the matching maximum-likelihood loss for
the output (squared error for numbers, log-loss for yes/no, cross-entropy
for many classes), the gradient at the output is simply
**prediction minus truth**. Neural networks rely on this every time they
train.

## No formula this time

Unlike linear regression, there is no closed-form answer for $w$ and $b$.
Setting the gradient to zero gives equations nobody can solve by hand. So
gradient descent is not one option among two; it is *the* way to fit this
model. That is the step from "statistics" to "machine learning".

## Where this leads

- **Many classes:** replace the sigmoid with a softmax and log-loss with
  cross-entropy (lesson 07). That is exactly the last layer of an LLM,
  choosing the next token out of ~100,000 options.
- **A neuron:** $\sigma(\mathbf{w}\cdot\mathbf{x} + b)$ is a single artificial neuron.
  Stack them in layers and you have a neural network (lesson 10).

## Check yourself

- What does $\sigma(0)$ equal, and what does it mean for the decision boundary?
- If $w = 0.7$, by how much does one extra hour of study multiply the odds of passing?
- Why does log-loss cost so much more for $p = 0.01$ than for $p = 0.1$, when the truth is 1?
- What is $\frac{\partial L_i}{\partial z}$ for one point, in four characters?

Then try the [exercises](../exercises/README.md).
