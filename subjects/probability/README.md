# Probability

The mathematics of uncertainty — from coin flips to martingales.

- **The plan:** [PLAN.md](PLAN.md) — 10 parts, ~55 chapters, with signature visuals and problems.
- **How it's taught:** [principles/](../../principles/)

## Layout

```
concept/     intuition-first chapters with visuals
text/        rigorous textbook chapters
exercises/   problems + full solutions
assets/      figures and images (generated SVGs live here)
sims/        Python simulations that check results  (added with chapters)
widgets/     interactive HTML widgets for this subject (added with chapters)
```

## Colour vocabulary

Event A = blue · Event B = orange · A∩B = purple · complement / guides = grey.
Used in every figure and widget (see `tools/figures/style.py`, `tools/widgets/_template.html`).

## Notation

| Symbol | Meaning |
|---|---|
| $\Omega$ | Sample space |
| $A, B$ | Events (subsets of $\Omega$) |
| $A^c$ | Complement of $A$ |
| $P(A)$ | Probability of $A$ |
| $P(A \mid B)$ | Probability of $A$ given $B$ |
| $X, Y$ | Random variables |
| $X \sim \text{Bin}(n,p)$ | $X$ has the Binomial distribution |
| $\mathbb{E}[X]$, $\operatorname{Var}(X)$ | Expectation, variance |
| $\mathbb{1}_A$ | Indicator of $A$ |
