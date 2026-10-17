# Probability

The mathematics of uncertainty — from coin flips to martingales.

- **The plan:** [PLAN.md](PLAN.md) — 10 parts, ~55 chapters, with signature visuals and problems.
- **How it's taught:** [principles/](../../principles/)

## Layout

```
text/        chapters: intuition, definitions, worked examples, theory, pitfalls
exercises/   problems per chapter, solutions at the bottom of each file
widgets/     interactive HTML (run `python3 -m http.server` from the repo root)
sims/        Python simulations that check results (`.venv/bin/python subjects/probability/sims/<name>.py`)
assets/      generated figures
```

## Chapters

| # | Chapter | Exercises | Widgets |
|---|---|---|---|
| 0.1 | [Three meanings of probability](text/0.1-meanings.md) | [→](exercises/0.1-meanings.md) | [coin convergence](widgets/coin-convergence.html) |
| 0.2 | [A short history](text/0.2-history.md) | [→](exercises/0.2-history.md) | [problem of points](widgets/problem-of-points.html) |
| 0.3 | [Your intuition is broken](text/0.3-intuition.md) | [→](exercises/0.3-intuition.md) | [predict, then simulate](widgets/predict-then-simulate.html) |
| 1.1 | [Sample spaces and events](text/1.1-sample-spaces.md) | [→](exercises/1.1-sample-spaces.md) | [event explorer](widgets/event-explorer.html) |
| 1.2 | [The axioms](text/1.2-axioms.md) | [→](exercises/1.2-axioms.md) | [area = probability](widgets/venn-area.html) |
| 1.3 | [Counting](text/1.3-counting.md) | [→](exercises/1.3-counting.md) | [birthday problem](widgets/birthday.html) |
| 1.4 | [Story proofs](text/1.4-story-proofs.md) | [→](exercises/1.4-story-proofs.md) | [lattice paths](widgets/lattice-paths.html) |
| 1.5 | [Inclusion–exclusion in action](text/1.5-inclusion-exclusion.md) | [→](exercises/1.5-inclusion-exclusion.md) | [hat check](widgets/hat-check.html) |

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
