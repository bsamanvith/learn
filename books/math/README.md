# Math on demand

This book is not read front to back. Each entry is a detour that an ML lesson
sends you to at the moment you need it. The "needed by" column tells you
which lesson triggers it, so you always know *why* you are learning it.

Entries marked **inline** are already explained inside the ML lesson that
needs them. The rest get their own short lesson here as we reach them.

## Calculus

| Topic | Needed by | Status |
|-------|-----------|--------|
| Derivative as slope ("if I nudge x, how much does y move?") | ML 02 | inline |
| Partial derivatives and the gradient vector | ML 02 | inline |
| Chain rule | ML 02 (light), ML 11 backprop (heavy) | inline (light) |
| Derivative of the sigmoid, of log | ML 03 | inline |
| Jacobians and gradients of matrix expressions | ML 11 | planned |
| Second derivative, curvature, Hessian | ML 06 (why scaling matters) | planned |

## Probability

| Topic | Needed by | Status |
|-------|-----------|--------|
| Random variable, expectation, variance | ML 01, ML 08 | inline (light) |
| Bernoulli distribution | ML 03 | inline |
| Likelihood and maximum likelihood estimation | ML 03 | inline |
| Gaussian distribution; why squared error = Gaussian likelihood | ML 01 sidebar, ML 05 | planned |
| Conditional probability and Bayes' rule | ML 09 | planned |
| Categorical distribution, cross-entropy, KL divergence | ML 07, ML 14 | planned |
| Markov chains and expected discounted return | RL R2 | planned |
| Sampling, Monte Carlo estimates | ML 08, RL R4, RL R5 | planned |

## Statistics

| Topic | Needed by | Status |
|-------|-----------|--------|
| Mean, variance, standard deviation | ML 01 | inline |
| Bias-variance trade-off | ML 05 | planned |
| The bootstrap | Trees T2 | planned |
| Entropy and information gain | Trees T1 | planned |

## Linear algebra

| Topic | Needed by | Status |
|-------|-----------|--------|
| Vectors and the dot product | ML 01 | inline |
| Matrices, matrix-vector product, shapes | ML 06, ML 10 | planned |
| Eigenvalues as "how curved is the bowl in each direction" | ML 06 | planned |
| Convolution | ML 13 | planned |
