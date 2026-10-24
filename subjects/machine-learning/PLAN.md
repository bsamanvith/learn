# Machine learning: the roadmap

We learn ML **model-first**. Each lesson starts with a model you can picture
(a line through points, an S-curve splitting two classes) and pulls in math
only at the moment the model needs it. Every "math detour" below is a short,
self-contained lesson in [`math-on-demand.md`](math-on-demand.md) that you read when
you hit it, not before.

## Stage 1: Learning from data (we are here)

| # | Lesson | The one idea | Math detour pulled in |
|---|--------|--------------|-----------------------|
| 01 | [Linear regression](01-linear-regression/concept/README.md) | A model is a function with knobs; "learning" means choosing knobs that make a loss small | Mean, variance, squared error; vectors and the dot product |
| 02 | [Gradient descent](02-gradient-descent/concept/README.md) | Walk downhill on the loss surface, one small step at a time | Derivatives as slope, partial derivatives, the gradient vector, the chain rule |
| 03 | [Logistic regression](03-logistic-regression/concept/README.md) | Squash a line through a sigmoid to get a probability, then score it with log-loss | Logarithms, the Bernoulli distribution, likelihood and maximum likelihood |

Animations and the interactive lab live on the site
([https://bsamanvith.github.io/learn](https://bsamanvith.github.io/learn/), source in [`site/`](../../site) and [`lessons/`](lessons)):
[Watch it learn](https://bsamanvith.github.io/learn/machine-learning/watch-it-learn/) and the
[gradient descent lab](https://bsamanvith.github.io/learn/subjects/machine-learning/widgets/gradient-descent-lab.html).

## Stage 2: Making models honest (next)

| # | Lesson | The one idea | Math detour |
|---|--------|--------------|-------------|
| 04 | Train / validation / test splits | A model that memorises is useless; measure on data it has not seen | Sampling, expectation |
| 05 | Overfitting and regularisation (L2, L1) | Penalise big weights to prefer simpler fits | Norms; the bias-variance decomposition |
| 06 | Multiple features and feature scaling | Why normalising inputs makes gradient descent fast (you will see this in the lab) | Matrices, matrix-vector products, eigenvalues as "curvature" |
| 07 | Softmax regression (many classes) | Logistic regression generalised to K classes | Categorical distribution, cross-entropy, KL divergence |
| 08 | Stochastic and mini-batch gradient descent, momentum, Adam | Use a noisy estimate of the gradient to go faster | Expectation of a random variable, variance of an estimator |
| 09 | Evaluation metrics | Accuracy lies on imbalanced data; precision, recall, ROC, calibration | Conditional probability, Bayes' rule |

## Stage 3: Trees and boosting (a separate branch, not a prerequisite)

Trees are a different family from the "weights + gradient" models above.
They do not use gradient descent to fit the tree itself. They split the data
with greedy if/else questions. You can learn this branch any time after
Stage 2; it is not needed for neural nets, but it is the best tool for
tabular data (spreadsheets, logs, business data) and shows up constantly in
practice.

| # | Lesson | The one idea | Math detour |
|---|--------|--------------|-------------|
| T1 | Decision trees | Ask the yes/no question that best separates the data, then recurse | Entropy, information gain, Gini impurity |
| T2 | Random forests (bagging) | Average many noisy trees to cancel their errors | Variance of an average, the bootstrap |
| T3 | Gradient boosting (XGBoost, LightGBM) | Each new tree fits the *gradient of the loss* left by the previous trees | This is where gradient descent comes back: descent in "function space" |

## Stage 4: Neural networks

| # | Lesson | The one idea | Math detour |
|---|--------|--------------|-------------|
| 10 | The neuron and the multilayer perceptron | Logistic regression is a one-neuron network; stack them | Function composition |
| 11 | Backpropagation | The chain rule, applied backwards, computes every gradient at once | Chain rule on graphs, Jacobians |
| 12 | Training deep nets | Initialisation, activations, normalisation, dropout | Variance propagation |
| 13 | Convolutional networks (CNNs) | Share the same small weights across every position in an image | Convolution, translation invariance |
| 14 | Sequences, attention and transformers | Let each token look at every other token; the core of LLMs | Softmax as soft lookup, dot-product similarity |

## Stage 5: Reinforcement learning (AlphaGo and friends)

| # | Lesson | The one idea | Math detour |
|---|--------|--------------|-------------|
| R1 | Bandits and exploration | Try options, learn which pays best, balance exploring and exploiting | Expected value, confidence bounds |
| R2 | Markov decision processes and the Bellman equation | Value of a state = reward now + discounted value of what comes next | Markov chains, recursion, discounting |
| R3 | Q-learning and deep Q-networks | Learn the value of each action from experience | Fixed points, temporal-difference error |
| R4 | Policy gradients | Directly nudge the probabilities of good actions up | The log-derivative trick, variance reduction |
| R5 | AlphaGo / AlphaZero | Policy net + value net + Monte Carlo tree search, trained by self-play | Tree search, sampling |

## What you would miss before neural nets, CNNs and RL

If you went straight from Stage 1 to Stage 4 or 5, these are the gaps that
would trip you up, in rough order of how badly.

1. **The chain rule, properly.** Backpropagation *is* the chain rule. Lesson
   02 introduces it; you need to be comfortable chaining three or four
   functions and tracking which variable each derivative is "with respect
   to".
2. **Vectors, matrices and shapes.** Neural nets are matrix multiplications.
   You need to read `W @ x + b`, know what shape comes out, and know what a
   gradient with respect to a matrix looks like. (Lesson 06 detour.)
3. **Generalisation.** Train/test splits, overfitting, regularisation
   (Lessons 04 and 05). Without them you cannot tell a good network from one
   that memorised its training set.
4. **Cross-entropy and softmax.** Every classifier network, and every LLM,
   is trained with cross-entropy on a softmax. Lesson 03 gives you the
   two-class version; Lesson 07 generalises it.
5. **Stochastic gradient descent and optimisers.** Nobody trains a network
   with full-batch gradient descent. You need SGD, momentum and Adam (Lesson
   08), plus learning-rate schedules.
6. **Probability you will need for RL.** Expected value, conditional
   probability, Markov chains and sampling. RL is built on "the expected sum
   of future rewards", so expectation must feel natural.
7. **Maximum likelihood as the unifying idea.** Squared error is maximum
   likelihood under Gaussian noise; log-loss is maximum likelihood under a
   Bernoulli. Once you see that, every loss function in deep learning stops
   looking arbitrary. (Lessons 01 and 03 both point at this.)
8. **Convolution (for CNNs only).** A short detour: sliding a small filter
   over an image. Easy once matrices are comfortable.

Things you can safely skip for now: measure theory, proofs of convergence,
kernel methods / SVMs (historically important, rarely needed today), and
heavy linear algebra (SVD can wait until PCA or LLM internals).

## How this subject is laid out

```
lessons.json   the lesson list: title, stage, date added (read by the site)
meta.json      subject title, intro and stage names (read by the site)
lessons/       hand-written HTML lessons: the header and article only;
               the site adds navigation, contents and prev/next
pages/         hand-written HTML pages, like "Watch it learn"
widgets/       standalone interactive pages, like the gradient descent lab
assets/        learn.js (the canvas scene engine) and ml.css (lesson components)
sims/          from-scratch NumPy implementations
notes/         the original markdown concept notes and exercise sheets
```
