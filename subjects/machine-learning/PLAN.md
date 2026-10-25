# Machine learning: the curriculum

We learn ML **model-first**. Each lesson starts with a model you can picture and pulls in the
math only when the model needs it. The math shows up two ways:

- as a **detour box** inside the lesson, for a short idea (the chain rule, a Gaussian, the KL
  divergence), and
- as a lesson in the **math toolkit** (M1–M6), for anything a detour box can't hold. Lessons
  link to it when they need it.

Every lesson has the same anatomy: the idea → the model → the math in full (derivations, not
just results) → a figure → code from scratch → exercises with answers → further reading.

The site reads `lessons.json` and `meta.json`. This file is the plan. Lessons marked ✅ are
written; the rest are planned, in this order.

## Stage 1 · Learning from data

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 01 ✅ | Linear regression | Learning means choosing parameters that make a loss small. Geometry, calculus and probability give the same answer | Mean, variance, dot product, normal equations, projection, Gaussian MLE |
| 02 ✅ | Gradient descent and the delta rule | Walk downhill. The learning rate must stay below 2/λ_max | Gradient, chain rule, Hessian eigenvalues, conditioning |
| 03 ✅ | Logistic regression and log-loss | Squash a line into a probability, then score it with maximum likelihood | Sigmoid, log-odds, Bernoulli likelihood, convexity |

## Stage 2 · Making models honest

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 04 ✅ | Generalisation | Training error lies. Test error = bias² + variance + noise | Expectation, the bias–variance decomposition, cross-validation |
| 05 ✅ | Regularisation | Penalising big weights is a prior. Ridge shrinks, lasso selects | Norms, SVD view of ridge, MAP estimation, Laplace prior |
| 06 ✅ | Softmax regression and cross-entropy | K classes: softmax + cross-entropy, and its gradient is still "prediction − target" | Categorical distribution, entropy, KL divergence, log-sum-exp |
| 07 ✅ | Stochastic optimisation | A noisy gradient is cheaper and often better: SGD, momentum, Adam, schedules | Unbiased estimators, variance of a mean, exponential moving averages |
| 08 ✅ | Evaluating classifiers | Accuracy lies on imbalanced data: precision, recall, ROC, AUC, calibration | Conditional probability, Bayes' rule, base rates |

## Stage 3 · Trees and boosting (a side branch: take it any time after stage 2)

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 09 ✅ | Decision trees | Greedily ask the question that most reduces impurity | Entropy, information gain, Gini impurity |
| 10 ✅ | Bagging and random forests | Averaging many decorrelated trees cancels their variance | Variance of a correlated average, the bootstrap |
| 11 ✅ | Gradient boosting | Each new tree fits the negative gradient of the loss: descent in function space | Functional gradients, second-order (XGBoost) splits |

## Stage 4 · Neural networks

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 12 ✅ | Neurons and the multilayer perceptron | Logistic regression is one neuron; stacking them with a nonlinearity bends space | Function composition, why linear∘linear is linear, universal approximation |
| 13 ✅ | Backpropagation | The chain rule run backwards computes every gradient for the cost of about two forward passes | Jacobians, computational graphs, reverse-mode autodiff |
| 14 ✅ | Training deep networks | Keep signals and gradients at the right scale: initialisation, activations, normalisation, dropout | Variance propagation, Xavier and He init, batch and layer norm |

## Stage 5 · Deep architectures

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 15 ✅ | Convolutional networks | Share one small filter across every position: translation equivariance and far fewer weights | Discrete convolution, output-size arithmetic, receptive fields |
| 16 ✅ | Modern CNNs and residual networks | Depth works once each block learns a residual: LeNet → AlexNet → VGG → ResNet | Gradient flow through identity paths, 1×1 convolutions |
| 17 ✅ | Recurrent networks | One cell applied through time; gradients through time vanish or explode, and gates fix that | Backprop through time, products of Jacobians, LSTM/GRU gates |
| 18 ✅ | Word embeddings | Meaning as geometry: words that share contexts get nearby vectors | Skip-gram objective, negative sampling, PMI |
| 19 ✅ | Attention | Each output is a weighted average of values, with weights from query–key similarity | Softmax as soft lookup, scaled dot products and why √d |
| 20 ✅ | The Transformer | Self-attention + MLP + residuals + layer norm, stacked; positions added explicitly | Multi-head attention, masking, positional encodings, parameter counts |
| 21 ✅ | Vision transformers and contrastive learning | Images as patch sequences; CLIP learns by matching images to captions | InfoNCE loss, temperature, mutual information (light) |

## Stage 6 · Generative models

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 22 ✅ | Autoencoders and VAEs | Learn a latent space you can sample from by maximising a lower bound on likelihood | Latent-variable models, the ELBO, reparameterisation, Gaussian KL |
| 23 ✅ | Generative adversarial networks | A generator and a discriminator play a minimax game | Minimax objective, optimal discriminator, Jensen–Shannon divergence |
| 24 ✅ | Diffusion models | Destroy data with noise step by step, learn to undo one step at a time | Gaussian Markov chains, forward marginals, the noise-prediction loss (DDPM) |
| 25 ✅ | Score matching, guidance and latent diffusion | Diffusion learns the score ∇log p; guidance steers it; working in a latent space makes it cheap | Score functions, Langevin dynamics, SDE/ODE view, classifier-free guidance, flow matching |

## Stage 7 · Large language models

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 26 ✅ | Language modelling and tokenisation | A language model is next-token prediction; perplexity measures it; BPE decides the tokens | Chain rule of probability, cross-entropy per token, perplexity |
| 27 ✅ | Pretraining and scaling laws | A decoder-only transformer trained on the web; loss falls as a power law in compute, data and parameters | Power laws, compute ≈ 6ND, Chinchilla-optimal allocation |
| 28 ✅ | Fine-tuning and alignment | SFT teaches the format, RLHF and DPO teach preferences | Bradley–Terry model, KL-regularised RL, the DPO derivation |
| 29 ✅ | Making LLMs efficient | KV cache, quantisation, LoRA, mixture of experts, FlashAttention | Low-rank updates, memory arithmetic, IO-aware attention |
| 30 ✅ | In-context learning and reasoning | Prompts as conditioning; chain of thought, sampling, and test-time compute | Sampling temperature, top-p, self-consistency as majority vote |

## Stage 8 · Reinforcement learning (from Markov chains to AlphaZero)

Every algorithm gets its derivation, the reason it works (or the counterexample where it fails),
a from-scratch implementation on a small environment, and a problem set.

| # | Lesson | The one idea | Math pulled in |
|---|--------|--------------|----------------|
| 31 ✅ | Markov chains | The future depends only on the present; long-run behaviour is the stationary distribution | Transition matrices, stationary distributions, ergodicity, mixing, detailed balance |
| 32 ✅ | Bandits and exploration | Try options, learn which pays best, balance exploring and exploiting | Regret, ε-greedy, UCB and its confidence bound, Thompson sampling, the Lai–Robbins lower bound |
| 33 ✅ | Markov decision processes and the Bellman equations | Value = reward now + discounted value of what comes next | Returns, policies, V and Q, Bellman expectation and optimality equations |
| 34 ✅ | Dynamic programming | When the model is known, iterate the Bellman equation to the answer | Contraction mappings, Banach fixed point, policy evaluation, policy and value iteration, the policy improvement theorem |
| 35 ✅ | Monte Carlo and temporal-difference learning | Learn values from experience: wait for the return (MC) or bootstrap from the next estimate (TD) | Stochastic approximation, bias vs variance of targets, n-step returns, TD(λ) and eligibility traces |
| 36 ✅ | SARSA and Q-learning | Control without a model: on-policy vs off-policy learning of action values | Convergence conditions, maximisation bias, double Q-learning, the cliff-walking example |
| 37 ✅ | Function approximation and deep Q-networks | Replace the table with a network, and stabilise the bootstrapping | Semi-gradient TD, the deadly triad, replay buffers, target networks, Double/Dueling DQN |
| 38 ✅ | Policy gradients and actor–critic | Push up the log-probability of actions in proportion to their advantage | The policy gradient theorem, REINFORCE, baselines, actor–critic, GAE |
| 39 ✅ | Trust regions and PPO | Improve the policy without stepping too far from the data that measured it | Importance sampling, the performance difference lemma, natural gradient, TRPO, PPO's clipped objective |
| 40 ✅ | Model-based RL and planning | Learn a model of the world and plan inside it | Dyna, Monte Carlo tree search, UCT |
| 41 ✅ | AlphaGo, AlphaZero and MuZero | Policy net + value net + tree search, improved by self-play | PUCT, search as policy improvement, learned latent models |

RLHF and the RL used to train reasoning models are in stage 7 (lessons 28 and 30), which link back here.

## Stage 9 · Reading the papers

| # | Lesson | The one idea |
|---|--------|--------------|
| 42 | A guided tour of landmark papers | How to read a paper, and a map of the papers this curriculum rests on, each tied to the lesson that explains it |

## Problems book and assignments

Every lesson ends with short exercises and answers. On top of that:

- **Problem sets**, one per stage (P2 … P9, plus one for the math toolkit; P2 ✅, P3 ✅, P4 ✅, P5 ✅, P6 ✅ and P7 ✅ are written): 15–25 problems each, a mix of
  derivations, proofs, "compute this by hand" and short coding questions, graded easy → hard, with full
  worked solutions hidden behind a click. Together they form the problems book.
- **Assignments** for the important topics: a real programming project with starter code
  (`assignments/A<n>-<name>/starter.py`) and an autograder (`check.py`) that tests your implementation
  and tells you what is wrong. Planned:

| # | Assignment | Covers |
|---|-----------|--------|
| A1 ✅ | Linear and logistic regression from scratch, with regularisation and cross-validation | 01–05 |
| A2 | An optimiser zoo: SGD, momentum, Adam, schedules, on a real dataset | 06–08 |
| A3 ✅ | Gradient-boosted trees from scratch | 09–11 |
| A4 ✅ | A tiny autograd engine and an MLP trained on MNIST | 12–14 |
| A5 ✅ | A CNN with a residual block, hand-written convolution and its backward pass | 15–16 |
| A6 ✅ | A character-level GPT: attention, a transformer block, training and sampling | 19–20, 26 |
| A7 ✅ | A VAE and a DDPM diffusion model on 2D data and MNIST | 22–25 |
| A8 ✅ | DPO on a toy preference dataset; LoRA fine-tuning | 28–29 |
| A9 | Gridworld RL: policy/value iteration, Q-learning, SARSA | 31–36 |
| A10 | DQN on CartPole and PPO on a continuous-control task | 37–39 |
| A11 | AlphaZero for Connect Four (MCTS + self-play) | 40–41 |

## Math toolkit (read when a lesson sends you)

| # | Lesson | Used by |
|---|--------|---------|
| M1 ✅ | Linear algebra for ML: eigenvectors, SVD and PCA | 02, 05, 14, 18, 29 |
| M2 ✅ | Matrix calculus | 01, 06, 13, 14 |
| M3 ✅ | Probability for ML: expectation, Gaussians, MLE and MAP | 04, 05, 22, 24 |
| M4 ✅ | Information theory: entropy, cross-entropy, KL | 06, 09, 22, 23, 26 |
| M5 ✅ | Optimisation: convexity, constraints and Lagrange multipliers | 02, 05, 07 |
| M6 | Sampling and Monte Carlo | 07, 22, 24, 31, 35, 40 |

## Sources we lean on

Textbooks:
- Bishop, *Pattern Recognition and Machine Learning* (2006) and Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024)
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*; James et al., *An Introduction to Statistical Learning*
- Murphy, *Probabilistic Machine Learning: An Introduction* (2022) and *Advanced Topics* (2023)
- Goodfellow, Bengio & Courville, *Deep Learning* (2016)
- Prince, *Understanding Deep Learning* (2023)
- Zhang, Lipton, Li & Smola, *Dive into Deep Learning*
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- Cover & Thomas, *Elements of Information Theory*; Strang, *Linear Algebra and Learning from Data*

Courses: Stanford CS229 (ML), CS231n (vision), CS224n (NLP), CS236 (generative models),
Berkeley CS285 (deep RL), David Silver's RL course, Karpathy's *Neural Networks: Zero to Hero*.

Papers are cited in each lesson and collected in lesson 42.

## How this subject is laid out

```
tools (repo root):
  tools/inject_code.py   put a script and its real output into a lesson's @@CODE@@ marker
  tools/lesson_code.py   extract every lesson's code to code/ and check its recorded output still matches
  tools/check_links.py   fail on broken internal links in the built site (runs in CI)

lessons.json   the lesson list: title, stage, date added (read by the site)
meta.json      subject title, intro and stage names (read by the site)
lessons/       hand-written HTML lessons: the header and article only;
               the site adds navigation, contents and prev/next
figures/       SVG figures, generated by figures/make_*.py, pulled into lessons
               with <!-- figure: name --> so they follow the theme
pages/         hand-written HTML pages, like "Watch it learn"
widgets/       standalone interactive pages, like the gradient descent lab
assets/        learn.js (the canvas scene engine) and ml.css (lesson components)
code/          every lesson's "Code from scratch" block as a runnable script (generated)
assignments/   A<n>-<name>/starter.py + check.py (autograder)
sims/          from-scratch NumPy implementations
notes/         the original markdown concept notes and exercise sheets
```

## What you can safely skip for now

Measure theory, kernel methods and SVMs (historically important, rarely needed for deep
learning today), and convergence proofs beyond the quadratic case in lesson 02.
