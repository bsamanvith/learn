# Probability — the plan

From "what is a chance?" to martingales and measure theory, with a bridge to statistics,
information theory and ML at the end.

**Spine:** the core sequence of Harvard Stat 110 (Blitzstein & Hwang) and MIT 6.041
(Bertsekas & Tsitsiklis), which agree closely: probability models → conditioning →
random variables → distributions → joint behaviour → limit theorems → Markov processes.
We then go past both into stochastic processes, measure theory and applications.

**Mode:** probability (see [principles/subject-modes.md](../../principles/subject-modes.md)):
puzzle → intuition → model → compute → **simulate**.

Each part lists **concepts**, **signature visuals** (🎛 interactive, 🎞 animation,
🖼 static) and **signature problems**. Status column will be updated as chapters land.

---

## Part 0 — What is probability, anyway?  *(hook part, light)*

| Ch | Topic | Concepts |
|---|---|---|
| 0.1 | Three meanings of "probability" | Classical (equally likely), frequentist (long-run), Bayesian (degree of belief); propensity; why they mostly agree on the maths |
| 0.2 | A short history | Pascal–Fermat and the problem of points; Bernoulli; Laplace; Kolmogorov 1933 |
| 0.3 | Your intuition is broken (on purpose) | Preview of the paradoxes we'll solve later; why simulation is our friend |

- 🎛 Coin-flip convergence (running frequency → p) — *already built as the widget template*
- Problems: **The problem of points** (the 1654 letter), "Is 1/2 a fact about the coin or about you?"

## Part 1 — Foundations: sample spaces, axioms, counting

| Ch | Topic | Concepts |
|---|---|---|
| 1.1 | Sample spaces and events | Outcomes, events as sets, union/intersection/complement, De Morgan, partitions |
| 1.2 | The axioms | Kolmogorov's three axioms; consequences: complement rule, monotonicity, inclusion–exclusion, union bound |
| 1.3 | Naive probability & counting | Multiplication rule, permutations, combinations, sampling with/without replacement and ordered/unordered (the 2×2 table), stars and bars |
| 1.4 | Story proofs | Proving identities by counting two ways (Vandermonde, Pascal), binomial theorem |
| 1.5 | Inclusion–exclusion in action | Derangements, matching problem, surjections |

- 🎛 Venn diagram where area = probability (drag events, watch P(A∪B) update)
- 🖼 The sampling table (order matters? replacement?) with a picture for each cell
- 🎛 Birthday problem: slider on group size, plot P(match) vs n, simulate rooms
- Problems: **Birthday problem**, **De Méré's paradox** (one 6 in 4 rolls vs double-6 in 24), **Hat-check / derangements** (→ 1/e), poker hands, **Bose–Einstein stars and bars**

## Part 2 — Conditional probability and independence  *(the heart of the subject)*

| Ch | Topic | Concepts |
|---|---|---|
| 2.1 | Conditioning | P(A∣B) as zooming into B; multiplication rule; chain rule |
| 2.2 | Total probability & Bayes | Law of total probability, Bayes' rule, odds form, prior/likelihood/posterior |
| 2.3 | Independence | Pairwise vs mutual; conditional independence; independence ≠ disjointness |
| 2.4 | Conditioning as a problem-solving strategy | First-step analysis, conditioning on the right thing, symmetry |
| 2.5 | Famous traps | Monty Hall, Simpson's paradox, prosecutor's fallacy, base-rate neglect, two-child problem, Bertrand's box |

- 🎛 Grid of 1000 people for a medical test: sliders for prevalence, sensitivity, specificity → posterior
- 🎛 Probability tree builder (Bayes visually)
- 🎛 Monty Hall simulator (stay vs switch, n doors)
- 🖼 Simpson's paradox: slopes reversing when groups merge
- Problems: **Medical test**, **Monty Hall**, **Gambler's ruin** (first-step analysis), **Simpson's kidney stones**, **Prosecutor's fallacy (Sally Clark)**, **Bertrand's box**

## Part 3 — Discrete random variables

| Ch | Topic | Concepts |
|---|---|---|
| 3.1 | Random variables | RV as a function on Ω; PMF; CDF; functions of RVs |
| 3.2 | The named distributions & their stories | Bernoulli, Binomial, Hypergeometric, Geometric, Negative Binomial, Poisson, Discrete Uniform; how they're related |
| 3.3 | Expectation | Definition, linearity (even without independence!), indicator method, LOTUS |
| 3.4 | Variance | Variance, standard deviation, variance of sums, Poisson as Binomial limit |
| 3.5 | Clever expectations | Indicator tricks, symmetry, coupon collector, expected number of records/fixed points |

- 🎛 Distribution explorer: pick a distribution, slide parameters, see PMF + mean ± sd
- 🎞 Binomial → Poisson as n→∞, np fixed
- 🎛 Galton board
- Problems: **Coupon collector**, **Expected fixed points (always 1!)**, **St. Petersburg paradox**, **Banach matchbox**, capture–recapture (hypergeometric)

## Part 4 — Continuous random variables

| Ch | Topic | Concepts |
|---|---|---|
| 4.1 | Densities | PDF, CDF, why P(X = x) = 0, expectation and variance with integrals |
| 4.2 | Uniform and universality | Uniform; inverse-CDF sampling (any distribution from a uniform) |
| 4.3 | Normal | Standard normal, standardisation, 68–95–99.7, why it's everywhere (preview of CLT) |
| 4.4 | Exponential and memorylessness | Exponential, memoryless property, competing exponentials |
| 4.5 | Gamma, Beta and friends | Gamma (sum of exponentials), Beta (order statistics, Bayesian prior), chi-square, t, Cauchy (no mean!) |
| 4.6 | Transformations | Change of variables (1D), convolution, Box–Muller |
| 4.7 | Order statistics | Min/max, k-th order statistic, Beta connection |

- 🎛 Inverse-CDF sampler: uniform dots on y-axis map to samples on x-axis
- 🎛 Density = histogram limit (bins shrink as samples grow)
- 🎞 Cauchy running mean refusing to settle vs Normal running mean
- Problems: **Broken stick** (triangle probability), **Buffon's needle** (estimate π), **Bus waiting-time paradox**, **Bertrand's chord paradox** (what "random" means)

## Part 5 — Joint distributions

| Ch | Topic | Concepts |
|---|---|---|
| 5.1 | Joint, marginal, conditional | Discrete and continuous joint distributions; independence of RVs |
| 5.2 | Covariance and correlation | Covariance, correlation, uncorrelated ≠ independent, variance of sums revisited |
| 5.3 | Multinomial and multivariate normal | Multinomial; bivariate/multivariate normal, contours, linear transforms |
| 5.4 | Conditional expectation | E[Y∣X] as a random variable; Adam's law (tower), Eve's law (total variance); best predictor |
| 5.5 | Transformations in 2D | Jacobians, polar coordinates, sums and ratios of RVs |
| 5.6 | Generating functions | PGFs, MGFs, characteristic functions (preview); moments; sums of independent RVs |

- 🎛 Correlation explorer: scatter cloud with ρ slider, regression line vs E[Y∣X]
- 🎛 Bivariate normal contours (3D surface ↔ contour map)
- 🖼 Uncorrelated-but-dependent gallery
- Problems: **Two envelopes paradox**, **Random sums (Wald)**, **Chicken–egg (Poisson thinning)**, **Expected value of max of normals**

## Part 6 — Inequalities and limit theorems

| Ch | Topic | Concepts |
|---|---|---|
| 6.1 | Inequalities | Markov, Chebyshev, Jensen, Cauchy–Schwarz, Chernoff/Hoeffding (tail bounds) |
| 6.2 | Modes of convergence | In probability, almost surely, in distribution, in L²; how they relate |
| 6.3 | Law of large numbers | Weak LLN (via Chebyshev), strong LLN (statement + intuition); Monte Carlo |
| 6.4 | Central limit theorem | Statement, MGF proof sketch, normal approximation, continuity correction, when it fails |
| 6.5 | Beyond the CLT | Poisson limit, Berry–Esseen (rate), large deviations (taste) |

- 🎞🎛 CLT machine: pick any starting distribution, slide n, watch sums become normal
- 🎛 Chebyshev vs actual tail (how loose is the bound?)
- 🎛 Monte Carlo π
- Problems: **How many polls for ±3%?**, **Is this coin fair?** (CLT test), **Jensen and the AM-GM inequality**, **Hoeffding for A/B tests**

## Part 7 — Stochastic processes

| Ch | Topic | Concepts |
|---|---|---|
| 7.1 | Markov chains | Transition matrices, n-step, classification of states, stationary distribution, convergence, reversibility |
| 7.2 | Random walks | Simple random walk, reflection principle, ballot problem, recurrence in 1D/2D vs 3D (Pólya) |
| 7.3 | Poisson processes | Arrivals, interarrival exponentials, thinning and superposition, conditioning on counts |
| 7.4 | Branching processes | Galton–Watson, extinction probability via PGFs |
| 7.5 | Martingales | Definition, fair games, optional stopping, gambler's ruin revisited, ABRACADABRA |
| 7.6 | MCMC | Metropolis–Hastings, Gibbs sampling: Markov chains as computers |
| 7.7 | Brownian motion (preview) | Random walk scaling limit, basic properties |

- 🎛 Markov chain playground: draw states/arrows, watch the distribution flow to stationary
- 🎞 Random walks in 1D, 2D, 3D — returns or not?
- 🎛 Poisson process timeline with thinning colours
- 🎞 Brownian motion as zoomed-out random walk
- Problems: **Gambler's ruin (martingale proof)**, **ABRACADABRA monkey**, **PageRank**, **Pólya's urn**, **Extinction of surnames**, **Ballot problem**

## Part 8 — Measure-theoretic foundations  *(going properly deep)*

| Ch | Topic | Concepts |
|---|---|---|
| 8.1 | Why we need measure theory | Vitali set: not every subset can have a probability; Bertrand revisited |
| 8.2 | σ-algebras and probability spaces | σ-algebras, Borel sets, (Ω, 𝓕, P), π-λ / Carathéodory (statements + intuition) |
| 8.3 | Random variables as measurable functions | Measurability, distributions as pushforward measures |
| 8.4 | Lebesgue integration & expectation | Simple functions → integral; monotone & dominated convergence; Fatou |
| 8.5 | Independence, Borel–Cantelli, 0–1 laws | Product measures, Borel–Cantelli lemmas, Kolmogorov 0–1 |
| 8.6 | Conditional expectation done right | Conditioning on a σ-algebra, Radon–Nikodym (idea), filtrations |
| 8.7 | Proofs we postponed | Strong LLN, CLT via characteristic functions, martingale convergence |

- 🖼 Riemann vs Lebesgue integral (slice vertically vs horizontally)
- 🎛 Borel–Cantelli: events with Σ P < ∞ vs = ∞, watch which happen infinitely often
- Problems: **Infinite monkey theorem**, **Is "a random integer" possible?**, construct a non-measurable set, dominated convergence counterexamples

## Part 9 — Bridges  *(where probability goes next)*

| Ch | Topic | Concepts |
|---|---|---|
| 9.1 | Information theory | Entropy, cross-entropy, KL divergence, mutual information, coding |
| 9.2 | From probability to statistics | Likelihood, MLE, Bayesian inference, conjugate priors (Beta–Binomial), confidence vs credible |
| 9.3 | Randomised algorithms | Quicksort, hashing, Bloom filters, Monte Carlo vs Las Vegas, the probabilistic method |
| 9.4 | Probability in ML | Naive Bayes, generative models, softmax as distribution, variational inference (taste), diffusion models as noise processes |
| 9.5 | Probability in the sciences | Statistical mechanics (Boltzmann), genetics (Hardy–Weinberg, Wright–Fisher), radioactive decay, queues |

- 🎛 Beta–Binomial prior → posterior updater (flip coins, watch belief sharpen)
- 🎛 Entropy of a distribution as you reshape it
- Problems: **Estimate a coin's bias**, **German tank problem**, **20 Questions and entropy**, **Bloom filter false positives**

## Appendix — Paradox gallery & reference

- **Paradox gallery** (each with a simulator): Monty Hall · Birthday · Simpson's · Bertrand's chord · Bertrand's box · St. Petersburg · Two envelopes · Boy-or-girl · Inspection (bus) paradox · Gambler's fallacy · Sleeping Beauty · Parrondo's paradox
- **Distribution cheat-sheet:** story, PMF/PDF, mean, variance, MGF, relationships diagram
- **Notation table**
- **Formula sheet**

---

## Sources

| Source | Role | Notes |
|---|---|---|
| Blitzstein & Hwang, *Introduction to Probability* (2nd ed., CRC) | **Primary spine** | Story proofs, conditioning-first; free PDF and Stat 110 lectures/problem sets from the authors |
| Bertsekas & Tsitsiklis, *Introduction to Probability* (2nd ed., Athena) | Second voice | Engineering style, clear; MIT 6.041 OCW lectures + recitations |
| Ross, *A First Course in Probability* | Problem bank | Huge set of exercises incl. theoretical problems |
| Grimmett & Stirzaker, *Probability and Random Processes* | Depth for Parts 6–8 | Rigorous, excellent problems (with companion *One Thousand Exercises*) |
| Williams, *Probability with Martingales* | Part 8 + martingales | Short, beautiful measure-theoretic intro |
| Feller, *An Introduction to Probability Theory and Its Applications*, Vol. 1 | Culture & classic problems | Random walks, combinatorial gems |
| Seeing Theory (Brown University) | Visual inspiration | Interactive probability/statistics visuals |
| 3Blue1Brown | Visual inspiration | Bayes, CLT, convolutions videos |

## Order of work

1. Part 0 + Part 1 first (set the style; build the Venn and birthday widgets).
2. Part 2 next — the most important and most fun part; the medical-test grid widget.
3. Then strictly in order; Part 8 can be read after Part 6 by the brave.

## Progress

| Part | Status |
|---|---|
| 0 | idea |
| 1 | idea |
| 2 | idea |
| 3 | idea |
| 4 | idea |
| 5 | idea |
| 6 | idea |
| 7 | idea |
| 8 | idea |
| 9 | idea |
