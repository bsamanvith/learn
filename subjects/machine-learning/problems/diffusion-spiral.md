---
title: "A diffusion model draws a spiral"
kind: assignment
time: "2 hours"
after: "24"
---

# A diffusion model draws a spiral

Diffusion models make images by starting from pure noise and removing a little of it,
two hundred times. Tonight you build one small enough to watch: it learns 500 points on a
spiral and then draws new ones. Along the way you meet the most surprising fact about
diffusion. The *perfect* denoiser for your training set, which you can write down exactly,
is useless: all it can draw is copies of the training points. It is the network's
imperfection that makes it creative. Then you break the sampler twice and watch everything
collapse to a dot, then explode.

**Deliverable:** one script, `diffuse.py`, and one figure with five scatter plots: the
data, the exact denoiser's samples, the trained network's samples, and the two broken
samplers.

## Before you start

- Lesson 24 (diffusion models). Problem 26 of the
  [hard ML volume](../lessons/H2-hard-problems-ml.html) derives the two Gaussians used here,
  and problem 27 is Tweedie's formula.
- Your MLP with backprop and Adam from the [painting assignment](../problems/paint-coordinates.md)
  or the [digits assignment](../problems/digits-numpy.md).
- Data: with `rng = np.random.default_rng(0)`, angles `s = np.sqrt(rng.uniform(0, 1, 500)) * 3 * np.pi + np.pi / 2`,
  points $(s\cos s, s\sin s)/(3\pi)$, plus $N(0, 0.02^2)$ noise from the same `rng`. To score
  samples, make a dense noiseless curve the same way (20,000 points, `default_rng(99)`).
- Schedule: $T = 200$, $\beta_t$ evenly spaced from $10^{-4}$ to $0.08$,
  $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s\le t}\alpha_s$.

**You should see:** $\bar\alpha_T = 2.66\times10^{-4}$, so $x_T$ keeps only 1.6% of the
signal: pure noise, as far as the eye can tell.

## Step 1 · Noise the data

Sample $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$ at $t = 100$ for
200,000 random training points, and compare the variance per coordinate with
$\bar\alpha_t\operatorname{Var}(x_0) + 1 - \bar\alpha_t$.

**You should see:** variances 0.908 and 0.925 against the predicted 0.906 and 0.922.
**Plot** $x_t$ at $t = 10, 50, 100, 200$: the spiral thickens, blurs, and dissolves into a
round Gaussian cloud.

## Step 2 · The perfect denoiser

To run the process backwards you need $\hat x_0(x_t, t) = E[x_0 \mid x_t]$. For a dataset
of 500 points this is exact and needs no learning: Bayes gives each training point $x^{(i)}$
the weight $w_i \propto \exp\big(-\lVert x_t - \sqrt{\bar\alpha_t}x^{(i)}\rVert^2/(2(1 - \bar\alpha_t))\big)$,
and $\hat x_0 = \sum_iw_ix^{(i)}$. (Use the log-sum-exp trick.) Sample 1,000 points: start
at $x_T \sim N(0, I)$ and for $t = T, \dots, 2$ step to
$x_{t-1} \sim N(\tilde\mu, \tilde\beta_tI)$ with
$\tilde\mu = \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}\hat x_0$
and $\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$; at the last step output $\hat x_0$.
Use `default_rng(1)` for sampling.

Score the samples two ways: median distance to the true spiral, and median distance to the
nearest *training* point. For comparison, score 1,000 fresh points from the spiral itself.

**You should see:**

| samples | to the spiral | to the nearest training point |
|---|---|---|
| exact denoiser | 0.0093 | **0.0046** |
| fresh real data | 0.0149 | 0.0154 |

The exact denoiser's samples sit three times closer to training points than genuinely new
data would. It is drawing the training set back, give or take the last step's blur. It
had to: the exact $E[x_0 \mid x_t]$ for a finite dataset knows nothing except those 500
points.

## Step 3 · A network that has to guess

Train an MLP $18 \to 128 \to 128 \to 128 \to 2$ (ReLU) to predict the noise $\varepsilon$ from
$x_t$ and a time embedding: 8 sines and 8 cosines of $\frac tT f_k$ with frequencies $f_k$
spaced geometrically from 1 to 1000. Each step: draw a batch of 256 training points, times
$t$ and noises $\varepsilon$, form $x_t$, and minimise $\lVert\hat\varepsilon - \varepsilon\rVert^2$.
Use Adam at $10^{-3}$ for 10,000 steps, then $3\times10^{-4}$ for 5,000 more
(`default_rng(2)` for the weights, `default_rng(3)` for batches). Convert its prediction
to $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t}$ and sample exactly as in Step 2.

**You should see:** the loss falling from 1.81 to 0.222 after 1,000 steps and 0.152 at
the end, and then:

| samples | to the spiral | to the nearest training point |
|---|---|---|
| trained MLP | 0.0164 | **0.0148** |

The network's samples are as far from the training points as fresh data is (0.0148 against
0.0154), and almost as close to the spiral. It has learned the *shape* rather than the
points. **Draw it:** a clean spiral, with a few strays between the arms near the centre.
The network is too smooth to represent the exact denoiser, and that smoothness is what
generalises.

## Step 4 · Break the sampler, twice

1. **Leave out the noise.** Sample with $x_{t-1} = \tilde\mu$, skipping the random term.
   **You should see:** all 1,000 samples land on a single spot: standard deviation 0.006 and
   0.017, against the data's 0.54 and 0.64. Without the injected noise every path follows
   the mean and they all flow to the same place. The noise is not a nuisance; it is what
   makes different samples different. (Deterministic samplers like DDIM avoid this with a
   different update, not by dropping the noise from this one.)
2. **Skip the steps.** Jump straight from $x_T$ to the network's $\hat x_0$ in one go.
   **You should see:** a structureless spray with standard deviation 1.3 and 1.7, median
   distance 0.72 from the spiral. At $t = T$, $\hat x_0$ divides by
   $\sqrt{\bar\alpha_T} = 0.016$, so every small error in $\hat\varepsilon$ is magnified 60 times.
   Small steps are how diffusion avoids having to be right all at once.

## Deliverable

`diffuse.py` printing every checkpoint above and the five-panel figure. Add one sentence:
if a perfect denoiser can only copy, what exactly are we relying on when a diffusion model
produces a new image?

## Stretch

- **More data, less copying.** Train the network on 50 points instead of 500. Do its
  samples start to cling to the training points?
- **Fewer steps.** Use $T = 50$ with $\beta$ up to 0.3 and retrain. What happens to the
  quality, and why do large-$T$ schedules help?
- **Classifier-free guidance.** Split the spiral into its inner and outer half, give the
  network a class input (with a "no class" option during training), and sample with
  guidance weights 0, 1 and 3, as in problem 28 of the hard ML volume.

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/diffusion_spiral.py`](../sims/diffusion_spiral.py) prints every checkpoint on this
page and draws the figure with `--plot`. It takes under a minute. Its output:

```text
Step 0 · 500 training points on a spiral; T = 200 steps, β from 1e-04 to 0.08
         ᾱ_T = 2.66e-04: x_T keeps √ᾱ_T = 0.016 of the signal
Step 1 · at t = 100: E[x_t] = [-0.103 -0.025], Var(x_t) per coordinate = [0.908 0.925]  (theory: ᾱ Var(x0) + 1 − ᾱ = [0.906 0.922])
Step 2 · sampling with the exact optimal denoiser of the training set
         ideal denoiser                     median distance to the spiral 0.0093   to the nearest training point 0.0046
Step 3 · training an MLP 18 → 128 → 128 → 128 → 2 to predict the noise, 15000 Adam steps of batch 256
         step      1: noise-prediction loss 1.8127
         step   1000: noise-prediction loss 0.2220
         step   5000: noise-prediction loss 0.1831
         step  15000: noise-prediction loss 0.1521
         trained MLP                        median distance to the spiral 0.0164   to the nearest training point 0.0148
         (fresh real data, for comparison)  median distance to the spiral 0.0149   to the nearest training point 0.0154
Step 4 · breaking the sampler
         no noise in the reverse steps      median distance to the spiral 0.0064   to the nearest training point 0.0218
         spread of samples: std [0.006 0.017] against the data's [0.54  0.639]
         jump from x_T straight to x̂0      median distance to the spiral 0.7210   to the nearest training point 0.6899
         spread of one-shot samples: std [1.31  1.705]
```

Training is stochastic in a way that depends on the order of random draws, so your losses
and distances may differ in the second decimal place.

</details>
