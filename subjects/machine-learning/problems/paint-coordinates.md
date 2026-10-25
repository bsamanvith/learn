---
title: "Paint a picture with a neural network"
kind: assignment
time: "90–120 min"
after: "14"
---

# Paint a picture with a neural network

Here is a strange way to store a picture: train a neural network that takes a pixel's
coordinates $(x, y)$ and returns its brightness. The weights *are* the image. Tonight you
paint a smiley face this way and run into a surprise: a perfectly good MLP paints only a
blurry blob, however long you train it. Then one line of trigonometry fixes it, and
turning that line up too far makes it fail in a completely different way. The same trick
sits inside NeRF and inside the positional encodings of transformers.

**Deliverable:** one script, `paint.py`, and one figure: the target next to the network's
paintings for each input encoding, with the test PSNR in each title.

## Before you start

- Lessons 12 to 14 (MLPs, backpropagation, training). Reuse your backprop from the
  [digits assignment](../problems/digits-numpy.md) if you did it, and add Adam (lesson 07).
- The picture: a 64×64 grid of coordinates in $[-1, 1]^2$ (`np.meshgrid` of
  `np.linspace(-1, 1, 64)`) and a smiley with soft edges about a pixel wide. With
  `soft = lambda z: 1 / (1 + np.exp(-z / 0.03))`: face `0.8 * soft(0.85 - r)` with
  `r = np.hypot(x, y)`; multiply by `1 - soft(0.12 - dist)` for each eye at $(\pm0.32, -0.25)$;
  and by `1 - soft(0.06 - abs(np.hypot(x, y + 0.05) - 0.5)) * soft(y - 0.15)` for the mouth.
- Train on half the pixels in a checkerboard pattern and test on the other half, so
  "test" means painting the pixels in between. Score with PSNR, $10\log_{10}(1/\text{MSE})$
  in decibels: higher is better, and every 10 dB is ten times less error.

**You should see:** predicting the average brightness everywhere scores 9.3 dB. Any
network must beat that.

## Step 1 · The obvious network

An MLP $2 \to 128 \to 128 \to 1$ with ReLUs, He initialisation (`default_rng(0)`), mean
squared error, full-batch Adam with learning rate $2\times10^{-3}$, 2,000 steps.

**Predict first:** after 2,000 steps, will the face's edges be smooth and round?

**You should see:** train 24.8 dB and test 24.7 dB. Draw it: the face is there, but
creased by straight streaks radiating across it, with wobbly outlines and blotchy eyes.
A ReLU network of raw coordinates is a piecewise-linear function, and the pieces it has
found are far too coarse for round edges. Train and test agree, so this isn't
overfitting: the network is slow to learn fine detail at all. Neural networks fit the
low frequencies of a function first and the high ones very slowly, which is called
**spectral bias**.

## Step 2 · Fourier features

Before the network, map each coordinate $v = (x, y)$ to
$\gamma(v) = \big(\sin(2\pi Bv), \cos(2\pi Bv)\big)$, where $B$ is a $64\times2$ matrix of
random frequencies, $B = s\cdot G$ with $G$ from `default_rng(1).normal(0, 1, (64, 2))`.
The network now has 128 inputs. Try scales $s = 0.5, 1, 2, 5, 10$.

**You should see:**

| input | raw $(x, y)$ | $s = 0.5$ | $s = 1$ | $s = 2$ | $s = 5$ | $s = 10$ |
|---|---|---|---|---|---|---|
| train PSNR | 24.8 | 46.5 | 55.4 | 311.7 | 310.4 | 310.0 |
| test PSNR | 24.7 | **40.2** | 32.7 | 23.2 | 11.1 | 7.0 |

At $s = 0.5$ the painting is almost indistinguishable from the target, and the in-between pixels come out right: 15 dB better
than raw coordinates, a thirtyfold drop in error. The sines hand the network the
high frequencies it couldn't build for itself.

## Step 3 · Break it: too many frequencies

Look at $s = 5$ and $s = 10$: training PSNR above 300 dB, which is floating-point exact,
and test PSNR worse than painting a flat grey. **Draw the full 64×64 output** for $s = 10$:
the training pixels are perfect and the pixels between them are static. With frequencies
far above what the picture contains, the network can fit any values at the training
pixels, so it interpolates with noise. The scale $s$ is a smoothness knob, and it plays
the role that $\lambda$ played in the [overfitting assignment](../problems/overfit.md): too low
underfits, too high memorises.

## Deliverable

`paint.py` printing the table, and the figure of seven panels (the target and six
paintings). Add one sentence: what does $s$ control, and how would you choose it for a
photograph you had never seen?

## Stretch

- **Your own picture.** Load any image with `plt.imread`, convert to grey, and downsample
  to 128×128. Which $s$ is best now, and does it depend on how detailed the photo is?
- **Learning curves.** Record test PSNR every 100 steps for raw coordinates. Does it keep
  creeping up? How long would it take to match $s = 0.5$?
- **Colour and motion.** Make the output 3 numbers (RGB), or add time as a third input
  and fit a short animation.
- **The transformer connection.** Lesson 20's sinusoidal positional encoding is
  $\gamma$ with frequencies on a geometric ladder instead of random ones. Try that ladder
  here.

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/paint_coordinates.py`](../sims/paint_coordinates.py) prints every checkpoint on
this page, and draws the figure with `--plot`. It takes about a minute. Its output:

```text
Step 0 · a 64×64 drawing; train on 2048 pixels (a checkerboard), test on the other 2048
         predicting the mean brightness everywhere scores 9.3 dB

         input                          train PSNR   test PSNR
         raw (x, y)                        24.8 dB      24.7 dB
         Fourier, scale 0.5                46.5 dB      40.2 dB
         Fourier, scale 1                  55.4 dB      32.7 dB
         Fourier, scale 2                 311.7 dB      23.2 dB
         Fourier, scale 5                 310.4 dB      11.1 dB
         Fourier, scale 10                310.0 dB       7.0 dB
```

</details>
