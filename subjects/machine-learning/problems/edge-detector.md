---
title: "Build an edge detector, then teach one"
kind: assignment
time: "90 min"
after: "15"
---

# Build an edge detector, then teach one

Before convolutional networks learned their filters, people designed them by hand. Tonight
you do both. First you write convolution yourself, hand-design the Sobel edge detector,
and check the two properties that make convolution special. Then you start a 3×3 kernel
at zero and let gradient descent rediscover Sobel from examples. It does, to fourteen
decimal places, unless you train it on the wrong picture, and then it learns a different
filter that looks perfect on its training image and fails everywhere else.

**Deliverable:** one script, `edges.py`, and one figure: the drawing, its Sobel-x map, its
edge magnitude, and what the badly trained kernel sees.

## Before you start

- Lesson 15 (convolutional networks) and lesson 02 for gradient descent on a quadratic.
- Draw a 64×64 image with `yy, xx = np.mgrid[0:64, 0:64]`: a white square
  `img[10:30, 8:28] = 1`, a grey disc of radius 11 at $(44, 20)$ with value 0.6, a diagonal
  bar `(abs((xx - 6) - (yy - 34)) < 2) & (34 < yy) & (yy < 60)` with value 0.8, and a striped patch
  `img[40:56, 36:60] = 0.3 + 0.3 * (xx % 4 < 2)` (on that slice). Everything else is 0.
- Convolution, as libraries define it (strictly, cross-correlation): output $(i, j)$ is
  $\sum_{k,l}\text{img}[i + k, j + l]\,K[k, l]$, only where the kernel fits ("valid").
  `np.lib.stride_tricks.sliding_window_view(img, (3, 3))` gives every 3×3 window at once;
  one `np.einsum` finishes the job.

## Step 1 · Sobel by hand

$K_x = \begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix}$ measures how fast brightness
rises to the right, smoothing a little vertically; $K_y = K_x^\top$. The edge strength is
$\sqrt{g_x^2 + g_y^2}$.

**You should see:** 62×62 output maps. The largest $|g_x|$ is 4.00 (a full black-to-white
step: $1 + 2 + 1$), the largest magnitude 4.243, and 904 pixels have magnitude above 1.
Flat regions give exactly 0, because the kernel's weights sum to 0. **Draw** $g_x$ and the
magnitude: the square's left and right sides light up in $g_x$ (with opposite signs), its
top and bottom don't, and the magnitude outlines everything, including every stripe.

## Step 2 · The two properties that matter

1. **Equivariance.** Shift the image by $(3, 5)$ pixels with `np.roll` and convolve again.
   Compare with the shifted edge map, away from the wrapped border.
2. **Composition.** Blur with the 3×3 box filter (all weights $\frac19$), then apply Sobel.
   Find the single 5×5 kernel that does both at once, and check it.

**You should see:** a maximum difference of exactly 0 for the shift, and about
$7\times10^{-16}$ for blur-then-Sobel against the single 5×5 kernel, whose rows each sum to 0.
Two stacked linear convolutions are one bigger convolution, which is why CNNs need
nonlinearities between layers to gain anything from depth.

## Step 3 · Teach a kernel

Pretend you don't know Sobel. You have the drawing and its $g_x$ map, and you want the
3×3 kernel $K$ that minimises the mean squared difference between `conv(img, K)` and $g_x$.
Stack every 3×3 patch as a row of a matrix $P$ (3,844 rows, 9 columns); the problem is
then least squares, $\min_k\frac1n\lVert Pk - t\rVert^2$. Compute the Hessian $\frac2nP^\top P$
and its eigenvalues, then run gradient descent from $k = 0$ with step $1/\lambda_{\max}$.

**Predict first:** how many steps until the kernel matches Sobel to 6 decimals?

**You should see:** eigenvalues from $3.73\times10^{-3}$ to $2.751$, a condition number of
737, and a step of 0.364. The error falls slowly, then fast:

| steps | 100 | 1,000 | 10,000 | 50,000 |
|---|---|---|---|---|
| max $\lvert k - K_x\rvert$ | 0.448 | 0.082 | $4.0\times10^{-7}$ | $4.5\times10^{-14}$ |

It recovers Sobel exactly, because the drawing has edges in every direction, so every
patch direction is represented. The slowest direction (smallest eigenvalue) sets the pace,
just as in lesson 02.

## Step 4 · Break it: learn from a boring picture

Now train on a smooth Gaussian blob, `np.exp(-((xx - 30)**2 + (yy - 34)**2) / (2 * 14.0**2))`,
with its own Sobel-x output, again from zero for 50,000 steps at step $1/\lambda_{\max}$.

**You should see:** a condition number of $2.4\times10^{12}$, and a learned kernel that is
**not** Sobel (max error 0.665):

$$\begin{pmatrix}-1.33&0&1.33\\-1.34&0&1.33\\-1.33&0&1.33\end{pmatrix}$$

On the blob its output matches Sobel's to 0.0001. On the drawing it is off by up to 0.53.
On a smooth image every 3×3 patch is nearly a tilted plane, so the data only pins down
the total left-to-right difference, $\sum$ (right column) $-$ $\sum$ (left column) $= 4$. Gradient
descent from zero ends near the smallest kernel with that total, which spreads it evenly over
the three rows. The data never told it how to weight them. **Draw** this kernel applied to the
drawing: it still finds vertical edges, but the stripes and the disc come out differently.
A network learns only what its training data can distinguish.

## Deliverable

`edges.py` printing every checkpoint, and the figure. Add one sentence: what property of
a training image makes a filter identifiable, in terms of the matrix $P^\top P$?

## Stretch

- **A noisy target.** Add $N(0, 0.1^2)$ noise to $g_x$ in Step 3. How far is the learned
  kernel from Sobel now, and how does that depend on the number of patches?
- **Learn a blur.** Repeat Step 3 with the box filter as the target. Is it easier or harder
  to identify than Sobel on the drawing? Look at the eigenvectors of $P^\top P$.
- **A tiny CNN.** Learn two 3×3 kernels followed by a ReLU and a sum, so that the output
  matches the edge magnitude $\sqrt{g_x^2 + g_y^2}$ as well as possible. What do the two kernels
  become?

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/edge_detector.py`](../sims/edge_detector.py) prints every checkpoint on this page,
and draws the figure with `--plot`. Its output:

```text
Step 1 · image 64×64 → edge maps 62×62 ('valid' convolution)
         largest |gx| = 4.00, largest magnitude = 4.243; pixels with magnitude > 1: 904
         a flat region gives exactly 0.0; sum of the Sobel weights = 0
Step 2 · shift the image by (3, 5): edge map shifts by (3, 5), max difference 0.0e+00
         blur then Sobel = one 5×5 kernel: max difference 6.7e-16; 5×5 kernel row sums [0. 0. 0. 0. 0.]
Step 3 · learn a 3×3 kernel from (image, Sobel-x output), starting at zero, step 1/λ_max = 0.364
         Hessian eigenvalues from 3.73e-03 to 2.751: condition number 737
            100 steps: max |k − Sobel-x| = 4.48e-01
           1000 steps: max |k − Sobel-x| = 8.21e-02
          10000 steps: max |k − Sobel-x| = 4.01e-07
          50000 steps: max |k − Sobel-x| = 4.46e-14
Step 4 · train on a smooth blob instead: condition number 2.41e+12
         after 50000 steps: max |k − Sobel-x| = 0.665
         its output error on the blob: 0.0001; on the drawn scene: 0.532 (Sobel's largest output is 4)
```

</details>
