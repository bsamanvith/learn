---
title: "Handwritten digits in pure NumPy"
kind: assignment
time: "2 hours"
after: "14"
---

# Handwritten digits in pure NumPy

No framework, no autograd: a two-layer neural network written with nothing but matrix
products, trained on real handwritten digits until it reads 97% of the ones it has never
seen. Every gradient is one you derived in lesson 13, and you will check each one
numerically before trusting it. Then you break the network three ways and see each
failure in the numbers.

**Deliverable:** one script, `digits.py`, and two plots: learning curves, and a strip of
the test digits it gets wrong.

## Before you start

- Lessons 12 to 14: the MLP, backpropagation, and initialisation.
- Data: scikit-learn ships 1,797 handwritten digits as 8×8 images, so nothing downloads.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits()
X, y = d.data / 16.0, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
```

**You should see:** 1,347 training and 450 test images, pixels between 0 and 1. Show a
few with `plt.imshow(X[i].reshape(8, 8))` before going on: they are tiny and blurry,
and you can still read them.

## Step 1 · The forward pass, and the loss you should start at

The network is $h = \mathrm{ReLU}(XW_1 + b_1)$ with 64 hidden units, then
$P = \operatorname{softmax}(hW_2 + b_2)$ and the mean cross-entropy. Initialise with
`rng = np.random.default_rng(0)`, $W_1 \sim N(0, 2/64)$ (He), $W_2 \sim N(0, 1/64)$, and
zero biases. Subtract each row's maximum before the exponential.

**Predict first:** before any training, what loss should a network that knows nothing
have, on 10 classes?

**You should see:** 2.377, close to $\ln10 = 2.303$: near-uniform guesses. If your
starting loss is far from $\ln10$, something is wrong before you've trained at all.

## Step 2 · Backpropagation, checked

Write the backward pass: $\partial L/\partial z_2 = (P - Y)/n$ with $Y$ one-hot, then
$\partial L/\partial W_2 = h^\top\,\partial L/\partial z_2$, back through $W_2^\top$, multiply
by the ReLU's mask $[z_1 > 0]$, and finally $\partial L/\partial W_1 = X^\top\,\partial L/\partial z_1$.
Check five individual weights against central differences,
$\frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$ with $\epsilon = 10^{-5}$, on the
first 50 training images.

**You should see:** a worst relative error around $3\times10^{-8}$. Anything above
$10^{-4}$ means a bug. (A kink of the ReLU right at a checked point can spoil one check;
pick another weight.)

## Step 3 · Train it

Mini-batch SGD: shuffle each epoch, batches of 32, learning rate 0.1, 30 epochs, with
`rng` continuing from the initialisation. Record the training loss and test accuracy
after each epoch.

**You should see:**

| epoch | 1 | 5 | 10 | 30 |
|---|---|---|---|---|
| train loss | 1.437 | 0.409 | 0.169 | 0.073 |
| test accuracy | 52.9% | 88.7% | 95.6% | **97.3%** |

That is 12 mistakes out of 450. Build the confusion matrix and look at them: 8s read as
1s, 4s as 8s, 1s as 8s. **Plot 2:** the misread digits with their true and predicted
labels. Several are hard for a human too.

## Step 4 · Break it, three ways

1. **No hidden layer.** Fit scikit-learn's `LogisticRegression(max_iter=2000)`, which is
   softmax regression, lesson 06. **You should see:** 96.9%. On these small, centred
   digits a linear model is almost as good, which is a useful humility check before
   reaching for depth.
2. **Learning rate 3.** **You should see** the training loss jump to 5.57 after the first
   epoch and then wander around 2.4 to 2.6, never coming down, and test accuracy 10.2%:
   chance. The ReLUs have been knocked dead or the softmax saturated, and the network
   has stopped learning.
3. **Bad initialisation**, $W_1 \sim N(0, 1)$: **you should see** a starting loss of
   4.51, nearly double $\ln10$, because the logits start huge and confidently wrong. In
   this shallow network training still recovers (96.9% after 30 epochs). Lesson 14 shows
   why the same mistake is fatal 20 layers deep.
4. **Bonus: too narrow.** With only 4 hidden units, test accuracy is 78.2%: four ReLU
   features cannot separate ten classes.

## Deliverable

`digits.py` printing every checkpoint above, the learning-curve plot (training loss
and test error on a log scale), and the mistakes strip. Add one sentence: what does the
hidden layer buy you here over softmax regression, and how would you find out whether
more would help?

## Stretch

- **Momentum** (0.9) at the same learning rate. How many epochs to 97%?
- **Augment the data.** Shift each training image by one pixel in a random direction.
  Does test accuracy improve? This is the idea convolution builds in (lesson 15).
- **Look inside.** Reshape each column of $W_1$ to 8×8 and plot all 64 as images. Which
  look like strokes, and which look like noise?

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/digits_numpy.py`](../sims/digits_numpy.py) prints every checkpoint on this page;
run it with `--plot` for the learning curves and the mistakes. It runs in about two
seconds. Its output:

```text
Step 0 · 1797 images of 8×8 = 64 pixels; train 1347, test 450
Step 1 · loss at initialisation = 2.377   (ln 10 = 2.303)
Step 2 · gradient check: worst relative error over 5 weights = 3.2e-08
Step 3 · SGD, lr 0.1, batch 32, 30 epochs:
         epoch  1: train loss 1.4367   test accuracy 0.5289
         epoch  5: train loss 0.4086   test accuracy 0.8867
         epoch 10: train loss 0.1688   test accuracy 0.9556
         epoch 30: train loss 0.0730   test accuracy 0.9733
         mistakes: 12 of 450; most common: 8→1 ×2, 4→8 ×2, 1→8 ×2
Step 4 · no hidden layer (scikit-learn softmax regression): test accuracy 0.9689
         lr = 3.0:   train loss after each epoch 5.57, 2.46, 2.39, 2.57, 2.42;  test accuracy 0.102
         first-layer weights N(0, 1) instead of N(0, 2/64): loss at initialisation 4.51
         ... after 30 epochs: train loss 0.0341, test accuracy 0.9689
         only 4 hidden units: test accuracy 0.7822
```

</details>
