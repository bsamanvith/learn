"""Problem book assignment — build an edge detector, then teach one.

Convolution by hand on a drawn 64×64 image: Sobel filters find edges, a shift of the input
shifts the output (equivariance), and blur-then-Sobel is one 5×5 filter. Then a 3×3 kernel
starts at zero and learns Sobel-x by gradient descent from (input, output) pairs. On a busy
image it recovers the kernel exactly; on a smooth image the problem is ill-conditioned and
it learns a different kernel that only works on that image.

Run: .venv/bin/python subjects/machine-learning/sims/edge_detector.py [--plot]
"""

import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

S = 64
yy, xx = np.mgrid[0:S, 0:S]


def scene():
    img = np.zeros((S, S))
    img[10:30, 8:28] = 1.0                                    # a square
    img[np.hypot(xx - 44, yy - 20) < 11] = 0.6                # a disc
    img[(np.abs((xx - 6) - (yy - 34)) < 2) & (yy > 34) & (yy < 60)] = 0.8   # a diagonal bar
    img[40:56, 36:60] = 0.3 + 0.3 * (xx[40:56, 36:60] % 4 < 2)              # a striped patch
    return img


def conv(img, k):
    """'Valid' cross-correlation, as in every deep-learning library."""
    win = sliding_window_view(img, k.shape)
    return np.einsum("ijkl,kl->ij", win, k)


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
sobel_y = sobel_x.T
img = scene()

# Step 1 · Sobel by hand.
gx, gy = conv(img, sobel_x), conv(img, sobel_y)
mag = np.hypot(gx, gy)
print(f"Step 1 · image {S}×{S} → edge maps {gx.shape[0]}×{gx.shape[1]} ('valid' convolution)")
print(f"         largest |gx| = {np.abs(gx).max():.2f}, largest magnitude = {mag.max():.3f}; pixels with magnitude > 1: {(mag > 1).sum()}")
print(f"         a flat region gives exactly {mag[2, 2]:.1f}; sum of the Sobel weights = {sobel_x.sum():.0f}")

# Step 2 · equivariance and composition.
shifted = np.roll(np.roll(img, 3, 0), 5, 1)
err = np.abs(conv(shifted, sobel_x)[3:, 5:] - gx[:-3, :-5]).max()
print(f"\nStep 2 · shift the image by (3, 5): edge map shifts by (3, 5), max difference {err:.1e}")
box = np.ones((3, 3)) / 9
k5 = conv(np.pad(sobel_x[::-1, ::-1], 2), box)[::-1, ::-1]   # (box * sobel) as one 5×5 kernel
two = conv(conv(img, box), sobel_x)
one = conv(img, k5)
print(f"         blur then Sobel = one 5×5 kernel: max difference {np.abs(two - one).max():.1e}; 5×5 kernel row sums {k5.sum(1).round(3)}")


# Step 3 · learn Sobel-x from examples.
def learn(src, steps, lr):
    target = conv(src, sobel_x)
    P = sliding_window_view(src, (3, 3)).reshape(-1, 9)       # every 3×3 patch, as a row
    t = target.ravel()
    H = 2 * P.T @ P / len(t)                                  # Hessian of the mean squared error
    eig = np.linalg.eigvalsh(H)
    k = np.zeros(9)
    for s in range(steps):
        k -= lr * 2 * P.T @ (P @ k - t) / len(t)
    return k.reshape(3, 3), eig


eig_busy = np.linalg.eigvalsh(2 * (lambda P: P.T @ P / len(P))(sliding_window_view(img, (3, 3)).reshape(-1, 9)))
lr = 1.0 / eig_busy.max()
print(f"\nStep 3 · learn a 3×3 kernel from (image, Sobel-x output), starting at zero, step 1/λ_max = {lr:.3f}")
print(f"         Hessian eigenvalues from {eig_busy.min():.2e} to {eig_busy.max():.3f}: condition number {eig_busy.max() / eig_busy.min():,.0f}")
for steps in (100, 1000, 10000, 50000):
    k, _ = learn(img, steps, lr)
    print(f"         {steps:>6} steps: max |k − Sobel-x| = {np.abs(k - sobel_x).max():.2e}")
k_busy = k

# Step 4 · break it: a smooth training image.
smooth = np.exp(-((xx - 30) ** 2 + (yy - 34) ** 2) / (2 * 14.0**2))
P = sliding_window_view(smooth, (3, 3)).reshape(-1, 9)
eig_s = np.linalg.eigvalsh(2 * P.T @ P / len(P))
k_s, _ = learn(smooth, 50000, 1.0 / eig_s.max())
print(f"\nStep 4 · train on a smooth blob instead: condition number {eig_s.max() / eig_s.min():.2e}")
print(f"         after 50000 steps: max |k − Sobel-x| = {np.abs(k_s - sobel_x).max():.3f}")
print(f"         learned kernel:\n{np.array2string(k_s, precision=2, suppress_small=True)}")
fit_smooth = np.abs(conv(smooth, k_s) - conv(smooth, sobel_x)).max()
fit_scene = np.abs(conv(img, k_s) - gx).max()
print(f"         its output error on the blob: {fit_smooth:.4f}; on the drawn scene: {fit_scene:.3f} (Sobel's largest output is {np.abs(gx).max():.0f})")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
    for a, im, title in zip(ax, [img, gx, mag, conv(img, k_s), smooth],
                            ["drawing", "Sobel-x", "edge magnitude", "kernel learned on the blob, applied here", "the smooth blob"]):
        a.imshow(im, cmap="gray")
        a.set(title=title, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
