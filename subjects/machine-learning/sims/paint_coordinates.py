"""Problem book assignment — paint a picture with a neural network.

A small MLP maps a pixel's coordinates (x, y) to its brightness, and is trained to
reproduce a 64×64 drawing. With raw coordinates it paints a blurry blob: ReLU networks
learn low frequencies first (spectral bias). Feeding it random Fourier features
sin(2π Bv), cos(2π Bv) instead lets it paint sharp edges, and too-high frequencies make it
memorise noise (Tancik et al., 2020). Trained on every other pixel, tested on the rest.

Run: .venv/bin/python subjects/machine-learning/sims/paint_coordinates.py [--plot]
"""

import sys
import numpy as np

S = 64
g = np.linspace(-1, 1, S)
xx, yy = np.meshgrid(g, g)


def smiley(edge=0.03):
    """A smiley face with soft edges about one pixel wide, so the picture is band-limited."""
    soft = lambda z: 1 / (1 + np.exp(-z / edge))                # 0 → 1 across an edge
    img = 0.8 * soft(0.85 - np.hypot(xx, yy))                   # face
    for ex in (-0.32, 0.32):
        img *= 1 - soft(0.12 - np.hypot(xx - ex, yy + 0.25))   # eyes
    img *= 1 - soft(0.06 - np.abs(np.hypot(xx, yy + 0.05) - 0.5)) * soft(yy - 0.15)   # mouth
    return img


img = smiley()
coords = np.c_[xx.ravel(), yy.ravel()]
target = img.ravel()[:, None]
train = ((np.add.outer(np.arange(S), np.arange(S)) % 2) == 0).ravel()   # checkerboard split
test = ~train
psnr = lambda mse: 10 * np.log10(1 / mse)


def features(v, B):
    if B is None:
        return v
    z = 2 * np.pi * v @ B.T
    return np.c_[np.sin(z), np.cos(z)]


def fit(B, steps=2000, hidden=128, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed)
    F = features(coords, B)
    sizes = [F.shape[1], hidden, hidden, 1]
    W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
    b = [np.zeros(n) for n in sizes[1:]]
    params = W + b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    X, Y = F[train], target[train]
    for t in range(1, steps + 1):
        hs, zs = [X], []
        for i in range(3):
            z = hs[-1] @ W[i] + b[i]
            zs.append(z)
            hs.append(np.maximum(z, 0) if i < 2 else z)
        d = 2 * (hs[-1] - Y) / len(Y)                         # ∂ mean squared error / ∂ output
        gW, gb = [None] * 3, [None] * 3
        for i in (2, 1, 0):
            gW[i], gb[i] = hs[i].T @ d, d.sum(0)
            if i:
                d = (d @ W[i].T) * (zs[i - 1] > 0)
        for k, gk in enumerate(gW + gb):                       # Adam
            m[k] = 0.9 * m[k] + 0.1 * gk
            v[k] = 0.999 * v[k] + 0.001 * gk**2
            params[k] -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
    out = F
    for i in range(3):
        out = out @ W[i] + b[i]
        if i < 2:
            out = np.maximum(out, 0)
    mse_tr = np.mean((out[train] - target[train]) ** 2)
    mse_te = np.mean((out[test] - target[test]) ** 2)
    return out.reshape(S, S), psnr(mse_tr), psnr(mse_te)


print(f"Step 0 · a {S}×{S} drawing; train on {train.sum()} pixels (a checkerboard), test on the other {test.sum()}")
print(f"         predicting the mean brightness everywhere scores {psnr(np.var(target)):.1f} dB")

results = {}
print("\n         input                          train PSNR   test PSNR")
rng = np.random.default_rng(1)
Bbase = rng.normal(0, 1, (64, 2))
for name, B in [("raw (x, y)", None), ("Fourier, scale 0.5", 0.5 * Bbase), ("Fourier, scale 1", 1 * Bbase),
                ("Fourier, scale 2", 2 * Bbase), ("Fourier, scale 5", 5 * Bbase), ("Fourier, scale 10", 10 * Bbase)]:
    pic, tr, te = fit(B)
    results[name] = pic
    print(f"         {name:<30} {tr:7.1f} dB   {te:7.1f} dB")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(results) + 1, figsize=(3 * (len(results) + 1), 3.2))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("target")
    for a, (name, pic) in zip(ax[1:], results.items()):
        a.imshow(np.clip(pic, 0, 1), cmap="gray", vmin=0, vmax=1)
        a.set_title(name, fontsize=9)
    for a in ax:
        a.set(xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
