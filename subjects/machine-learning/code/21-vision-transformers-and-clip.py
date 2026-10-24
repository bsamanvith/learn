import numpy as np

rng = np.random.default_rng(0)

# 1. A vision transformer's first layer: cut the image into P x P patches, flatten each, apply one linear map.
def patchify(img, P):
    """(H, W, C) image -> (number of patches, P*P*C) matrix, patches in row-major order."""
    H, W, C = img.shape
    return img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4).reshape(-1, P * P * C)

img, P, d = rng.standard_normal((32, 32, 3)), 8, 16
E = rng.standard_normal((P * P * 3, d))
tokens = patchify(img, P) @ E                                     # (16 patches, d)
# the same thing as a convolution with a P x P kernel and stride P
conv = np.array([[np.sum(img[i:i + P, j:j + P, :][..., None] * E.reshape(P, P, 3, d), (0, 1, 2))
                  for j in range(0, 32, P)] for i in range(0, 32, P)]).reshape(-1, d)
print(f"patch embedding: {tokens.shape[0]} tokens of width {d}; equals a stride-{P} convolution: {np.allclose(tokens, conv)}")

# 2. CLIP on a toy world. Images are 12x12 with one of 6 shapes in one of 4 quadrants, plus noise.
# Captions are bags of words: the shape's name, the quadrant's name, and 0-2 filler words, in any order.
SHAPES = {
    "cross":  [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "square": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    "diag":   [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "vbar":   [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
    "hbar":   [[0, 0, 0], [1, 1, 1], [0, 0, 0]], "corner": [[1, 0, 0], [1, 0, 0], [1, 1, 1]]}
QUADS = ["top-left", "top-right", "bottom-left", "bottom-right"]
FILLER = ["a", "the", "photo", "of", "picture"]
VOCAB = list(SHAPES) + QUADS + FILLER
glyphs = np.array(list(SHAPES.values()), float)

def make(n):
    s, q = rng.integers(0, 6, n), rng.integers(0, 4, n)
    X = 0.3 * rng.standard_normal((n, 12, 12))
    for i in range(n):
        r, c = 6 * (q[i] // 2) + rng.integers(0, 4), 6 * (q[i] % 2) + rng.integers(0, 4)
        X[i, r:r + 3, c:c + 3] += glyphs[s[i]]
    words = [[s[i], 6 + q[i]] + list(10 + rng.integers(0, 5, rng.integers(0, 3))) for i in range(n)]
    return X.reshape(n, -1), [list(rng.permutation(w)) for w in words], s, q

def bag(words):
    """Bag-of-words matrix: row i averages one-hot vectors of caption i's words."""
    B = np.zeros((len(words), len(VOCAB)))
    for i, w in enumerate(words):
        np.add.at(B[i], w, 1 / len(w))
    return B

k = 32
W1 = rng.standard_normal((144, 128)) * np.sqrt(2 / 144); b1 = np.zeros(128)
W2 = rng.standard_normal((128, k)) * np.sqrt(1 / 128)            # image encoder: 144 -> 128 -> 32
Et = rng.standard_normal((len(VOCAB), k)); Wt = rng.standard_normal((k, k)) * np.sqrt(1 / k)   # text encoder
params = [W1, b1, W2, Et, Wt]

def normalise(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True); return Z / n, n

def encode_images(X):
    H = np.maximum(X @ W1 + b1, 0); return normalise(H @ W2) + (H,)

def encode_text(B):
    M = B @ Et; return normalise(M @ Wt) + (M,)

def clip_loss(X, B, tau):
    """Symmetric InfoNCE. Returns the loss and the gradients of all parameters."""
    (U, nu, H), (V, nv, M) = encode_images(X), encode_text(B)
    N = len(X); S = U @ V.T / tau                                  # S[i, j]: image i against caption j
    def ce_rows(S):                                                # cross-entropy with the diagonal as targets
        Z = S - S.max(1, keepdims=True); Pr = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        return -np.mean(np.log(np.diag(Pr))), (Pr - np.eye(N)) / N
    li, gi = ce_rows(S); lt, gt = ce_rows(S.T)
    G = (gi + gt.T) / 2 / tau                                      # d loss / d (U V^T)
    gU, gV = G @ V, G.T @ U
    gU = (gU - U * (gU * U).sum(1, keepdims=True)) / nu            # back through z / |z|
    gV = (gV - V * (gV * V).sum(1, keepdims=True)) / nv
    gH = (gU @ W2.T) * (H > 0)
    return (li + lt) / 2, [X.T @ gH, gH.sum(0), H.T @ gU, B.T @ (gV @ Wt.T), M.T @ gV]

# gradient check
X, words, _, _ = make(16); B = bag(words)
_, grads = clip_loss(X, B, 0.1); worst = 0
for p, g in zip(params, grads):
    for i in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(4)]:
        old = p[i]; p[i] = old + 1e-6; up = clip_loss(X, B, 0.1)[0]
        p[i] = old - 1e-6; dn = clip_loss(X, B, 0.1)[0]; p[i] = old
        worst = max(worst, abs((up - dn) / 2e-6 - g[i]) / (abs(g[i]) + 1e-6))
print(f"gradient check: max relative error {worst:.1e}")

# train with Adam, batch 256, temperature 0.07 as in CLIP
N, tau = 256, 0.07
m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
Xte, wte, ste, qte = make(2000); Bte = bag(wte)
def zero_shot(prompts, labels, groups=None):
    """Classify test images by the nearest prompt embedding; no labels were used in training.
    groups[i] is the class of prompt i, when several prompts describe one class."""
    T, _, _ = encode_text(bag(prompts))
    pred = np.argmax(encode_images(Xte)[0] @ T.T, 1)
    return np.mean((pred if groups is None else np.array(groups)[pred]) == labels)
def retrieval(n=256):
    """For each of n images, retrieve the best of the n captions; is it a correct description?"""
    U, V = encode_images(Xte[:n])[0], encode_text(Bte[:n])[0]
    j = np.argmax(U @ V.T, 1); key = 4 * ste[:n] + qte[:n]
    return np.mean(j == np.arange(n)), np.mean(key[j] == key)
for t in range(1, 6001):
    X, words, _, _ = make(N)
    loss, grads = clip_loss(X, bag(words), tau)
    for p, g, mi, vi in zip(params, grads, m, v):
        mi *= 0.9; mi += 0.1 * g; vi *= 0.999; vi += 0.001 * g * g
        p -= 3e-3 * (mi / (1 - 0.9 ** t)) / (np.sqrt(vi / (1 - 0.999 ** t)) + 1e-8)
    if t in (1, 100, 300, 1000, 3000, 6000):
        print(f"step {t:4d}: loss {loss:.3f}  (log N = {np.log(N):.3f}; MI bound {np.log(N) - loss:.2f} nats)"
              f"  zero-shot shape {zero_shot([[i] for i in range(6)], ste):.3f}"
              f"  quadrant {zero_shot([[6 + j] for j in range(4)], qte):.3f}")

# prompts shaped like the training captions work better than a bare word ("a photo of a ..." in CLIP)
full = [[i, 6 + j] for i in range(6) for j in range(4)]
print(f"zero-shot shape with two-word prompts (any quadrant counts): {zero_shot(full, ste, [i for i in range(6) for j in range(4)]):.3f}")
# a caption names both shape and quadrant: 24 combinations
both = zero_shot([[i, 6 + j] for i in range(6) for j in range(4)], 4 * ste + qte)
print(f"zero-shot over all 24 (shape, quadrant) captions: {both:.3f}   (chance {1 / 24:.3f})")
exact, right = retrieval()
print(f"retrieval among 256 test captions: its own caption {exact:.3f}, a correct description {right:.3f}")
print(f"I(image; caption) is at most log 24 = {np.log(24):.2f} nats here; the InfoNCE bound can never exceed log N")

# patch embedding: 16 tokens of width 16; equals a stride-8 convolution: True
# gradient check: max relative error 1.6e-06
# step    1: loss 7.715  (log N = 5.545; MI bound -2.17 nats)  zero-shot shape 0.164  quadrant 0.303
# step  100: loss 4.426  (log N = 5.545; MI bound 1.12 nats)  zero-shot shape 0.232  quadrant 0.898
# step  300: loss 3.974  (log N = 5.545; MI bound 1.57 nats)  zero-shot shape 0.221  quadrant 0.934
# step 1000: loss 3.348  (log N = 5.545; MI bound 2.20 nats)  zero-shot shape 0.465  quadrant 0.965
# step 3000: loss 3.061  (log N = 5.545; MI bound 2.48 nats)  zero-shot shape 0.568  quadrant 0.979
# step 6000: loss 2.992  (log N = 5.545; MI bound 2.55 nats)  zero-shot shape 0.672  quadrant 0.988
# zero-shot shape with two-word prompts (any quadrant counts): 0.828
# zero-shot over all 24 (shape, quadrant) captions: 0.819   (chance 0.042)
# retrieval among 256 test captions: its own caption 0.066, a correct description 0.805
# I(image; caption) is at most log 24 = 3.18 nats here; the InfoNCE bound can never exceed log N
