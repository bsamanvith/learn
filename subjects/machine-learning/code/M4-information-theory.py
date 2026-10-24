import heapq
import numpy as np

def H(p):                                          # entropy in bits; 0 log 0 = 0
    p = np.asarray(p, float); p = p[p > 0]
    return -np.sum(p * np.log2(p))

def KL(p, q):                                      # KL(p || q) in bits
    p, q = np.asarray(p, float), np.asarray(q, float); m = p > 0
    return np.sum(p[m] * np.log2(p[m] / q[m]))

print("fair coin", H([0.5, 0.5]), "| biased coin 0.9", round(H([0.9, 0.1]), 3), "| fair die", round(H(np.ones(6) / 6), 3))

# Huffman coding: repeatedly merge the two least likely symbols
def huffman_lengths(p):
    heap = [(pi, [i]) for i, pi in enumerate(p)]; heapq.heapify(heap)
    length = np.zeros(len(p), int)
    while len(heap) > 1:
        p1, s1 = heapq.heappop(heap); p2, s2 = heapq.heappop(heap)
        for i in s1 + s2:
            length[i] += 1                         # every symbol in a merged group gets one more bit
        heapq.heappush(heap, (p1 + p2, s1 + s2))
    return length

p = np.array([0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])
L = huffman_lengths(p)
print("code lengths", L, "| Kraft sum", np.sum(2.0 ** -L))
print(f"entropy {H(p):.4f} bits <= average Huffman length {p @ L:.4f} < entropy + 1")

# KL is not symmetric
P, Q = [0.5, 0.5], [0.9, 0.1]
print(f"KL(P||Q) = {KL(P, Q):.4f}   KL(Q||P) = {KL(Q, P):.4f}")
print(f"cross-entropy H(P,Q) = {-np.sum(np.array(P) * np.log2(Q)):.4f} = H(P) + KL(P||Q) = {H(P) + KL(P, Q):.4f}")

# Mutual information from a joint table
joint = np.array([[0.30, 0.10],                    # rows: weather (sun, rain); columns: umbrella (no, yes)
                  [0.05, 0.55]])
px, py = joint.sum(1), joint.sum(0)
I = H(px) + H(py) - H(joint.ravel())
print(f"H(X) {H(px):.4f}  H(Y) {H(py):.4f}  H(X,Y) {H(joint.ravel()):.4f}  I(X;Y) {I:.4f}"
      f"  = KL(joint || product) {KL(joint.ravel(), np.outer(px, py).ravel()):.4f}")

# Forward vs reverse KL: fit one Gaussian to a two-bump mixture on a grid
x = np.linspace(-8, 8, 4001); dx = x[1] - x[0]
gauss = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)
target = 0.5 * gauss(-2.5, 0.8) + 0.5 * gauss(2.5, 0.8)
kl = lambda a, b: np.sum(a * np.log((a + 1e-300) / (b + 1e-300))) * dx      # in nats
grid = [(m, s) for m in np.linspace(-4, 4, 161) for s in np.linspace(0.3, 4, 186)]
fwd = min(grid, key=lambda ms: kl(target, gauss(*ms)))
rev = min(grid, key=lambda ms: kl(gauss(*ms), target))
print(f"forward KL(p||q) fit: mean {fwd[0]:+.2f}, sd {fwd[1]:.2f}   (mixture mean 0, sd {np.sqrt(np.sum(target * x**2) * dx):.2f})")
print(f"reverse KL(q||p) fit: mean {rev[0]:+.2f}, sd {rev[1]:.2f}   (one bump: mean ±2.5, sd 0.8)")

# fair coin 1.0 | biased coin 0.9 0.469 | fair die 2.585
# code lengths [1 3 3 3 4 5 5] | Kraft sum 1.0
# entropy 2.3564 bits <= average Huffman length 2.4200 < entropy + 1
# KL(P||Q) = 0.7370   KL(Q||P) = 0.5310
# cross-entropy H(P,Q) = 1.7370 = H(P) + KL(P||Q) = 1.7370
# H(X) 0.9710  H(Y) 0.9341  H(X,Y) 1.5438  I(X;Y) 0.3613  = KL(joint || product) 0.3613
# forward KL(p||q) fit: mean +0.00, sd 2.62   (mixture mean 0, sd 2.62)
# reverse KL(q||p) fit: mean -2.50, sd 0.82   (one bump: mean ±2.5, sd 0.8)
