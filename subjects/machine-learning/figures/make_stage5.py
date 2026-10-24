"""Figures for stage 5 (lessons 15-21). Run from the repo root:
    python3 subjects/machine-learning/figures/make_stage5.py
Figures that illustrate a lesson's code run that code (code/<lesson>.py), so they show
exactly what the lesson prints."""
import contextlib
import io
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.figures.svgplot import Plot, c  # noqa: E402

OUT = Path(__file__).resolve().parent
CODE = OUT.parent / "code"


def run(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(CODE / f"{name}.py"))


def image(p, A, x0, y0, size, diverging=False, frame=True):
    """Draw array A as a grid of cells in pixel coordinates, top-left at (x0, y0)."""
    h, w = A.shape
    cell = size / max(h, w)
    m = np.max(np.abs(A)) or 1
    for i in range(h):
        for j in range(w):
            v = A[i, j] / m
            if diverging:
                col, op = ("accent" if v > 0 else "data"), abs(v)
            else:
                col, op = "fg", max(v, 0)
            if op > 0.02:
                p.rect(x0 + j * cell, y0 + i * cell, cell + 0.3, cell + 0.3, color=col, opacity=round(float(op) * 0.9, 2), px=True)
    if frame:
        p.rect(x0, y0, w * cell, h * cell, color="card", opacity=0, stroke="rule", px=True)


# ---------- 15: equivariance of a learned filter; the learned filters ----------
ns = run("15-convolutional-networks")
K, bk = ns["K"], ns["bk"]
r = np.random.default_rng(5)
X = 0.3 * r.standard_normal((1, 1, 16, 16)); X[0, 0, 5:10, 1:6] += ns["plus"]
imgs = [X, np.roll(X, 9, axis=3)]                       # the same image, shifted 9 pixels right
Zs = [np.maximum(ns["conv_forward"](X, K, bk)[0][0], 0) for X in imgs]
f = int(np.argmax([Zs[0][k].max() for k in range(8)]))      # the filter that responds most to "+"
p = Plot(900, 330)
p.text(20, 22, "input (pattern at a seen and an unseen position)", color="fg", size=12, px=True)
p.text(400, 22, f"filter {f + 1}, after ReLU", color="fg", size=12, px=True)
p.text(640, 22, "the 8 learned 5×5 filters", color="fg", size=12, px=True)
for k, (X, Z) in enumerate(zip(imgs, Zs)):
    y0 = 40 + k * 145
    image(p, X[0, 0], 20, y0, 128)
    image(p, Z[f], 410, y0 + 7, 114)
    p.arrow(160, y0 + 64, 400, y0 + 64, color="muted", width=1.2, pixels=True)
    p.text(280, y0 + 56, "convolve, ReLU", color="muted", size=11, anchor="middle", px=True)
    p.text(530, y0 + 140, f"global max = {Z[f].max():.2f}", color="accent", size=11, anchor="end", px=True)
    p.text(150, y0 + 140, "seen position" if k == 0 else "shifted 9 pixels right", color="muted", size=11, anchor="end", px=True)
for k in range(8):
    image(p, K[k, 0], 640 + (k % 4) * 62, 48 + (k // 4) * 70, 52, diverging=True)
    p.text(640 + (k % 4) * 62 + 26, 48 + (k // 4) * 70 + 66, str(k + 1), color="muted", size=10, anchor="middle", px=True)
p.text(640, 210, "orange: positive weight", color="accent", size=11, px=True)
p.text(640, 228, "blue: negative weight", color="data", size=11, px=True)
print("15 filter", f + 1, [round(float(Z[f].max()), 3) for Z in Zs])
p.save(OUT / "cnn-equivariance.svg", "A '+' pattern at a position seen in training and at an unseen position; the feature map of one learned filter shows the response moving with the pattern, and its global maximum is the same; the eight learned 5 by 5 filters look like fragments of the two patterns")


# ---------- 17: Jacobian norms through time; memory task accuracy ----------
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(CODE / "17-recurrent-networks.py"))
accs = {}
for line in buf.getvalue().splitlines():
    if line.startswith("T ="):
        T = int(line.split(":")[0].split("=")[1]); parts = line.split()
        accs[T] = (float(parts[parts.index("RNN") + 1]), float(parts[parts.index("LSTM") + 1]))
p = Plot(900, 300)
p.panel(0, 0, 470, 300, (0, 50), (-20, 5), margin=(36, 14, 34, 50))
p.axes(xlabel="steps t", ylabel="||∂h_t/∂h_0||", yticks=[-20, -15, -10, -5, 0, 5],
       fmt=lambda v: "1" if v == 0 else f"1e{int(v)}")
for v in [0, 10, 20, 30, 40, 50]:
    p.seg(v, -20, v, 5, color="rule", width=1)
    p.text(p.X(v), p.box[1] + p.box[3] + 15, str(v), size=11, anchor="middle", px=True)
for scale, col in [(0.5, "data"), (1.0, "accent"), (2.0, "c")]:
    r = np.random.default_rng(1); W = scale * np.linalg.qr(r.standard_normal((32, 32)))[0]
    h, J, ys = r.standard_normal(32) * 0.1, np.eye(32), [0.0]
    for t in range(1, 51):
        h = np.tanh(W @ h + r.standard_normal(32) * 0.5)
        J = (1 - h ** 2)[:, None] * W @ J
        ys.append(np.log10(np.linalg.norm(J, 2)))
    p.line(np.arange(51), ys, color=col, width=2.2)
    p.text(49, ys[-1] + (0.8 if scale != 0.5 else 1.2), f"W = {scale:g} × orthogonal", color=col, size=11, anchor="end")
p.text(p.box[0], 16, "gradient through an untrained tanh RNN", color="fg", size=12, px=True)
p.panel(490, 0, 410, 300, (0, 3), (0, 1.12), margin=(36, 14, 34, 40))
p.axes(ylabel="test accuracy", yticks=[0, 0.25, 0.5, 0.75, 1])
for i, T in enumerate(sorted(accs)):
    for j, (name, col) in enumerate([("RNN", "data"), ("LSTM", "accent")]):
        a = accs[T][j]; x0 = i + 0.18 + j * 0.33
        p.rect(x0, 0, 0.3, a, color=col, opacity=0.75)
        p.text(x0 + 0.15, a + 0.03, f"{a:.2f}", color=col, size=10.5, anchor="middle")
    p.text(p.X(i + 0.5), p.box[1] + p.box[3] + 16, f"T = {T}", size=11, anchor="middle", px=True)
p.seg(0, 0.5, 3, 0.5, color="muted", dash="4 4"); p.text(0.995, 0.53, "chance", color="muted", size=10.5, anchor="middle")
p.text(p.box[0], 16, "remember the first of T bits:", color="fg", size=12, px=True)
p.text(p.box[0] + 218, 16, "RNN", color="data", size=12, px=True); p.text(p.box[0] + 252, 16, "vs", color="fg", size=12, px=True)
p.text(p.box[0] + 274, 16, "LSTM", color="accent", size=12, px=True)
print("17", accs)
p.save(OUT / "rnn-gradients.svg", "Left: the norm of the Jacobian from the first to the t-th hidden state of an untrained tanh RNN decays exponentially for W of scale 0.5 and 1 and grows for scale 2. Right: on the task of remembering the first of T bits, the RNN and LSTM are both perfect at T = 5, the RNN falls to chance at T = 50 while the LSTM still scores about 0.9")


# ---------- 16: training curves, plain vs residual ----------
ns = run("16-modern-cnns")
p = Plot(900, 300)
for k, depth in enumerate([12, 24]):
    p.panel(k * 450, 0, 450, 300, (0, 1500), (0, 0.9), margin=(36, 14, 34, 44))
    p.axes(xlabel="training step", ylabel="training log-loss", xticks=[0, 500, 1000, 1500], yticks=[0, 0.3, 0.6, 0.9])
    for residual, col in [(False, "data"), (True, "accent")]:
        hist, acc, _ = ns["train"](depth, residual)
        p.line(np.arange(1500), np.minimum(hist, 0.9), color=col, width=2)
        print("16", depth, residual, round(hist[299], 3), round(hist[-1], 3), acc)
    p.text(1480, 0.83, "plain", color="data", size=11, anchor="end")
    p.text(1480, 0.76, "residual", color="accent", size=11, anchor="end")
    p.text(p.box[0], 16, f"{2 * depth + 2} layers", color="fg", size=12, px=True)
p.save(OUT / "resnet-training.svg", "Training loss of plain and residual networks with 26 and 50 layers: at 26 layers the plain network learns after a long plateau; at 50 it never leaves the starting loss, while the residual networks train quickly at both depths")


# ---------- 18: person words in the first two principal components; score vs PMI ----------
ns = run("18-word-embeddings")
E, idx18 = ns["E"], ns["idx"]
names = list(ns["people"])
P8 = np.array([E[idx18[w]] for w in names]); P8 = P8 - P8.mean(0)
U, s, Vt = np.linalg.svd(P8, full_matrices=False)
print("18 explained variance", np.round(s ** 2 / np.sum(s ** 2), 3))
grp = lambda key: np.mean([P8[i] for i, w in enumerate(names) if key in ns["people"][w].split()], 0)
ax1 = grp("royal") - grp("common"); ax1 /= np.linalg.norm(ax1)
ax2 = grp("f") - grp("m"); ax2 -= (ax2 @ ax1) * ax1; ax2 /= np.linalg.norm(ax2)
ax3 = grp("child") - grp("adult"); ax3 -= (ax3 @ ax1) * ax1 + (ax3 @ ax2) * ax2; ax3 /= np.linalg.norm(ax3)
Y = P8 @ np.c_[ax1 + 0.45 * ax3, ax2 + 0.3 * ax3]            # an oblique view of the three attribute axes
lo, hi = Y.min(0) - 0.25, Y.max(0) + 0.25
p = Plot(900, 330)
p.panel(0, 0, 440, 330, (lo[0], hi[0]), (lo[1], hi[1]), margin=(36, 14, 20, 20))
pos = {w: Y[i] for i, w in enumerate(names)}
for a, b in [("man", "king"), ("woman", "queen"), ("boy", "prince"), ("girl", "princess")]:
    p.arrow(*pos[a], *pos[b], color="accent", width=1.6)
for a, b in [("man", "woman"), ("king", "queen"), ("boy", "girl"), ("prince", "princess")]:
    p.seg(*pos[a], *pos[b], color="data", width=1.2, dash="4 4")
for w in names:
    p.dots([pos[w][0]], [pos[w][1]], color="fg", r=3.4)
    p.text(pos[w][0], pos[w][1] + 0.06 * (hi[1] - lo[1]), w, color="fg", size=11.5, anchor="middle")
p.text(p.box[0], 16, "person words on the status, gender and age directions", color="fg", size=12, px=True)
p.text(p.box[0] + 4, 318, "→ add 'royal'", color="accent", size=11, px=True)
p.text(p.box[0] + 120, 318, "- - change gender", color="data", size=11, px=True)
seen = ns["seen"]; pmi = ns["pmi"] - np.log(ns["k"]); dots = ns["dots"]
p.panel(460, 0, 440, 330, (-3, 1.5), (-3.5, 1.5), margin=(36, 14, 34, 44))
p.axes(xlabel="PMI(w, c) − log k", ylabel="c·w", xticks=[-3, -2, -1, 0, 1], yticks=[-3, -2, -1, 0, 1])
p.seg(-3, -3, 1.5, 1.5, color="muted", dash="5 4")
p.dots(np.clip(pmi, -3, 1.5), np.clip(dots, -3.5, 1.5), color="accent", r=2.0, opacity=0.5)
p.text(1.4, -3.2, f"correlation {np.corrcoef(dots, pmi)[0, 1]:.2f}", color="fg", size=11, anchor="end")
p.text(p.box[0], 16, "skip-gram scores against shifted PMI", color="fg", size=12, px=True)
p.save(OUT / "embeddings.svg", "Left: the eight person words projected onto two principal components; the offsets that add 'royal' are nearly parallel, as are the offsets that change gender. Right: skip-gram scores c·w for frequent pairs plotted against PMI minus log k, scattered around the diagonal")


# ---------- 19: attention entropy against d; trained attention weights ----------
ns = run("19-attention")
r = np.random.default_rng(3)
ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
p = Plot(900, 320)
p.panel(0, 0, 430, 320, (0, 10), (0, 3), margin=(36, 14, 34, 40))
p.axes(xlabel="dimension d", ylabel="entropy of the weights (nats)", yticks=[0, 1, 2, 3])
for i, dd in enumerate(ds):
    if i % 2 == 0:
        p.text(p.X(i), p.box[1] + p.box[3] + 15, str(dd), size=11, anchor="middle", px=True)
        p.seg(i, 0, i, 3, color="rule", width=1)
for scaled, col in [(False, "data"), (True, "accent")]:
    ent = []
    for dd in ds:
        q, k = r.standard_normal((2000, dd)), r.standard_normal((2000, 16, dd))
        sc = np.einsum("nd,ntd->nt", q, k) / (np.sqrt(dd) if scaled else 1)
        a = ns["softmax"](sc); ent.append(-(a * np.log(a + 1e-30)).sum(1).mean())
    p.line(np.arange(len(ds)), ent, color=col, width=2.2); p.dots(np.arange(len(ds)), ent, color=col, r=3)
p.seg(0, np.log(16), 10, np.log(16), color="muted", dash="4 4")
p.text(9.9, np.log(16) + 0.08, "uniform over 16 keys", color="muted", size=10.5, anchor="end")
p.text(9.9, 2.15, "scaled by 1/√d", color="accent", size=11, anchor="end")
p.text(9.9, 0.3, "unscaled", color="data", size=11, anchor="end")
p.text(p.box[0], 16, "how sharp is attention at initialisation?", color="fg", size=12, px=True)
Xe, Qe, ye = ns["make"](12)
_, (_, _, _, ae, _) = ns["attend"](*ns["W"], Xe, Qe)
right = np.argmax(np.einsum("npv,nv->np", Xe[:, :, :16], Qe), 1)
x0, y0, cw, ch = 520, 42, 40, 21
for i in range(12):
    for j in range(8):
        p.rect(x0 + j * cw, y0 + i * ch, cw - 2, ch - 2, color="accent", opacity=round(float(ae[i, j]) * 0.95, 2), px=True, rx=2)
    p.items.append(f'<rect x="{x0 + right[i] * cw - 1}" y="{y0 + i * ch - 1}" width="{cw}" height="{ch}" rx="3" style="fill:none;stroke:{c("fg")};stroke-width:1.5"/>')
    p.text(x0 - 8, y0 + i * ch + 14, f"{i + 1}", color="muted", size=10, anchor="end", px=True)
for j in range(8):
    p.text(x0 + j * cw + cw / 2 - 1, y0 + 12 * ch + 14, f"{j + 1}", color="muted", size=10, anchor="middle", px=True)
p.text(x0 + 8 * cw, y0 + 12 * ch + 28, "pair", color="muted", size=10.5, anchor="end", px=True)
p.text(x0, 22, "trained head: weight on each pair (ring = matching key)", color="fg", size=12, px=True)
print("19 weight on right pair", np.round(ae[np.arange(12), right], 2))
p.save(OUT / "attention.svg", "Left: without scaling, the entropy of attention weights over 16 random keys falls toward zero as the dimension grows; with scaling by one over root d it stays near the uniform value. Right: a trained attention head's weights on 12 examples, concentrated on the pair whose key matches the query")


# ---------- 20: training loss; a first-layer head's attention on the reversal task ----------
ns = run("20-transformer")
model, L, SEP = ns["model"], ns["L"], ns["SEP"]
losses = np.array(ns["losses"])
p = Plot(900, 340)
p.panel(0, 0, 430, 340, (0, 3000), (-10, 1), margin=(36, 14, 34, 50))
p.axes(xlabel="step", ylabel="training loss (log scale)", yticks=[-9, -6, -3, 0], fmt=lambda v: f"1e{int(v)}" if v else "1")
for s in [0, 1000, 2000, 3000]:
    p.text(p.X(s), p.box[1] + p.box[3] + 15, str(s), size=11, anchor="middle", px=True)
p.rect(0, -10, 200, 11, color="muted", opacity=0.12)
p.text(230, -9.6, "warm-up", color="muted", size=10.5, anchor="start")
sm = np.convolve(np.log10(losses + 1e-12), np.ones(20) / 20, mode="valid")
p.line(np.arange(len(sm)) + 10, np.clip(sm, -10, 1), color="accent", width=2)
p.text(p.box[0], 16, "next-token loss on the reversed half", color="fg", size=12, px=True)
r = np.random.default_rng(7)
a = r.integers(0, 10, (200, L))
seq = np.concatenate([a, np.full((200, 1), SEP), a[:, ::-1]], 1)[:, :-1]
model.forward(seq)
A = model.blocks[0].attn.cache[3]                              # (B, heads, T, T)
rows = np.arange(L, 2 * L)
target = 2 * L - 1 - rows                                     # position of the digit to copy
score = [A[:, h][:, rows, target].mean() for h in range(A.shape[1])]
h = int(np.argmax(score))
M = A[:, h].mean(0)[rows]                                     # 8 rows x 16 columns
x0, y0, cw, ch = 500, 46, 23, 30
labels = [f"d{i + 1}" for i in range(L)] + [">"] + [f"o{i + 1}" for i in range(L - 1)]
for i in range(L):
    for j in range(2 * L):
        if j <= rows[i]:
            p.rect(x0 + j * cw, y0 + i * ch, cw - 2, ch - 2, color="accent", opacity=round(float(M[i, j]) * 0.95, 2), px=True, rx=2)
            p.items.append(f'<rect x="{x0 + j * cw}" y="{y0 + i * ch}" width="{cw - 2}" height="{ch - 2}" rx="2" style="fill:none;stroke:{c("rule")};stroke-width:0.6"/>')
    p.text(x0 - 6, y0 + i * ch + 19, f"out {i + 1}", color="muted", size=10, anchor="end", px=True)
for j in range(2 * L):
    p.text(x0 + j * cw + cw / 2 - 1, y0 + L * ch + 14, labels[j], color="muted", size=9.5, anchor="middle", px=True)
p.text(x0 + 4.5 * cw, y0 + L * ch + 32, "input digits", color="muted", size=10.5, anchor="middle", px=True)
p.text(x0 + 12.5 * cw, y0 + L * ch + 32, "outputs so far", color="muted", size=10.5, anchor="middle", px=True)
p.text(x0 - 40, 22, f"layer 1, head {h + 1}: where each output digit looks", color="fg", size=12, px=True)
print("20 head scores", np.round(score, 2))
p.save(OUT / "transformer.svg", "Left: the training loss of the numpy GPT falls from 2.7 to below one in a thousand within 300 steps, and to about one in a billion by step 3000. Right: one first-layer attention head, averaged over 200 test sequences; the output digit j attends to input digit 9 minus j, an anti-diagonal")
