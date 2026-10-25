"""Figures for stage 7 (lessons 26-30). Run from the repo root:
    python3 subjects/machine-learning/figures/make_stage6.py
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



# ---------- 26: n-gram cross-entropy against the true entropy, and BPE compression ----------
ns = run("26-language-modelling-and-tokenisation")
true_bpc = sum(b for _, b in ns["test_pairs"]) / len(ns["test"])
ns_ = list(range(1, 13))
bpc = [ns["ngram_bits_per_char"](n) for n in ns_]
p = Plot(900, 330)
p.panel(0, 0, 450, 330, (0.5, 12.5), (0, 4.2), margin=(46, 20, 44, 52))
p.axes("context length n (characters)", "bits per character", xticks=[1, 2, 3, 4, 5, 6, 8, 10, 12], yticks=[0, 1, 2, 3, 4])
p.line([0.5, 12.5], [true_bpc] * 2, color="muted", width=1.4, dash="5 4")
p.text(12.3, true_bpc + 0.45, f"dashed: true model on the test set, {true_bpc:.3f}", anchor="end", size=11)
p.line(ns_, bpc, color="accent")
p.dots(ns_, bpc, color="accent", r=3.4)
p.text(p.box[0], 18, "held-out cross-entropy of n-gram models", color="fg", size=12, px=True)
merges, words = ns["merges"], ns["pretokenise"](ns["test"])
ms = list(range(0, 151, 10))
cpt = [len(ns["test"]) / sum(len(ns["bpe_encode"](w, merges[:m])) for w in words) for m in ms]
p.panel(450, 0, 450, 330, (0, 150), (0, 5), margin=(46, 20, 44, 52))
p.axes("BPE merges learned", "characters per token", xticks=[0, 50, 100, 150], yticks=[0, 1, 2, 3, 4, 5])
p.line(ms, cpt, color="data")
p.dots(ms, cpt, color="data", r=3.0)
p.text(p.box[0], 18, "compression: how long the tokens are", color="fg", size=12, px=True)
p.save(OUT / "lm-ngrams-bpe.svg", "Left: held-out bits per character of character n-gram models against n, falling from 3.99 at n = 1 to 0.26 at n = 8 and levelling off on a dashed line at the true model's 0.264. Right: average characters per BPE token against the number of merges, rising from 1 to about 4.6 at 150 merges, where most words are single tokens")


# ---------- 27: held-out loss against tokens for six widths, with the fitted Chinchilla form ----------
ns = run("27-pretraining-and-scaling-laws")
res, widths, n_params = ns["results"], ns["widths"], ns["n_params"]
E0, A, Bc, alpha, beta = ns["E0"], ns["A"], ns["Bc"], ns["alpha"], ns["beta"]
Dgrid = sorted({k[1] for k in res})
lx = lambda v: np.log10(v)
p = Plot(900, 380)
p.panel(0, 0, 900, 380, (lx(3000), lx(1e6)), (1.7, 3.6), margin=(40, 150, 46, 52))
p.axes("training tokens D (log scale)", "held-out loss, nats per character", xticks=[lx(v) for v in [1e4, 1e5, 1e6]],
       yticks=[1.8, 2.0, 2.4, 2.8, 3.2, 3.6], fmt=lambda v: f"{10 ** v:,.0f}" if v > 3.9 else f"{v:.1f}")
p.rect(lx(3000), 1.7, lx(1e5) - lx(3000), 1.9, color="muted", opacity=0.08)
p.text(lx(3300), 1.78, "not used in the fit", size=10.5)
cols = [f"hsl({h}, 60%, 48%)" for h in (200, 170, 130, 60, 30, 0)]
label_y, prev = {}, -1
for H in sorted(widths, key=lambda H: res[(n_params(H), Dgrid[-1])]):      # spread the labels apart
    prev = label_y[H] = max(res[(n_params(H), Dgrid[-1])], prev + 0.1)
for H, col in zip(widths, cols):
    N = n_params(H)
    p.dots([lx(d) for d in Dgrid], [res[(N, d)] for d in Dgrid], color=col, r=3.0)
    Ds = np.geomspace(1e5, Dgrid[-1], 40)
    p.line([lx(d) for d in Ds], E0 + A / N ** alpha + Bc / Ds ** beta, color=col, width=1.6)
    p.text(lx(Dgrid[-1]) + 0.04, label_y[H] - 0.02, f"H = {H}", size=10.5, color=col)
p.text(p.box[0], 22, f"dots: measured; lines: L = {E0:.2f} + {A:.0f}/N^{alpha:.2f} + {Bc:.0f}/D^{beta:.2f}", color="fg", size=12, px=True)
p.save(OUT / "scaling-law.svg", "Held-out loss against training tokens on a log scale for MLP language models of six widths. All curves fall from about 3.3 nats; by 900,000 tokens the loss is ordered by width, from 2.17 for H = 16 down to 1.82 for H = 512. Fitted Chinchilla-form curves, drawn from 100,000 tokens on, follow the dots closely")


# ---------- 28: reward over-optimisation along the KL-regularised path, and DPO's trajectory ----------
ns = run("28-fine-tuning-and-alignment")
betas = np.geomspace(20, 0.005, 60)
path = [ns["tilt"](ns["r_hat"], b) for b in betas]
kls = [ns["kl"](q) for q in path]
p = Plot(900, 360)
p.panel(0, 0, 900, 360, (0, 8), (-0.5, 3.3), margin=(40, 270, 44, 52))
p.axes("KL(π ‖ π_ref), nats", "expected reward", xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], yticks=[0, 1, 2, 3])
p.line([0, 8], [3, 3], color="muted", width=1.2, dash="4 4")
p.text(7.9, 3.08, "best possible true reward", size=11, anchor="end")
p.line(kls, [q @ ns["r_hat"] for q in path], color="data", dash="6 4")
p.line(kls, [q @ ns["r_true"] for q in path], color="accent")
tk, tr = zip(*ns["dpo_track"])
p.line(tk, tr, color="fg", width=1.2, opacity=0.6)
p.dots(tk, tr, color="fg", r=3.2)
bx, by, bw, bh = p.box
for i, (col, dash, lab) in enumerate([("data", "6 4", "reward model's score, RLHF path"),
                                      ("accent", None, "true reward, RLHF path"),
                                      ("fg", None, "true reward, DPO checkpoints")]):
    y = by + 30 + 24 * i
    p.items.append(f'<line x1="{bx + bw + 16}" y1="{y}" x2="{bx + bw + 40}" y2="{y}" style="stroke:{c(col)};stroke-width:2.2{";stroke-dasharray:" + dash if dash else ""}"/>')
    p.text(bx + bw + 46, y + 4, lab, size=10.5, px=True)
p.text(p.box[0], 22, "optimising a learned reward: the proxy keeps rising, the true reward peaks and falls", color="fg", size=12, px=True)
p.save(OUT / "rlhf-overoptimisation.svg", "Expected reward against KL divergence from the reference model. Along the RLHF path the reward model's score (dashed) rises steadily to 2.4, while the true reward (solid orange) rises to a peak of 1.45 near 2.5 nats and then falls to below 0.3 at 7.6 nats. DPO's checkpoints (dark dots) reach a true reward of 2.9 at about 4.9 nats before falling back slightly")


# ---------- 30: self-consistency, majority vote accuracy against the number of samples ----------
ns = run("30-in-context-learning-and-reasoning")
Ns = list(range(1, 42, 2))
p = Plot(900, 340)
p.panel(0, 0, 900, 340, (0, 42), (0, 1.02), margin=(40, 220, 44, 52))
p.axes("number of sampled answers N", "accuracy of the majority answer", xticks=[1, 5, 10, 15, 20, 25, 30, 35, 41], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
p.line([0, 42], [0.4, 0.4], color="muted", width=1.2, dash="4 4")
series = [(0.4, 10, "accent", "40% right|wrong answers spread|over 10 values"),
          (0.4, 1, "data", "40% right|one wrong answer at 60%"),
          (0.25, 10, "fg", "25% right|wrong answers spread|over 10 values")]
bx, by, bw, bh = p.box
for i, (pr, nw, col, lab) in enumerate(series):
    acc = [ns["majority_accuracy"](pr, nw, N, trials=4000) for N in Ns]
    p.line(Ns, acc, color=col)
    p.dots(Ns, acc, color=col, r=2.6)
    y = by + 24 + 60 * i
    p.items.append(f'<line x1="{bx + bw + 14}" y1="{y}" x2="{bx + bw + 36}" y2="{y}" style="stroke:{c(col)};stroke-width:2.2"/>')
    for j, part in enumerate(lab.split("|")):
        p.text(bx + bw + 42, y + 4 + 14 * j, part, size=10.5, px=True)
p.text(p.box[0], 22, "self-consistency works when the right answer is the most likely single answer", color="fg", size=12, px=True)
p.save(OUT / "self-consistency.svg", "Accuracy of the majority answer against the number of samples, from 1 to 41. A model right 40% of the time with wrong answers spread over 10 values climbs to nearly 100%; the same model with a single wrong answer at 60% falls towards 0; a model right 25% of the time with spread-out wrong answers also climbs towards 100%, more slowly")
