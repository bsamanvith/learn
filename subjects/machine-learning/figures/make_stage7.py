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
