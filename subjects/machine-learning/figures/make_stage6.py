"""Figures for stage 6 (lessons 22-25). Run from the repo root:
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



# ---------- 22: the VAE's latent space, coloured by angle; decoded grid ----------
ns = run("22-autoencoders-and-vaes")
m, th = ns["m"], ns["th_te"]
p = Plot(900, 420)
p.panel(0, 0, 440, 420, (-3, 3), (-3, 3), margin=(40, 16, 36, 40))
p.axes(xlabel="μ₁", ylabel="μ₂", xticks=[-2, 0, 2], yticks=[-2, 0, 2])
for (x, y), a in zip(m, th):
    p.dots([x], [y], color=f"hsl({a / np.pi * 360:.0f}, 65%, 50%)", r=2.2, opacity=0.8)
for k in range(36):                                            # a legend: hue against angle
    p.rect(p.box[0] + 262 + k * 4, 14, 4, 8, color=f"hsl({k * 10}, 65%, 50%)", opacity=0.9, px=True)
p.text(p.box[0] + 257, 22, "angle 0°", color="muted", size=10, anchor="end", px=True)
p.text(p.box[0] + 410, 22, "180°", color="muted", size=10, px=True)
p.text(p.box[0], 22, "encoder means of test images", color="fg", size=12, px=True)
g = np.linspace(-2.4, 2.4, 9)
Z = np.array([[a, b] for b in g[::-1] for a in g])
img = 1 / (1 + np.exp(-ns["decode"](Z)[1]))
x0, y0, cell = 500, 40, 40
for i in range(9):
    for j in range(9):
        image(p, img[i * 9 + j].reshape(16, 16), x0 + j * cell, y0 + i * cell, cell - 3, frame=False)
p.rect(x0 - 2, y0 - 2, 9 * cell + 1, 9 * cell + 1, color="card", opacity=0, stroke="rule", px=True)
p.text(x0, 22, "decoded from a grid of codes, z from −2.4 to 2.4", color="fg", size=12, px=True)
p.save(OUT / "vae.svg", "Left: the 2-D latent means of 500 test bars, coloured by angle, forming a ring whose angle around the origin follows twice the bar angle. Right: a 9 by 9 grid of images decoded from codes between minus 2.4 and 2.4; going around the centre rotates the bar and going outward lengthens it, with blurred images near the centre")
