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


# ---------- 23: GAN samples on the 8-Gaussian mixture at four points in training ----------
ns = run("23-gans")
snaps, centres = ns["snapshots"], ns["centres"]
p = Plot(900, 250)
for k, t in enumerate(sorted(snaps)):
    p.panel(k * 225, 0, 225, 250, (-3, 3), (-3, 3), margin=(14, 8, 32, 14))
    p.rect(-3, -3, 6, 6, color="card", opacity=0, stroke="rule")
    S = snaps[t][:1000]
    S = S[(np.abs(S) < 2.95).all(1)]
    p.dots(S[:, 0], S[:, 1], color="accent", r=1.3, opacity=0.45)
    for cx, cy in centres:
        p.items.append(f'<circle cx="{p.X(cx):.1f}" cy="{p.Y(cy):.1f}" r="{p.X(0.15) - p.X(0):.1f}" style="fill:none;stroke:{c("fg")};stroke-width:1.2"/>')
    p.text(p.box[0], 22, f"step {t:,}", color="fg", size=12, px=True)
p.save(OUT / "gan.svg", "Four scatter plots of 1,000 generator samples at steps 0, 500, 2,000 and 8,000, with eight rings marking the data modes on a circle; the samples move from a wide scatter, through rays aimed at the modes, onto the eight modes joined by thin bridges")


# ---------- 24: forward diffusion of the spiral, and the learned reverse process ----------
ns = run("24-diffusion-models")
r = np.random.default_rng(4)
x0 = ns["data"](1000)
abar, kept, S = ns["abar"], ns["kept"], ns["S"]
p = Plot(900, 380)
steps = [0, 10, 25, 50, 100]
for k, t in enumerate(steps):
    xt = x0 if t == 0 else np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * r.standard_normal(x0.shape)
    rev = S if t == 0 else kept[t]
    for row, (P, label) in enumerate([(xt, f"forward, t = {t}"), (rev[:1000], f"reverse, t = {t}")]):
        p.panel(k * 180, row * 190, 180, 190, (-3, 3), (-3, 3), margin=(8, 4, 26, 8))
        p.rect(-3, -3, 6, 6, color="card", opacity=0, stroke="rule")
        P = P[(np.abs(P) < 2.95).all(1)]
        p.dots(P[:, 0], P[:, 1], color="accent" if row else "data", r=1.1, opacity=0.5)
        p.text(p.box[0], row * 190 + 18, label, color="fg", size=11, px=True)
p.save(OUT / "diffusion.svg", "Two rows of five scatter plots. Top: a spiral of points dissolving into Gaussian noise at steps 0, 10, 25, 50 and 100 of the forward process. Bottom: samples of the learned reverse process at the same steps, condensing from noise at step 100 back into the spiral at step 0")


# ---------- 25: classifier-free guidance on three overlapping blobs ----------
ns = run("25-score-guidance-latent-diffusion")
Xall, yall = ns["data"](1500)
p = Plot(900, 310)
for k, w in enumerate([0.0, 1.0, 3.0]):
    p.panel(k * 300, 0, 300, 310, (-3.5, 3.5), (-3.2, 3.2), margin=(14, 10, 30, 14))
    p.rect(-3.5, -3.2, 7, 6.4, color="card", opacity=0, stroke="rule")
    p.dots(Xall[:, 0], Xall[:, 1], color="muted", r=1.0, opacity=0.25)
    S = ns["ancestral"](600, 0, w)
    S = S[(np.abs(S[:, 0]) < 3.45) & (np.abs(S[:, 1]) < 3.15)]
    p.dots(S[:, 0], S[:, 1], color="accent", r=1.5, opacity=0.6)
    for cx, cy in ns["centres"]:
        p.dots([cx], [cy], color="fg", r=3)
    p.text(p.box[0], 20, f"class 0, guidance w = {w:.0f}", color="fg", size=12, px=True)
p.save(OUT / "guidance.svg", "Three scatter plots of samples of class 0 in orange over grey data from three overlapping blobs. With guidance 0 the samples match the class-0 blob; with guidance 1 and 3 they move outward, away from the other two blobs, and tighten")
