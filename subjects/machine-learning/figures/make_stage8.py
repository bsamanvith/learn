"""Figures for stage 8 (lessons 31-41). Run from the repo root:
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



# ---------- 31: mixing of random walks on a cycle, and Metropolis-Hastings traces ----------
ns = run("31-markov-chains")
walk, lazy, uniform, tv, lam2 = ns["walk"], ns["lazy"], ns["uniform"], ns["tv"], ns["lam2"]
p = Plot(900, 340)
steps = np.arange(0, 401, 5)
p.panel(0, 0, 450, 340, (0, 400), (-5, 0.2), margin=(40, 20, 44, 56))
p.axes("steps n", "TV distance to uniform (log10)", xticks=[0, 100, 200, 300, 400], yticks=[-5, -4, -3, -2, -1, 0])
for Q, col, lab, yl in [(walk, "data", "plain walk: periodic, stuck at 0.5", -0.15), (lazy, "accent", "lazy walk", -2.0)]:
    d = []
    for n in steps:
        d.append(np.log10(max(tv(np.linalg.matrix_power(Q, int(n))[0], uniform), 1e-12)))
    p.line(steps, d, color=col)
p.line(steps, [np.log10(lam2) * n for n in steps], color="muted", width=1.3, dash="5 4")
p.text(395, -0.75, "plain walk: periodic, stuck at 0.5", anchor="end", size=10.5)
p.text(150, -2.6, "lazy walk", size=10.5)
p.text(250, -4.4, "dashed: λ₂ⁿ", size=10.5)
p.text(p.box[0], 22, "a walk on a cycle of 20 states", color="fg", size=12, px=True)
p.panel(450, 0, 450, 340, (0, 3000), (-6.5, 6.5), margin=(40, 20, 44, 46))
p.axes("iteration", "x", xticks=[0, 1000, 2000, 3000], yticks=[-6, -3, 0, 3, 6])
for step, col in [(0.1, "data"), (4.0, "accent")]:
    xs, _ = ns["metropolis"](step, n=3000)
    p.line(np.arange(3000), xs, color=col, width=1.0, opacity=0.85)
p.text(p.box[0], 22, "Metropolis–Hastings: step 0.1 (blue), step 4 (orange)", color="fg", size=12, px=True)
p.save(OUT / "markov-mixing.svg", "Left: total variation distance to the uniform distribution on a log scale against the number of steps, for two random walks on a cycle of 20 states. The plain walk stays at 0.5 because it is periodic; the lazy walk falls in a straight line on the log scale, parallel to the dashed line lambda-two to the n. Right: the first 3,000 iterations of two Metropolis-Hastings chains targeting modes at minus 3 and plus 3; with step 0.1 the chain stays near one mode, with step 4 it jumps between both")


# ---------- 32: cumulative regret of six bandit algorithms ----------
ns = run("32-bandits-and-exploration")
res, T, c_lr = ns["results"], ns["T"], ns["c_lr"]
styles = [("greedy", "fg"), ("epsilon-greedy 0.1", "muted"), ("epsilon-greedy 0.01", "both"),
          ("UCB1", "data"), ("Thompson sampling", "accent"), ("gradient bandit", "good")]
p = Plot(900, 360)
t = np.unique(np.geomspace(1, T, 400).astype(int))
p.panel(0, 0, 430, 360, (0, T), (0, 2100), margin=(40, 16, 44, 56))
p.axes("pulls t", "cumulative regret", xticks=[0, 10000, 20000, 30000, 40000, 50000], yticks=[0, 500, 1000, 1500, 2000],
       fmt=lambda v: f"{int(v / 1000)}k" if v >= 1000 and v % 1000 == 0 and v > 2100 else f"{int(v)}")
for name, col in styles:
    p.line(t, res[name][t - 1], color=col, width=1.8)
p.text(48000, 1900, "greedy", anchor="end", size=10.5, color="fg")
p.text(48000, 1500, "ε = 0.1", anchor="end", size=10.5)
p.text(p.box[0], 22, "linear time: greedy and ε = 0.1 grow linearly", color="fg", size=12, px=True)
p.panel(450, 0, 450, 360, (1, np.log10(T)), (0, 700), margin=(40, 150, 44, 50))
p.axes("log₁₀ t", "", xticks=[1, 2, 3, 4], yticks=[0, 100, 200, 300, 400, 500, 600, 700])
tt = t[t >= 10]
for name, col in styles:
    y = np.minimum(res[name][tt - 1], 700)
    p.line(np.log10(tt), y, color=col, width=1.8)
p.line(np.log10(tt), c_lr * np.log(tt), color="muted", width=1.3, dash="5 4")
ends = {"greedy": 712, "epsilon-greedy 0.1": 668, "epsilon-greedy 0.01": res["epsilon-greedy 0.01"][-1],
        "UCB1": res["UCB1"][-1], "gradient bandit": res["gradient bandit"][-1] + 12,
        "Thompson sampling": res["Thompson sampling"][-1] - 22}
labels = {"greedy": "greedy (off scale)", "epsilon-greedy 0.1": "ε = 0.1 (off scale)", "epsilon-greedy 0.01": "ε = 0.01",
          "UCB1": "UCB1", "gradient bandit": "gradient", "Thompson sampling": "Thompson"}
for name, col in styles:
    p.text(p.X(np.log10(T)) + 8, p.Y(ends[name]) + 4, labels[name], color=col, size=10.5, px=True)
p.text(p.X(np.log10(T)) + 8, p.Y(c_lr * np.log(T)) + 16, "Lai–Robbins bound", size=10.5, px=True)
p.text(p.box[0], 22, "log scale of t: log T regret becomes a straight line", color="fg", size=12, px=True)
p.save(OUT / "bandit-regret.svg", "Cumulative regret of six bandit algorithms on a 10-armed Bernoulli bandit, averaged over 200 runs of 50,000 pulls. Left, against t on a linear axis: greedy and epsilon-greedy with epsilon 0.1 grow linearly, reaching about 2,000 and 1,700. Right, against log t: UCB1, epsilon 0.01 and the gradient bandit bend into roughly straight lines, Thompson sampling stays lowest at about 68, below the dashed Lai-Robbins asymptotic lower bound of 10.1 log t, which only binds as t grows")


# ---------- 33: optimal policies of the 4 x 3 gridworld for five living rewards ----------
ns = run("33-mdps-and-bellman-equations")
mdp, all_values, q_from_v, idx, terminal, wall, gamma = (ns[k] for k in ["mdp", "all_values", "q_from_v", "idx", "terminal", "wall", "gamma"])
start = ns["start"]
p = Plot(900, 186)
glyph = "↑→↓←"
for k, living in enumerate([-2.0, -0.4, -0.04, -0.01, 0.1]):
    Pl, Rl = mdp(living)
    _, V = all_values(Pl, Rl, gamma)
    Vb = V[np.flatnonzero(np.all(V >= V.max(0) - 1e-10, axis=1))[0]]
    act = q_from_v(Pl, Rl, Vb, gamma).argmax(0)
    free = [s for c, s in idx.items() if c not in terminal]
    lo, hi = min(Vb[s] for s in free), max(Vb[s] for s in free)
    x0, y0, cell = k * 180 + 14, 62, 38
    p.text(x0, 24, f"r = {living:+.2f}", color="fg", size=12, px=True)
    p.text(x0, 42, f"V*(start) = {Vb[start]:+.2f}", size=10.5, px=True)
    for (i, j), s in idx.items():
        x, y = x0 + j * cell, y0 + i * cell
        if (i, j) in terminal:
            good = terminal[(i, j)] > 0
            p.rect(x, y, cell, cell, color="good" if good else "bad", opacity=0.25, stroke="rule", px=True)
            p.text(x + cell / 2, y + cell / 2 + 5, "+1" if good else "−1", color="fg", size=13, anchor="middle", px=True)
            continue
        op = 0.06 + 0.5 * (Vb[s] - lo) / (hi - lo + 1e-12)
        p.rect(x, y, cell, cell, color="accent", opacity=round(float(op), 2), stroke="rule", px=True)
        p.text(x + cell / 2, y + cell / 2 + 6, glyph[act[s]], color="fg", size=17, anchor="middle", px=True)
    p.rect(x0 + wall[1] * cell, y0 + wall[0] * cell, cell, cell, color="muted", opacity=0.55, stroke="rule", px=True)
    p.text(x0 + 4, y0 + 3 * cell - 4, "S", size=9.5, px=True)
p.save(OUT / "gridworld-policies.svg", "Five copies of the 4 by 3 gridworld, each showing the optimal action in every cell as an arrow and the optimal value as shading, for living rewards of minus 2, minus 0.4, minus 0.04, minus 0.01 and plus 0.1. At minus 2 the agent heads for the nearest exit, even the minus 1 square. At minus 0.04 it takes the long safe route round the top. At minus 0.01 it avoids even a small risk of slipping into minus 1. At plus 0.1 it never leaves")


# ---------- 34: value iteration's error against the contraction bound, and Jack's optimal policy ----------
ns = run("34-dynamic-programming")
errs_j, V_pi, gam, acts, pi_star, n = ns["jack_errs"], ns["V_pi"], ns["gamma"], ns["actions"], ns["pi_star"], ns["n"]
p = Plot(900, 380)
K = len(errs_j)
p.panel(0, 0, 500, 380, (0, 200), (-8, 3), margin=(40, 20, 44, 56))
p.axes("sweeps k", "log₁₀ ‖V_k − V*‖∞", xticks=[0, 50, 100, 150, 200], yticks=[-8, -6, -4, -2, 0, 2])
ks = np.arange(1, K + 1)
p.line(ks, np.log10(gam ** ks * V_pi.max()), color="muted", width=1.3, dash="5 4")
p.line(ks, np.log10(np.maximum(errs_j, 1e-12)), color="accent")
grid_errs = np.array(ns["errs"])
keep = np.log10(np.maximum(grid_errs, 1e-16)) > -8
p.line(np.arange(1, len(grid_errs) + 1)[keep], np.log10(grid_errs[keep]), color="data")
p.text(120, 0.2, "bound γᵏ‖V*‖, γ = 0.9", size=10.5)
p.text(150, -3.3, "Jack's car rental", color="accent", size=10.5)
p.text(38, -6.5, "gridworld, γ = 0.99", color="data", size=10.5)
p.text(p.box[0], 22, "value iteration converges geometrically", color="fg", size=12, px=True)
A = np.array([[acts[pi_star[r * n + c]] for c in range(n)] for r in range(n - 1, -1, -1)], float)
x0, y0, size = 600, 62, 270
image(p, A, x0, y0, size, diverging=True)
cell = size / n
for v in [0, 5, 10, 15, 20]:
    p.text(x0 + (v + 0.5) * cell, y0 + size + 16, str(v), size=10, anchor="middle", px=True)
    p.text(x0 - 8, y0 + (20 - v + 0.5) * cell + 4, str(v), size=10, anchor="end", px=True)
p.text(x0 + size / 2, y0 + size + 34, "cars at location 2", size=10.5, anchor="middle", px=True)
p.text(x0 - 26, y0 - 8, "cars at location 1", size=10.5, px=True)
for r in range(n):
    for c_ in range(n):
        a = int(A[r, c_])
        if a != 0 and (r % 2 == 0) and (c_ % 2 == 0):
            p.text(x0 + (c_ + 0.5) * cell, y0 + (r + 0.5) * cell + 3.5, f"{a:+d}", color="fg", size=8, anchor="middle", px=True)
p.text(x0 - 26, 22, "optimal moves: + from 1 to 2, − from 2 to 1", color="fg", size=12, px=True)
p.save(OUT / "dp-convergence.svg", "Left: the error of value iteration in the sup norm, on a log scale, against the number of sweeps. For Jack's car rental the error falls in a straight line parallel to the dashed contraction bound gamma to the k times the size of V star, with gamma 0.9. For the gridworld with gamma 0.99 it falls much faster than the bound, about a factor 0.45 per sweep, because episodes end. Right: the optimal policy of Jack's car rental as a 21 by 21 grid, cars at location 1 on the vertical axis and at location 2 on the horizontal; up to 5 cars move from location 1 to 2 when location 1 is full and location 2 nearly empty, and up to 4 move back when location 1 is nearly empty")


# ---------- 35: TD against MC on the 5-state walk, and n-step TD on the 19-state walk ----------
ns = run("35-monte-carlo-and-td-learning")
curves5, nstep_err, alphas = ns["curves5"], ns["nstep_err"], ns["alphas"]
p = Plot(900, 360)
p.panel(0, 0, 440, 360, (0, 100), (0, 0.25), margin=(40, 16, 44, 52))
p.axes("episodes", "RMS error, averaged over 100 runs", xticks=[0, 25, 50, 75, 100], yticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25])
ep = np.arange(1, 101)
for (m, a), cur in curves5.items():
    col = "accent" if m == "TD" else "data"
    op = {0.05: 1.0, 0.1: 0.7, 0.15: 0.45, 0.01: 1.0, 0.02: 0.75, 0.03: 0.55, 0.04: 0.4}[a]
    p.line(ep, cur, color=col, width=1.6, opacity=op)
p.text(98, curves5[("TD", 0.05)][-1] - 0.012, "TD, α = 0.05", color="accent", size=10, anchor="end")
p.text(98, curves5[("TD", 0.15)][-1] + 0.008, "TD, α = 0.15", color="accent", size=10, anchor="end")
p.text(55, 0.15, "MC, α = 0.01 to 0.04", color="data", size=10)
p.text(p.box[0], 22, "5-state walk: TD (orange) learns faster than MC (blue)", color="fg", size=12, px=True)
p.panel(460, 0, 440, 360, (0, 1), (0.2, 0.56), margin=(40, 70, 44, 50))
p.axes("step size α", "RMS error over the first 10 episodes", xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0.2, 0.3, 0.4, 0.5])
shades = {1: 1.0, 2: 0.85, 4: 0.7, 8: 0.55, 16: 0.45, 32: 0.35, 512: 0.25}
for nn, err in nstep_err.items():
    keep = err < 0.56
    p.line(alphas[keep], err[keep], color="accent" if nn == 4 else "fg", width=1.8 if nn == 4 else 1.3, opacity=1.0 if nn == 4 else shades[nn])
    last = np.flatnonzero(keep)[-1]
    p.text(p.X(alphas[last]) + 5, p.Y(err[last]) + (-6 if nn == 512 else 4), f"n={nn}", size=10, color="accent" if nn == 4 else "muted", px=True)
p.text(p.box[0], 22, "19-state walk: intermediate n is best", color="fg", size=12, px=True)
p.save(OUT / "td-vs-mc.svg", "Left: root mean square error of the value estimates on the 5-state random walk against the number of episodes, averaged over 100 runs, for TD(0) with step sizes 0.05, 0.1 and 0.15 and constant-alpha Monte Carlo with step sizes 0.01 to 0.04. TD's errors fall faster and lower. Right: on the 19-state random walk, the error averaged over the first 10 episodes against the step size, for n-step TD with n from 1 to 512; n equal to 4 reaches the lowest error, about 0.26, and both one-step TD and Monte Carlo are worse")


# ---------- 36: cliff walking online reward, and maximisation bias ----------
ns = run("36-sarsa-and-q-learning")
cliff, frac_q, frac_double = ns["cliff"], ns["frac_q"], ns["frac_double"]
p = Plot(900, 360)
p.panel(0, 0, 440, 360, (0, 500), (-100, 0), margin=(40, 16, 44, 52))
p.axes("episodes", "reward per episode (50 runs, smoothed)", xticks=[0, 100, 200, 300, 400, 500], yticks=[-100, -75, -50, -25, 0])
k = 10
for name, col in [("SARSA", "data"), ("Q-learning", "accent"), ("Expected SARSA", "fg")]:
    sm = np.convolve(cliff[name], np.ones(k) / k, mode="valid")
    p.line(np.arange(k, 501), np.maximum(sm, -100), color=col, width=1.6)   # the first episodes, far below -100, are cut off
p.text(495, cliff["Expected SARSA"][400:].mean() + 7, "Expected SARSA", color="fg", size=10.5, anchor="end")
p.text(495, cliff["SARSA"][400:].mean() - 9, "SARSA", color="data", size=10.5, anchor="end")
p.text(495, cliff["Q-learning"][400:].mean() - 9, "Q-learning", color="accent", size=10.5, anchor="end")
p.text(p.box[0], 22, "cliff walking: what ε-greedy agents earn while learning", color="fg", size=12, px=True)
p.panel(460, 0, 440, 360, (0, 300), (0, 1), margin=(40, 16, 44, 50))
p.axes("episodes", "fraction choosing 'left' in A (2,000 runs)", xticks=[0, 100, 200, 300], yticks=[0, 0.25, 0.5, 0.75, 1.0])
ep = np.arange(1, 301)
p.line([0, 300], [0.05, 0.05], color="muted", width=1.2, dash="4 4")
p.line(ep, frac_q, color="accent", width=1.6)
p.line(ep, frac_double, color="data", width=1.6)
p.text(150, 0.62, "Q-learning", color="accent", size=10.5)
p.text(70, 0.3, "double Q-learning", color="data", size=10.5)
p.text(298, 0.012, "optimal under ε = 0.1", size=10, anchor="end")
p.text(p.box[0], 22, "maximisation bias", color="fg", size=12, px=True)
p.save(OUT / "sarsa-q-learning.svg", "Left: reward per episode on the cliff-walking task, averaged over 50 runs and smoothed, for SARSA, Q-learning and Expected SARSA with epsilon 0.1 and step size 0.5. After about 100 episodes Q-learning earns about minus 49 per episode, because it walks along the cliff edge and its exploration sometimes pushes it off, while SARSA earns about minus 26 and Expected SARSA about minus 21. Right: the fraction of runs choosing the action left in state A, against episodes, for Q-learning and double Q-learning; Q-learning chooses the worse action left over 80 percent of the time for the first 50 episodes, double Q-learning drops quickly towards the optimal 5 percent")
