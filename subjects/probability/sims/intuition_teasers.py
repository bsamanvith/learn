"""0.3 — The seven teasers: simulate, then compare with the exact answers.

Run: python3 subjects/probability/sims/intuition_teasers.py
"""

from math import exp, factorial, prod, sqrt
import random

rng = random.Random(0)
N = 100_000


def birthday():
    days = [rng.randrange(365) for _ in range(23)]
    return len(set(days)) < 23


def monty(switch=True):
    prize, pick = rng.randrange(3), rng.randrange(3)
    opened = rng.choice([d for d in range(3) if d not in (pick, prize)])  # host knows
    final = next(d for d in range(3) if d not in (pick, opened)) if switch else pick
    return final == prize


def hats(n=10):
    perm = list(range(n))
    rng.shuffle(perm)
    return all(perm[i] != i for i in range(n))


def streak(n=100, run=6):
    best = cur = 0
    for _ in range(n):
        cur = cur + 1 if rng.random() < 0.5 else 0
        best = max(best, cur)
    return best >= run


def test(prev=0.01, sens=0.9, fpr=0.09):
    """Returns None when the test is negative (we only look at positives)."""
    sick = rng.random() < prev
    positive = rng.random() < (sens if sick else fpr)
    return sick if positive else None


def two_children():
    kids = [rng.choice("BG"), rng.choice("BG")]
    return kids == ["G", "G"] if "G" in kids else None


def fixed_points(n=52):
    perm = list(range(n))
    rng.shuffle(perm)
    return sum(perm[i] == i for i in range(n))


def streak_exact(n=100, run=6):
    # f[k] = P(no run of `run` heads in k flips): condition on the first tail.
    f = [1.0] * (n + 1)
    for k in range(run, n + 1):
        f[k] = sum(0.5 ** (j + 1) * f[k - j - 1] for j in range(run))
    return 1 - f[n]


def estimate(trial):
    hits = [r for r in (trial() for _ in range(N)) if r is not None]
    p = sum(hits) / len(hits)
    return p, 2 * sqrt(p * (1 - p) / len(hits))


exact = {
    "1 birthdays (23 people)": 1 - prod((365 - k) / 365 for k in range(23)),
    "2 Monty Hall, switch": 2 / 3,
    "3 hats, nobody matches (n=10)": sum((-1) ** k / factorial(k) for k in range(11)),
    "4 run of 6 heads in 100": streak_exact(),
    "5 sick given positive": 0.01 * 0.9 / (0.01 * 0.9 + 0.99 * 0.09),
    "6 both girls given a girl": 1 / 3,
}
trials = [birthday, monty, hats, streak, test, two_children]

print(f"{'teaser':32} {'simulated':>16} {'exact':>8}")
for (name, ex), trial in zip(exact.items(), trials):
    p, e = estimate(trial)
    print(f"{name:32} {p:8.4f} ± {e:.4f} {ex:8.4f}")

fp = [fixed_points() for _ in range(N)]
mean = sum(fp) / N
print(f"{'7 avg fixed points (52 cards)':32} {mean:8.4f} ± {2 * sqrt(1 / N):.4f} {1:8.4f}")

print("\nMonty Hall, two models of the host (switching):")
p, e = estimate(monty)
print(f"  host knows, always opens an empty door:      {p:.4f} ± {e:.4f}  (exact 2/3)")


def ignorant_monty():
    prize, pick = rng.randrange(3), rng.randrange(3)
    opened = rng.choice([d for d in range(3) if d != pick])  # host guesses
    if opened == prize:
        return None  # game spoiled; we only keep games like the one we saw
    final = next(d for d in range(3) if d not in (pick, opened))
    return final == prize


p, e = estimate(ignorant_monty)
print(f"  host guesses, happens to open an empty door: {p:.4f} ± {e:.4f}  (exact 1/2)")
print("Same door opened, different answer: the simulation encodes the *model*.")
