"""1.5 — Derangements three ways, matches vs Poisson(1), and all-faces-appear.

Run: python3 subjects/probability/sims/derangements.py
"""

from itertools import permutations
from math import comb, e, exp, factorial
import numpy as np


def by_inclusion_exclusion(n):
    return sum((-1) ** k * factorial(n) // factorial(k) for k in range(n + 1))


def by_recursion(n):
    d = [1, 0]
    for m in range(2, n + 1):
        d.append((m - 1) * (d[-1] + d[-2]))
    return d[n]


def by_brute_force(n):
    return sum(all(p[i] != i for i in range(n)) for p in permutations(range(n)))


print(f"{'n':>2} {'incl-excl':>10} {'recursion':>10} {'brute':>7} {'round(n!/e)':>11} {'D_n/n!':>8}")
for n in range(1, 13):
    ie, rec = by_inclusion_exclusion(n), by_recursion(n)
    bf = by_brute_force(n) if n <= 8 else None
    assert ie == rec and (bf is None or bf == ie)
    assert ie == round(factorial(n) / e)
    print(f"{n:>2} {ie:>10} {rec:>10} {bf if bf is not None else '-':>7} "
          f"{round(factorial(n) / e):>11} {ie / factorial(n):8.5f}")
print(f"1/e = {1 / e:.5f}")

rng = np.random.default_rng(0)
n, N = 20, 100_000
fixed = (np.argsort(rng.random((N, n)), axis=1) == np.arange(n)).sum(axis=1)
print(f"\nNumber of matches for n = {n}, {N:,} shuffles, against Poisson(1):")
for k in range(6):
    print(f"  k={k}  simulated {np.mean(fixed == k):.4f}   e^-1/k! = {exp(-1) / factorial(k):.4f}")
print(f"  mean {fixed.mean():.3f}, variance {fixed.var():.3f}  (both exactly 1 in theory)")

exact = sum((-1) ** k * comb(6, k) * ((6 - k) / 6) ** 10 for k in range(7))
rolls = rng.integers(0, 6, size=(N, 10))
sim = np.mean([len(set(r)) == 6 for r in rolls])
print(f"\nAll six faces in 10 rolls: inclusion–exclusion {exact:.4f}, simulated {sim:.4f}")
