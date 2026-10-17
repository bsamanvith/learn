"""1.1 — Events are sets: two dice, De Morgan, the distributive law, by brute force.

Run: python3 subjects/probability/sims/two_dice_events.py
"""

from fractions import Fraction
from itertools import product
import random

omega = set(product(range(1, 7), repeat=2))  # 36 ordered outcomes


def P(event):
    return Fraction(len(event), len(omega))


def complement(event):
    return omega - event


events = {
    "sum is 7": {w for w in omega if sum(w) == 7},
    "doubles": {w for w in omega if w[0] == w[1]},
    "first die even": {w for w in omega if w[0] % 2 == 0},
    "some 6": {w for w in omega if 6 in w},
    "sum >= 10": {w for w in omega if sum(w) >= 10},
}

print("Counted probabilities:")
for name, e in events.items():
    print(f"  P({name:15}) = {len(e):2}/36 = {float(P(e)):.4f}")

# Check the set laws on every pair/triple of named events.
names = list(events)
for a, b in product(names, repeat=2):
    A, B = events[a], events[b]
    assert complement(A | B) == complement(A) & complement(B)  # De Morgan
    assert complement(A & B) == complement(A) | complement(B)  # De Morgan
    assert P(A | B) == P(A) + P(B) - P(A & B)                  # two-event formula
for a, b, c in product(names, repeat=3):
    A, B, C = events[a], events[b], events[c]
    assert A & (B | C) == (A & B) | (A & C)                     # distributive
    assert A | (B & C) == (A | B) & (A | C)
print(f"\nDe Morgan, distributive laws and P(A∪B) formula hold for all {len(names)**3} combinations.")

A, B = events["sum is 7"], events["doubles"]
print(f"'sum is 7' and 'doubles' disjoint? {not (A & B)}")

# Counting vs simulated frequency.
rng = random.Random(0)
N = 200_000
rolls = [(rng.randint(1, 6), rng.randint(1, 6)) for _ in range(N)]
print(f"\nSimulated frequencies over {N:,} rolls:")
for name, e in events.items():
    freq = sum(r in e for r in rolls) / N
    print(f"  {name:15} counted {float(P(e)):.4f}   simulated {freq:.4f}")
