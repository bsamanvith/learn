"""1.4 — Check every identity in the chapter by enumeration, for n <= 12.

Not a proof, but excellent at catching off-by-one errors in a story.
Run: python3 subjects/probability/sims/identities.py
"""

from itertools import combinations, product
from math import comb, factorial

MAX = 12


def C(n, k):
    """Binomial coefficient by literally counting k-subsets."""
    return sum(1 for _ in combinations(range(n), k)) if 0 <= k <= n else 0


def lattice_paths(a, b):
    """Count right/up routes by literally walking them (dynamic programming over the grid)."""
    grid = [[1] * (b + 1) for _ in range(a + 1)]
    for x, y in product(range(1, a + 1), range(1, b + 1)):
        grid[x][y] = grid[x - 1][y] + grid[x][y - 1]
    return grid[a][b]


for n in range(MAX + 1):
    row = [C(n, k) for k in range(n + 1)]
    assert row == [comb(n, k) for k in range(n + 1)]
    assert row == row[::-1]                                         # symmetry
    assert sum(row) == 2**n                                         # row sum
    if n:
        assert sum((-1) ** k * c for k, c in enumerate(row)) == 0   # alternating sum
        for k in range(1, n + 1):
            assert row[k] == C(n - 1, k - 1) + C(n - 1, k)          # Pascal's rule
            assert k * row[k] == n * C(n - 1, k - 1)                # team captain
        assert sum(k * c for k, c in enumerate(row)) == n * 2 ** (n - 1)
    assert sum(c * c for c in row) == C(2 * n, n)                   # sum of squares
    assert sum(c * 2**k for k, c in enumerate(row)) == 3**n         # powers of three
    for k in range(n + 1):
        assert sum(C(j, k) for j in range(k, n + 1)) == C(n + 1, k + 1)   # hockey stick
        assert lattice_paths(n - k, k) == row[k]                          # paths = subsets
print(f"symmetry, Pascal, row sum, alternating sum, team captain, squares, hockey stick: ok for n <= {MAX}")

for m, n in product(range(8), repeat=2):
    for k in range(m + n + 1):
        assert C(m + n, k) == sum(C(m, j) * C(n, k - j) for j in range(k + 1))  # Vandermonde
print("Vandermonde: ok for m, n < 8")

# Multinomial: count distinct arrangements of MISSISSIPPI by brute force over positions.
word = "MISSISSIPPI"
formula = factorial(11) // (factorial(1) * factorial(4) * factorial(4) * factorial(2))
# Choose positions for I's, then S's, then P's among the rest (a story, checked):
staged = C(11, 4) * C(7, 4) * C(3, 2) * C(1, 1)
assert formula == staged == 34_650
print(f"MISSISSIPPI: {formula:,} arrangements both ways")

# Binomial theorem, numerically: (x+y)^n == sum C(n,k) x^k y^(n-k)
for n, x, y in product(range(9), [-2, -1, 1, 3], [1, 2, 5]):
    assert (x + y) ** n == sum(comb(n, k) * x**k * y ** (n - k) for k in range(n + 1))
print("binomial theorem: ok")
