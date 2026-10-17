"""Shared matplotlib style for static figures.

Usage:
    from tools.figures.style import use, PALETTE
    use()
    ... plot ...
    plt.savefig("subjects/probability/assets/fig.svg")
"""

import matplotlib.pyplot as plt

# Same colours as tools/widgets/_template.html (light mode).
PALETTE = {
    "a": "#2f6feb",      # event A
    "b": "#e8833a",      # event B
    "both": "#8e5cd9",   # A and B
    "muted": "#8b949e",  # complements, axes, guides
    "fg": "#1f2328",
}


def use():
    plt.rcParams.update({
        "figure.figsize": (6, 3.5),
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["fg"],
        "axes.prop_cycle": plt.cycler(color=[PALETTE["a"], PALETTE["b"], PALETTE["both"]]),
        "font.size": 11,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
    })
