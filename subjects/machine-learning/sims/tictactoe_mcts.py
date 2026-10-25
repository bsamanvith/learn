"""Problem book assignment — Monte Carlo tree search plays tic-tac-toe.

Solve tic-tac-toe exactly with minimax, then build MCTS (UCT with random rollouts) and watch
it go from blundering to unbeatable as the number of simulations per move grows. The
break: switch off the exploration bonus and it stops looking at the move that saves the game.

Run: .venv/bin/python subjects/machine-learning/sims/tictactoe_mcts.py
"""

import math
import random
from functools import lru_cache

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for i, j, k in LINES:
        if b[i] and b[i] == b[j] == b[k]:
            return b[i]
    return 0


def moves(b):
    return [i for i in range(9) if b[i] == 0]


def play(b, i, p):
    return b[:i] + (p,) + b[i + 1:]


def to_move(b):
    return 1 if b.count(1) == b.count(-1) else -1


# Step 1 · the whole game.
seen = set()


def explore(b):
    if b in seen:
        return
    seen.add(b)
    if winner(b) or not moves(b):
        return
    for i in moves(b):
        explore(play(b, i, to_move(b)))


EMPTY = (0,) * 9
explore(EMPTY)
terminal = [b for b in seen if winner(b) or not moves(b)]
print(f"Step 1 · reachable positions: {len(seen)} (of which {len(terminal)} are finished games)")


@lru_cache(maxsize=None)
def value(b):
    """Minimax value for X (+1 win, 0 draw, −1 loss) with perfect play from b."""
    w = winner(b)
    if w or not moves(b):
        return w
    p = to_move(b)
    vals = [value(play(b, i, p)) for i in moves(b)]
    return max(vals) if p == 1 else min(vals)


print(f"         minimax value of the empty board: {value(EMPTY)} (a draw)")
first = {i: value(play(EMPTY, i, 1)) for i in range(9)}
print(f"         X's opening moves, value with perfect play after them: {[first[i] for i in range(9)]}")
opp = {i: value(play(play(EMPTY, 0, 1), i, -1)) for i in range(1, 9)}
losing = [i for i, v in opp.items() if v == 1]
print(f"         after X takes a corner (0), O's replies that lose: {losing}; the only safe reply is the centre (4)")


def perfect_move(b, rng):
    p = to_move(b)
    best = max(value(play(b, i, p)) * p for i in moves(b))
    return rng.choice([i for i in moves(b) if value(play(b, i, p)) * p == best])


def random_move(b, rng):
    return rng.choice(moves(b))


# Step 2 · MCTS.
class Node:
    __slots__ = ("b", "parent", "move", "children", "untried", "N", "W")

    def __init__(self, b, parent=None, move=None):
        self.b, self.parent, self.move = b, parent, move
        self.children = []
        self.untried = [] if winner(b) else moves(b)
        self.N, self.W = 0, 0.0          # visits, and total reward for the player who moved INTO this node


def mcts_move(b, sims, rng, c=math.sqrt(2)):
    root = Node(b)
    for _ in range(sims):
        node = root
        while not node.untried and node.children:                      # 1 · select by UCB
            node = max(node.children, key=lambda ch: ch.W / ch.N + c * math.sqrt(math.log(node.N) / ch.N))
        if node.untried:                                               # 2 · expand one new child
            i = node.untried.pop(rng.randrange(len(node.untried)))
            child = Node(play(node.b, i, to_move(node.b)), node, i)
            node.children.append(child)
            node = child
        s = node.b                                                      # 3 · random rollout
        while not winner(s) and moves(s):
            s = play(s, rng.choice(moves(s)), to_move(s))
        w = winner(s)
        while node is not None:                                        # 4 · back up
            node.N += 1
            if node.parent is not None:
                mover = to_move(node.parent.b)
                node.W += 1.0 if w == mover else 0.5 if w == 0 else 0.0
            node = node.parent
    return max(root.children, key=lambda ch: ch.N).move


def match(x_player, o_player, games, seed):
    rng = random.Random(seed)
    tally = {1: 0, 0: 0, -1: 0}
    for _ in range(games):
        b = EMPTY
        while not winner(b) and moves(b):
            mover = x_player if to_move(b) == 1 else o_player
            b = play(b, mover(b, rng), to_move(b))
        tally[winner(b)] += 1
    return tally


def fmt(t, games):
    return f"X wins {100 * t[1] / games:5.1f}%   draws {100 * t[0] / games:5.1f}%   O wins {100 * t[-1] / games:5.1f}%"


G = 2000
print(f"\nStep 2 · baselines over {G} games")
print(f"         random vs random:   {fmt(match(random_move, random_move, G, 0), G)}")
print(f"         perfect vs random:  {fmt(match(perfect_move, random_move, G, 1), G)}")
print(f"         random vs perfect:  {fmt(match(random_move, perfect_move, G, 2), G)}")

print("\nStep 3 · MCTS as O against a perfect X (O can at best draw), 100 games per row")
for sims in (10, 30, 100, 300, 1000, 3000):
    t = match(perfect_move, lambda b, r, s=sims: mcts_move(b, s, r), 100, 3)
    print(f"         {sims:>4} simulations per move: {fmt(t, 100)}")
print("         MCTS as X against random O, 100 games:")
for sims in (10, 100):
    t = match(lambda b, r, s=sims: mcts_move(b, s, r), random_move, 100, 4)
    print(f"         {sims:>4} simulations per move: {fmt(t, 100)}")

# Step 4 · break it: no exploration bonus.
print("\nStep 4 · the same 300-simulation MCTS as O against perfect X, with exploration c = 0")
t = match(perfect_move, lambda b, r: mcts_move(b, 300, r, c=0.0), 100, 3)
print(f"         c = 0:   {fmt(t, 100)}")
rng = random.Random(5)
after_corner = play(EMPTY, 0, 1)
for c in (math.sqrt(2), 0.0):
    picks = [mcts_move(after_corner, 300, rng, c=c) for _ in range(200)]
    print(f"         c = {c:.2f}: after X's corner opening, O replies in the centre {100 * picks.count(4) / 200:.0f}% of the time")
