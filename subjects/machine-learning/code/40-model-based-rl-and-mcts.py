import math
import numpy as np

# ---- Part 1: Dyna-Q on the Dyna maze (Sutton and Barto, example 8.1). A 6 x 9 grid with seven
# wall cells; reward 1 on reaching the goal, 0 otherwise; gamma = 0.95, alpha = 0.1, epsilon = 0.1.
# After every real step the agent replays n remembered transitions from its learned model.
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]                        # up, right, down, left

def make_maze(H, W, walls, start, goal):
    walls = set(walls)
    def step(s, a):
        i, j = s[0] + moves[a][0], s[1] + moves[a][1]
        if not (0 <= i < H and 0 <= j < W) or (i, j) in walls:
            i, j = s                                               # bump into the wall: stay put
        return (i, j), float((i, j) == goal)
    return dict(H=H, W=W, step=step, start=start, goal=goal, walls=walls)

dyna_maze = make_maze(6, 9, [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)], (2, 0), (0, 8))

def greedy(q, rng):
    return int(rng.choice(np.flatnonzero(q == q.max())))          # break ties at random

def dyna_q(maze, n, episodes, seed, alpha=0.1, eps=0.1, gamma=0.95):
    rng = np.random.default_rng(seed)
    Q = np.zeros((maze["H"], maze["W"], 4)); model = {}
    lengths = []
    for _ in range(episodes):
        s, t = maze["start"], 0
        while s != maze["goal"]:
            a = int(rng.integers(4)) if rng.random() < eps else greedy(Q[s], rng)
            s2, r = maze["step"](s, a); t += 1
            target = r if s2 == maze["goal"] else r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])                 # direct RL: one Q-learning update
            model[(s, a)] = (s2, r)                                # model learning: remember what happened
            keys = list(model)
            for k in rng.integers(len(keys), size=n):              # planning: n simulated updates
                ps, pa = keys[k]; ps2, pr = model[(ps, pa)]
                ptarget = pr if ps2 == maze["goal"] else pr + gamma * Q[ps2].max()
                Q[ps][pa] += alpha * (ptarget - Q[ps][pa])
            s = s2
        lengths.append(t)
    return np.array(lengths)

dyna = {n: np.mean([dyna_q(dyna_maze, n, 50, seed) for seed in range(30)], axis=0) for n in (0, 5, 50)}
for n, L in dyna.items():
    first = 1 + int(np.flatnonzero(L <= 20)[0])                   # the shortest path is 14 steps
    print(f"Dyna-Q, n = {n:2d}: mean steps in episode 2: {L[1]:6.1f}; episode 5: {L[4]:5.1f}; episodes 41-50: {L[40:].mean():5.1f};"
          f" first episode averaging 20 steps or fewer: {first}")

# ---- Part 2: the shortcut maze (example 8.3). A wall with a gap at its left end; after 3,000
# steps a second gap opens at its right end, giving a shorter path. Dyna-Q+ adds kappa * sqrt(tau)
# to the reward of a transition not tried for tau steps, and lets planning try untried actions.
def shortcut_maze(opened):
    wall = [(3, j) for j in range(1, 9 if not opened else 8)]
    return make_maze(6, 9, wall, (5, 3), (0, 8))

def dyna_changing(plus, seed, n=50, steps=6000, switch=3000, alpha=0.5, eps=0.1, gamma=0.95, kappa=1e-3):
    rng = np.random.default_rng(seed)
    Q = np.zeros((6, 9, 4)); model, last = {}, {}
    maze = shortcut_maze(False); s = maze["start"]; cum = np.zeros(steps); total = 0.0
    for t in range(steps):
        if t == switch:
            maze = shortcut_maze(True)
        a = int(rng.integers(4)) if rng.random() < eps else greedy(Q[s], rng)
        s2, r = maze["step"](s, a); total += r; cum[t] = total
        target = r if s2 == maze["goal"] else r + gamma * Q[s2].max()
        Q[s][a] += alpha * (target - Q[s][a])
        if plus and s not in {k[0] for k in model}:
            for b in range(4):                                     # untried actions: modelled as staying put, reward 0
                model.setdefault((s, b), (s, 0.0)); last.setdefault((s, b), t)
        model[(s, a)] = (s2, r); last[(s, a)] = t
        keys = list(model)
        for k in rng.integers(len(keys), size=n):
            ps, pa = keys[k]; ps2, pr = model[(ps, pa)]
            if plus:
                pr = pr + kappa * math.sqrt(t - last[(ps, pa)])    # the exploration bonus
            ptarget = pr if ps2 == maze["goal"] else pr + gamma * Q[ps2].max()
            Q[ps][pa] += alpha * (ptarget - Q[ps][pa])
        s = maze["start"] if s2 == maze["goal"] else s2
    return cum

shortcut = {name: np.mean([dyna_changing(plus, seed) for seed in range(10)], axis=0) for name, plus in [("Dyna-Q", False), ("Dyna-Q+", True)]}
for name, cum in shortcut.items():
    print(f"{name:7s}: cumulative reward after 3,000 steps {cum[2999]:5.1f}; after 6,000 {cum[-1]:5.1f};"
          f" reward rate in steps 5,001-6,000: one goal per {1000 / (cum[-1] - cum[4999]):4.1f} steps")

# ---- Part 3: Monte Carlo tree search (UCT) on tic-tac-toe, checked against exact minimax.
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

def winner(b):
    for i, j, k in LINES:
        if b[i] != 0 and b[i] == b[j] == b[k]:
            return b[i]
    return 0

def legal(b):
    return [i for i in range(9) if b[i] == 0]

def play(b, m, player):
    return b[:m] + (player,) + b[m + 1:]

memo = {}
def minimax(b, player):
    """Value of position b for the player to move: +1 win, 0 draw, -1 loss, with perfect play."""
    key = (b, player)
    if key not in memo:
        w = winner(b)
        if w != 0:
            memo[key] = 1 if w == player else -1
        elif not legal(b):
            memo[key] = 0
        else:
            memo[key] = max(-minimax(play(b, m, player), -player) for m in legal(b))
    return memo[key]

def optimal_moves(b, player):
    v = minimax(b, player)
    return [m for m in legal(b) if -minimax(play(b, m, player), -player) == v]

class Node:
    __slots__ = ("children", "untried", "N", "W")
    def __init__(self, b):
        self.children, self.untried, self.N, self.W = {}, legal(b) if winner(b) == 0 else [], 0, 0.0

def uct(b, player, sims, rng, c=math.sqrt(2)):
    """Run `sims` simulations from b; return the root (W counts reward for the player who moved INTO a node)."""
    root = Node(b)
    for _ in range(sims):
        node, board, who, path = root, b, player, [root]
        while not node.untried and node.children:                 # 1. selection by UCB1
            logN = math.log(node.N)
            m, node = max(node.children.items(), key=lambda kv: kv[1].W / kv[1].N + c * math.sqrt(logN / kv[1].N))
            board = play(board, m, who); who = -who; path.append(node)
        if node.untried:                                           # 2. expansion: add one child
            m = node.untried.pop(int(rng.integers(len(node.untried))))
            board = play(board, m, who); who = -who
            child = Node(board); node.children[m] = child; node = child; path.append(node)
        sim = board; turn = who                                    # 3. simulation: random play to the end
        while winner(sim) == 0 and legal(sim):
            moves_ = legal(sim); sim = play(sim, moves_[int(rng.integers(len(moves_)))], turn); turn = -turn
        w = winner(sim)
        mover = -who                                               # the player who moved into the leaf
        for nd in reversed(path):                                  # 4. backup: reward 1 / 0.5 / 0 for the mover
            nd.N += 1
            nd.W += 0.5 if w == 0 else float(w == mover)
            mover = -mover
    return root

def uct_move(b, player, sims, rng):
    root = uct(b, player, sims, rng)
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]  # play the most-visited move

# Test positions: every position reachable in play where at least one legal move is a mistake
# (it changes the minimax value). How often does UCT's chosen move keep the value?
positions = []
def collect(b, player, seen):
    if (b, player) in seen or winner(b) != 0 or not legal(b):
        return
    seen.add((b, player))
    if len(optimal_moves(b, player)) < len(legal(b)):
        positions.append((b, player))
    for m in legal(b):
        collect(play(b, m, player), -player, seen)
collect((0,) * 9, 1, set())
rng = np.random.default_rng(0)
test = [positions[i] for i in rng.choice(len(positions), 300, replace=False)]
budgets = [10, 30, 100, 300, 1000, 3000]
accuracy = {}
for sims in budgets:
    accuracy[sims] = np.mean([uct_move(b, p, sims, rng) in optimal_moves(b, p) for b, p in test])
print(f"tic-tac-toe: {len(memo):,} positions solved by minimax; {len(positions):,} reachable positions where some move is a mistake")
print("UCT picks a value-preserving move in 300 such positions: " + ", ".join(f"{s} simulations {accuracy[s]:.1%}" for s in budgets))

# The root's value estimate from the empty board converges to the minimax value (a draw = 0.5).
root = uct((0,) * 9, 1, 20000, np.random.default_rng(1))
best = max(root.children.items(), key=lambda kv: kv[1].N)
print(f"empty board, 20,000 simulations: most-visited move {best[0]} (visits {best[1].N:,}), its mean reward {best[1].W / best[1].N:.3f};"
      f" visits per move {[root.children[m].N for m in range(9)]}")

# Full games: UCT against a perfect player who picks uniformly among optimal moves.
def match(sims, uct_player, games, rng):
    out = {"win": 0, "draw": 0, "loss": 0}
    for _ in range(games):
        b, who = (0,) * 9, 1
        while winner(b) == 0 and legal(b):
            if who == uct_player:
                m = uct_move(b, who, sims, rng)
            else:
                opt = optimal_moves(b, who); m = opt[int(rng.integers(len(opt)))]
            b = play(b, m, who); who = -who
        w = winner(b)
        out["draw" if w == 0 else ("win" if w == uct_player else "loss")] += 1
    return out

rng = np.random.default_rng(2)
for sims in (30, 300, 3000):
    x, o = match(sims, 1, 100, rng), match(sims, -1, 100, rng)
    print(f"UCT with {sims:4d} simulations vs perfect play, 100 games each: as X lost {x['loss']:3d}, drew {x['draw']:3d};"
          f" as O lost {o['loss']:3d}, drew {o['draw']:3d}")

# Dyna-Q, n =  0: mean steps in episode 2:  671.7; episode 5: 241.6; episodes 41-50:  18.1; first episode averaging 20 steps or fewer: 30
# Dyna-Q, n =  5: mean steps in episode 2:  152.0; episode 5:  20.9; episodes 41-50:  16.8; first episode averaging 20 steps or fewer: 6
# Dyna-Q, n = 50: mean steps in episode 2:   40.7; episode 5:  17.3; episodes 41-50:  17.2; first episode averaging 20 steps or fewer: 3
# Dyna-Q : cumulative reward after 3,000 steps 107.3; after 6,000 273.3; reward rate in steps 5,001-6,000: one goal per 18.1 steps
# Dyna-Q+: cumulative reward after 3,000 steps 116.9; after 6,000 332.2; reward rate in steps 5,001-6,000: one goal per 12.5 steps
# tic-tac-toe: 5,478 positions solved by minimax; 3,191 reachable positions where some move is a mistake
# UCT picks a value-preserving move in 300 such positions: 10 simulations 80.3%, 30 simulations 91.7%, 100 simulations 95.3%, 300 simulations 99.3%, 1000 simulations 99.3%, 3000 simulations 100.0%
# empty board, 20,000 simulations: most-visited move 4 (visits 13,646), its mean reward 0.748; visits per move [824, 451, 1355, 675, 13646, 446, 1018, 564, 1021]
# UCT with   30 simulations vs perfect play, 100 games each: as X lost  22, drew  78; as O lost  74, drew  26
# UCT with  300 simulations vs perfect play, 100 games each: as X lost   0, drew 100; as O lost  23, drew  77
# UCT with 3000 simulations vs perfect play, 100 games each: as X lost   0, drew 100; as O lost   0, drew 100
