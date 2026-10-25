import math
import numpy as np

# AlphaZero on tic-tac-toe, from scratch: a policy-value network, PUCT search guided by it,
# self-play, and training on the search's own output. Graded against exact minimax.
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

def terminal(b):
    return winner(b) != 0 or not legal(b)

memo = {}
def minimax(b, player):
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

# ---- The network: 18 inputs (my stones, opponent's stones), two ReLU layers of 64,
# a policy head (9 logits, illegal moves masked) and a value head (tanh).
def encode(b, player):
    a = np.array(b)
    return np.concatenate([(a == player), (a == -player)]).astype(float)

class Net:
    def __init__(self, rng, h=64):
        he = lambda i, o: rng.normal(0, math.sqrt(2 / i), (i, o))
        self.p = {"W1": he(18, h), "b1": np.zeros(h), "W2": he(h, h), "b2": np.zeros(h),
                  "Wp": np.zeros((h, 9)), "bp": np.zeros(9), "Wv": np.zeros((h, 1)), "bv": np.zeros(1)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}; self.t = 0

    def forward(self, X):
        p = self.p
        h1 = np.maximum(X @ p["W1"] + p["b1"], 0); h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        logits = h2 @ p["Wp"] + p["bp"]
        logits = np.where(X[:, :9] + X[:, 9:] > 0, -1e9, logits)  # occupied squares are illegal
        logits -= logits.max(1, keepdims=True)
        prob = np.exp(logits); prob /= prob.sum(1, keepdims=True)
        v = np.tanh(h2 @ p["Wv"] + p["bv"])[:, 0]
        return prob, v, (X, h1, h2)

    def train_step(self, X, pi, z, lr=1e-3, wd=1e-4):
        prob, v, (X, h1, h2) = self.forward(X)
        B = len(X); p = self.p
        loss_v = np.mean((z - v) ** 2); loss_p = -np.mean(np.sum(pi * np.log(prob + 1e-12), 1))
        dlogits = (prob - pi) / B                                  # cross-entropy through the softmax
        dv = (-2 * (z - v) / B * (1 - v ** 2))[:, None]            # squared error through the tanh
        g = {"Wp": h2.T @ dlogits, "bp": dlogits.sum(0), "Wv": h2.T @ dv, "bv": dv.sum(0)}
        dh2 = (dlogits @ p["Wp"].T + dv @ p["Wv"].T) * (h2 > 0)
        g["W2"], g["b2"] = h1.T @ dh2, dh2.sum(0)
        dh1 = dh2 @ p["W2"].T * (h1 > 0)
        g["W1"], g["b1"] = X.T @ dh1, dh1.sum(0)
        self.t += 1
        for k in p:                                                # Adam, with L2 weight decay
            gk = g[k] + wd * p[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk; self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss_v, loss_p

    def evaluate(self, b, player):
        prob, v, _ = self.forward(encode(b, player)[None])
        return prob[0], v[0]

# ---- PUCT search. Each node stores, for its children, the prior P, visit count N and total
# value W, all from the point of view of the player to move at the node.
class Node:
    __slots__ = ("b", "player", "P", "N", "W", "children", "value", "done")
    def __init__(self, b, player, net):
        self.b, self.player, self.children = b, player, {}
        w = winner(b)
        self.done = w != 0 or not legal(b)
        if self.done:
            self.value = -1.0 if w != 0 else 0.0                   # the previous mover just won, or a draw
            return
        prob, self.value = net.evaluate(b, player)                 # the network replaces the rollout
        self.P = prob; self.N = np.zeros(9); self.W = np.zeros(9)

def puct_search(b, player, net, sims, rng, c_puct=1.5, noise=False):
    root = Node(b, player, net)
    if noise:                                                      # Dirichlet noise for exploration in self-play
        moves = legal(b); eta = rng.dirichlet([1.0] * len(moves))
        root.P = root.P.copy(); root.P[moves] = 0.75 * root.P[moves] + 0.25 * eta
    for _ in range(sims):
        node, path = root, []
        while not node.done:                                       # selection: argmax Q + U
            moves = legal(node.b)
            Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            U = c_puct * node.P * math.sqrt(node.N.sum() + 1) / (1 + node.N)
            m = moves[int(np.argmax((Q + U)[moves]))]
            path.append((node, m))
            if m not in node.children:                             # expansion: evaluate the new leaf
                node.children[m] = Node(play(node.b, m, node.player), -node.player, net)
                node = node.children[m]
                break
            node = node.children[m]
        v = node.value                                             # from the leaf's player-to-move view
        for parent, m in reversed(path):                           # backup, flipping the sign each ply
            v = -v
            parent.N[m] += 1; parent.W[m] += v
    return root

# ---- Symmetries of the board, used to multiply the training data by 8.
grid = np.arange(9).reshape(3, 3)
SYMS = []
for k in range(4):
    r = np.rot90(grid, k)
    SYMS += [r.flatten(), np.fliplr(r).flatten()]

def self_play_game(net, sims, rng, temp_moves=3):
    b, player, history = (0,) * 9, 1, []
    while not terminal(b):
        root = puct_search(b, player, net, sims, rng, noise=True)
        pi = root.N / root.N.sum()                                 # the search's improved policy
        history.append((encode(b, player), pi, player))
        if len(history) <= temp_moves:
            m = int(rng.choice(9, p=pi))                           # sample early moves (temperature 1)
        else:
            m = int(np.argmax(root.N))
        b = play(b, m, player); player = -player
    w = winner(b)
    data = []
    for x, pi, who in history:
        z = 0.0 if w == 0 else (1.0 if w == who else -1.0)         # the outcome from the mover's view
        for s in SYMS:
            data.append((np.concatenate([x[:9][s], x[9:][s]]), pi[s], z))
    return data

# ---- Test sets: every reachable non-terminal position, and the 300 positions of lesson 40.
positions, every = [], []
def collect(b, player, seen):
    if (b, player) in seen or winner(b) != 0 or not legal(b):
        return
    seen.add((b, player)); every.append((b, player))
    if len(optimal_moves(b, player)) < len(legal(b)):
        positions.append((b, player))
    for m in legal(b):
        collect(play(b, m, player), -player, seen)
collect((0,) * 9, 1, set())
test = [positions[i] for i in np.random.default_rng(0).choice(len(positions), 300, replace=False)]
X_all = np.array([encode(b, p) for b, p in every]); v_all = np.array([minimax(b, p) for b, p in every])
X_mis = np.array([encode(b, p) for b, p in positions])

def grade(net, rng, sims=None):
    prob, v, _ = net.forward(X_all)
    value_ok = np.mean(np.abs(v - v_all) < 0.5)                    # nearest of -1, 0, +1 is the true value
    probm, _, _ = net.forward(X_mis)
    policy_ok = np.mean([int(np.argmax(probm[i])) in optimal_moves(b, p) for i, (b, p) in enumerate(positions)])
    if sims is None:
        return value_ok, policy_ok
    search_ok = np.mean([int(np.argmax(puct_search(b, p, net, sims, rng).N)) in optimal_moves(b, p) for b, p in test])
    return value_ok, policy_ok, search_ok

def train(temp_moves, iterations=30, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    net = Net(rng)
    buffer, history = [], []
    if verbose:
        v0, p0, s0 = grade(net, rng, sims=30)
        print(f"untrained: value correct {v0:.1%}, raw policy optimal {p0:.1%}, search (30 simulations) optimal on the 300 test positions {s0:.1%}")
    for it in range(1, iterations + 1):
        for _ in range(40):
            buffer.append(self_play_game(net, 30, rng, temp_moves))
        buffer = buffer[-400:]                                     # keep the last 10 iterations' games
        data = [d for game in buffer for d in game]
        for _ in range(300):
            idx = rng.integers(len(data), size=128)
            X = np.array([data[i][0] for i in idx]); P = np.array([data[i][1] for i in idx]); Z = np.array([data[i][2] for i in idx])
            lv, lp = net.train_step(X, P, Z)
        history.append((it,) + grade(net, rng))
        if verbose and it in (1, 5, 10, 20, 30):
            v_, p_, s_ = grade(net, rng, sims=30)
            print(f"iteration {it:2d}: value loss {lv:.3f}, policy loss {lp:.3f}; value correct {v_:.1%}, raw policy optimal {p_:.1%},"
                  f" search (30 simulations) optimal {s_:.1%}")
    seen = {tuple(x) for game in buffer for x, _, _ in game}       # positions in the buffer (all 8 symmetric forms)
    in_buf = np.array([tuple(x) in seen for x in X_all])
    _, v, _ = net.forward(X_all)
    ok = np.abs(v - v_all) < 0.5
    return net, history, (in_buf.sum(), ok[in_buf].mean(), ok[~in_buf].mean())

print(f"{len(every):,} non-terminal positions; {len(positions):,} where some move is a mistake")
print("self-play sampling the first 5 moves from the search policy:")
net, history, cover = train(temp_moves=5)
print(f"the last 400 games cover {cover[0]:,} of the 4,520 positions; value correct on them {cover[1]:.1%}, on the rest {cover[2]:.1%}")
_, history3, cover3 = train(temp_moves=3, verbose=False)
print(f"sampling only the first 3 moves: after 30 iterations value correct {history3[-1][1]:.1%}, raw policy optimal {history3[-1][2]:.1%};"
      f" the last 400 games cover {cover3[0]:,} positions")

# The trained network, with and without search, against a perfect player (random among optimal moves).
def match(sims, as_player, games, rng):
    lost = 0
    for _ in range(games):
        b, who = (0,) * 9, 1
        while not terminal(b):
            if who == as_player:
                if sims == 0:
                    prob, _ = net.evaluate(b, who); m = int(np.argmax(prob))
                else:
                    m = int(np.argmax(puct_search(b, who, net, sims, rng).N))
            else:
                opt = optimal_moves(b, who); m = opt[int(rng.integers(len(opt)))]
            b = play(b, m, who); who = -who
        lost += winner(b) == -as_player
    return lost

rng = np.random.default_rng(1)
for sims in (0, 30):
    print(f"{'raw policy network' if sims == 0 else 'PUCT, 30 simulations':20s} vs perfect play, 100 games each: lost {match(sims, 1, 100, rng)} as X, {match(sims, -1, 100, rng)} as O")

# 4,520 non-terminal positions; 3,191 where some move is a mistake
# self-play sampling the first 5 moves from the search policy:
# untrained: value correct 23.3%, raw policy optimal 41.4%, search (30 simulations) optimal on the 300 test positions 93.0%
# iteration  1: value loss 0.530, policy loss 1.625; value correct 31.0%, raw policy optimal 73.3%, search (30 simulations) optimal 95.0%
# iteration  5: value loss 0.337, policy loss 1.296; value correct 61.3%, raw policy optimal 94.0%, search (30 simulations) optimal 99.3%
# iteration 10: value loss 0.272, policy loss 1.164; value correct 71.9%, raw policy optimal 94.9%, search (30 simulations) optimal 99.3%
# iteration 20: value loss 0.170, policy loss 0.954; value correct 77.0%, raw policy optimal 94.7%, search (30 simulations) optimal 98.7%
# iteration 30: value loss 0.111, policy loss 0.943; value correct 76.6%, raw policy optimal 93.8%, search (30 simulations) optimal 98.7%
# the last 400 games cover 1,128 of the 4,520 positions; value correct on them 97.3%, on the rest 69.8%
# sampling only the first 3 moves: after 30 iterations value correct 65.5%, raw policy optimal 87.8%; the last 400 games cover 932 positions
# raw policy network   vs perfect play, 100 games each: lost 0 as X, 0 as O
# PUCT, 30 simulations vs perfect play, 100 games each: lost 0 as X, 0 as O
