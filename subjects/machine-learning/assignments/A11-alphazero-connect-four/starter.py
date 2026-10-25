"""Assignment A11: AlphaZero for Connect Four, in NumPy.

Fill in every function marked TODO, using only NumPy. Then run

    python3 check.py            # checks this file (starter.py)
    python3 check.py mine.py    # or checks another file with the same names

Given: a policy-value network with its backward pass, Adam, and two opponents to measure
against. Yours: the rules of the game, PUCT search guided by an evaluation function, the
AlphaZero loss, self-play with symmetry augmentation, and the training loop.

The board is a (ROWS, COLS) array of int8: 0 empty, +1 and -1 the two players' pieces. Row 0 is
the top, so a piece dropped in a column lands in the largest empty row index. Player +1 moves
first. An "evaluate" function maps (board, player to move) to (prior over the COLS columns,
value in [-1, 1] for the player to move).
"""
import math
import numpy as np

ROWS, COLS = 6, 7


# ---------------------------------------------------------------- Part 1: the game

def legal_moves(board):
    """The columns whose top cell is empty, in increasing order (a list of ints)."""
    raise NotImplementedError  # TODO


def play(board, col, player):
    """Drop player's piece into col. Returns (a new board, the row where it landed); the input
    board is not modified."""
    raise NotImplementedError  # TODO


def wins(board, row, col, k=4):
    """True if the piece at (row, col) is part of k in a row horizontally, vertically or
    diagonally."""
    raise NotImplementedError  # TODO


def encode(board, player):
    """Network input from the point of view of the player to move: the player's pieces as a
    flattened 0/1 plane, followed by the opponent's. Shape (2 * ROWS * COLS,)."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- Part 2: PUCT search

class Node:
    """A search-tree node for the position `board` with `player` to move. If the position is
    terminal (the previous move won, or the board is full), `terminal` is True and `value` is the
    result for the player to move (-1 or 0). Otherwise the node is expanded lazily: P, N, W are
    arrays over the COLS columns (prior, visit count, total value for the player to move here)
    and `children` maps a column to its child Node."""
    def __init__(self, board, player, terminal=False, value=0.0):
        self.board, self.player, self.terminal, self.value = board, player, terminal, value
        self.children, self.P = {}, None
        self.N = np.zeros(board.shape[1]); self.W = np.zeros(board.shape[1])


def puct_search(board, player, evaluate, sims, rng, c_puct=1.5, noise=0.0, alpha=1.0):
    """Run `sims` simulations of PUCT from (board, player) and return the root Node.
    Each simulation: from the root, repeatedly pick argmax over legal columns of
        Q + c_puct * P * sqrt(sum N + 1) / (1 + N),   Q = W / N (0 where N = 0)
    (ties to the lowest column) until reaching a column with no child or a terminal child. A new
    child is created with play(); it is terminal if wins() (value -1 for its player to move) or
    the board is full (value 0); otherwise call evaluate() on it, store its prior P and use its
    value. Back the value up the path, flipping the sign at every ply, adding 1 to N and the
    value to W of each (node, column) edge on the path. The root is evaluated once before the
    first simulation; if noise > 0, mix Dirichlet(alpha) noise into the root prior over legal
    columns: P = (1 - noise) P + noise eta. Illegal columns must get prior 0."""
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- given: the network and Adam

class PolicyValueNet:
    """An MLP trunk (2 * ROWS * COLS -> hidden -> hidden, ReLU) with a policy head (COLS logits)
    and a value head (tanh). forward(X) returns (logits, v); backward(dlogits, dv) computes the
    gradients of the loss given dL/dlogits and dL/dv, and stores them in self.grads."""
    def __init__(self, rng, hidden=128, rows=ROWS, cols=COLS):
        n_in = 2 * rows * cols
        he = lambda a, b: rng.normal(0, math.sqrt(2 / a), (a, b))
        self.params = [he(n_in, hidden), np.zeros(hidden), he(hidden, hidden), np.zeros(hidden),
                       np.zeros((hidden, cols)), np.zeros(cols), np.zeros((hidden, 1)), np.zeros(1)]

    def forward(self, X):
        W1, b1, W2, b2, Wp, bp, Wv, bv = self.params
        self.X = X; self.h1 = np.maximum(X @ W1 + b1, 0); self.h2 = np.maximum(self.h1 @ W2 + b2, 0)
        self.v = np.tanh(self.h2 @ Wv + bv)[:, 0]
        return self.h2 @ Wp + bp, self.v

    def backward(self, dlogits, dv):
        W1, b1, W2, b2, Wp, bp, Wv, bv = self.params
        dvp = (dv * (1 - self.v ** 2))[:, None]
        dh2 = (dlogits @ Wp.T + dvp @ Wv.T) * (self.h2 > 0)
        dh1 = dh2 @ W2.T * (self.h1 > 0)
        self.grads = [self.X.T @ dh1, dh1.sum(0), self.h1.T @ dh2, dh2.sum(0),
                      self.h2.T @ dlogits, dlogits.sum(0), self.h2.T @ dvp, dvp.sum(0)]

    def evaluate(self, board, player):
        """The evaluate function for puct_search: softmax of the logits over the legal columns."""
        logits, v = self.forward(encode(board, player)[None])
        logits = logits[0] - logits[0].max()
        mask = board[0] == 0
        p = np.where(mask, np.exp(logits), 0.0)
        return p / p.sum(), float(v[0])


class Adam:
    def __init__(self, params, lr, weight_decay=1e-4):
        self.params, self.lr, self.wd, self.t = params, lr, weight_decay, 0
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g + self.wd * p
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


def rollout_evaluate(rng):
    """An evaluate function for puct_search with no learning: a uniform prior over the columns and
    the result of one random game played out from the position, for the player to move. With it,
    puct_search is plain Monte Carlo tree search with random rollouts. (given)"""
    def evaluate(board, player):
        b, p = board, player
        while True:
            moves = legal_moves(b)
            if not moves:
                return np.ones(board.shape[1]) / board.shape[1], 0.0
            c = moves[int(rng.integers(len(moves)))]
            b, r = play(b, c, p)
            if wins(b, r, c):
                return np.ones(board.shape[1]) / board.shape[1], 1.0 if p == player else -1.0
            p = -p
    return evaluate


def random_player(board, player, rng):
    """A legal column chosen uniformly at random. (given)"""
    moves = legal_moves(board)
    return moves[int(rng.integers(len(moves)))]


def tactician(board, player, rng):
    """Wins at once if it can, otherwise blocks an immediate win by the opponent, otherwise avoids
    moves that let the opponent win at once, otherwise plays at random. (given)"""
    moves = legal_moves(board)
    for p in (player, -player):
        for c in moves:
            b, r = play(board, c, p)
            if wins(b, r, c):
                return c
    safe = []
    for c in moves:
        b, _ = play(board, c, player)
        if not any(wins(*play(b, d, -player), d) for d in legal_moves(b)):
            safe.append(c)
    pool = safe or moves
    return pool[int(rng.integers(len(pool)))]


# ---------------------------------------------------------------- Part 3: learning

def az_loss(logits, v, pi, z, legal):
    """The AlphaZero loss averaged over the batch: (z - v)^2 - sum_a pi_a log p_a, where p is the
    softmax of the logits over the legal columns (legal is a 0/1 mask of the same shape as
    logits; illegal columns get probability 0 and no gradient). Returns (loss, dlogits, dv)."""
    raise NotImplementedError  # TODO


def self_play_game(evaluate, sims, rng, temp_moves=8, noise=0.25):
    """Play one game with puct_search for both sides (noise at the root). For the first temp_moves
    moves sample the move from the visit distribution N / sum N; afterwards play argmax N.
    Returns training examples (x, pi, z, legal): one per position, plus its left-right mirror
    image, where x = encode(board, player), pi = the visit distribution, z = the final result for
    the player to move at that position (+1, 0, -1) and legal = the 0/1 mask of legal columns."""
    raise NotImplementedError  # TODO


def train(net, iterations, games_per_iteration, sims, rng, steps_per_iteration=200, batch=128,
          lr=1e-3, buffer_games=500, callback=None):
    """The AlphaZero loop: each iteration plays games_per_iteration self-play games with
    net.evaluate, keeps the examples of the last buffer_games games, and takes
    steps_per_iteration Adam steps on random minibatches with az_loss. callback(iteration, net),
    if given, is called after each iteration. Returns the list of mean losses per iteration."""
    raise NotImplementedError  # TODO


def match(agent, opponent, games, rng):
    """Play `games` games between agent(board, player, rng) -> column and opponent, alternating
    who moves first (the agent first in even-numbered games). Returns (wins, draws, losses) for
    the agent."""
    raise NotImplementedError  # TODO

