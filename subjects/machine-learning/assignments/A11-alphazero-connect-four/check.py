"""Autograder for assignment A11. Usage: python3 check.py [your_file.py]

The rules are checked on hand-made positions, the search on its bookkeeping and on positions
with a forced win or a forced block, the loss against finite differences, and self-play on the
structure of the data it returns. Takes under a minute."""
import importlib.util
import sys
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

results = []


def test(name, hint=""):
    def wrap(fn):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                fn()
            results.append((name, True, ""))
        except NotImplementedError:
            results.append((name, None, "not implemented yet"))
        except AssertionError as e:
            results.append((name, False, (str(e) or "wrong answer") + (f"\n        hint: {hint}" if hint else "")))
        except Exception:
            results.append((name, False, "crashed:\n" + "".join(traceback.format_exc(limit=-2))))
        return fn
    return wrap


def close(a, b, tol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: shape {a.shape}, expected {b.shape}"
    err = np.max(np.abs(a - b)) if a.size else 0
    assert err <= tol, f"{what}: off by {err:.3g} (tolerance {tol:g})"


def board(rows):
    """A board from strings, top row first: 'x' = +1, 'o' = -1, '.' = empty."""
    return np.array([[{"x": 1, "o": -1, ".": 0}[ch] for ch in r] for r in rows], dtype=np.int8)


uniform = lambda b, p: (np.ones(S.COLS) / S.COLS, 0.0)


# ---------------------------------------------------------------- Part 1

@test("legal_moves and play", "a piece lands in the lowest empty row of its column (the largest row index); copy the board")
def _():
    b = board([".x.....", ".o.....", ".x.....", ".o.....", ".x.....", "xo....."])
    assert S.legal_moves(b) == [0, 2, 3, 4, 5, 6], f"legal moves {S.legal_moves(b)}"
    b2, r = S.play(b, 0, -1)
    assert r == 4 and b2[4, 0] == -1, f"the piece should land at row 4 of column 0, got row {r}"
    assert b[4, 0] == 0, "play must not modify its input board"
    b3, r = S.play(b2, 3, 1)
    assert r == 5 and b3[5, 3] == 1, "a piece in an empty column lands in the bottom row (row 5)"


@test("wins", "count matching pieces in both directions along each of the four lines through (row, col)")
def _():
    h = board(["......."] * 5 + [".xxxx.."])
    assert S.wins(h, 5, 2), "horizontal four, checked from a middle piece"
    v = board(["......."] * 2 + ["...o..."] * 4)
    assert S.wins(v, 2, 3), "vertical four, checked from the top piece"
    d = board(["......."] * 2 + ["...x...", "..xo...", ".xoo...", "xooo..."])
    assert S.wins(d, 2, 3) and S.wins(d, 5, 0), "diagonal four (rising to the right)"
    a = board(["......."] * 2 + ["o......", "xo.....", "xxo....", "xxxo..."])
    assert S.wins(a, 4, 2), "anti-diagonal four"
    t = board(["......."] * 5 + ["xxx.xxx"])
    assert not S.wins(t, 5, 2) and not S.wins(t, 5, 4), "three plus three with a gap is not four"
    e = board(["......."] * 5 + ["xxxo..."])
    assert not S.wins(e, 5, 0), "three at the edge is not four"
    assert S.wins(board(["......."] * 5 + ["...xxx."]), 5, 4, k=3), "the k argument sets the length"


@test("encode", "first the player-to-move's pieces, then the opponent's, each flattened row by row")
def _():
    b = board(["......."] * 5 + ["xo....x"])
    x = S.encode(b, -1)
    assert x.shape == (84,), f"shape {x.shape}"
    assert x[35 + 1] == 1 and x[42 + 35] == 1 and x[42 + 41] == 1 and x.sum() == 3, "o's piece in the first plane, x's two in the second"


# ---------------------------------------------------------------- Part 2

@test("puct_search: bookkeeping and signs", "each simulation adds 1 to N along its path and adds the leaf value, negated once per ply, to W")
def _():
    rng = np.random.default_rng(0)
    b = np.zeros((6, 7), dtype=np.int8)
    root = S.puct_search(b, 1, lambda bb, p: (np.ones(7) / 7, 0.5), 7, rng)
    close(root.N, np.ones(7), 0, "with a uniform prior, 7 simulations visit each column once")
    close(root.W, -0.5 * np.ones(7), 1e-12, "W at the root: each child is worth 0.5 to the player to move there, so -0.5 to the root's player")
    root = S.puct_search(b, 1, lambda bb, p: (np.ones(7) / 7, 0.5), 50, rng)
    assert root.N.sum() == 50, f"visit counts at the root should sum to the 50 simulations, got {root.N.sum()}"


@test("puct_search: priors and illegal moves", "multiply the prior by the legal mask and renormalise; only legal columns can be selected")
def _():
    rng = np.random.default_rng(1)
    b = board(["x......", "o......", "x......", "o......", "x......", "o......"])
    pr = np.array([0.5, 0.05, 0.05, 0.05, 0.25, 0.05, 0.05])
    root = S.puct_search(b, 1, lambda bb, p: (pr, 0.0), 40, rng)
    assert root.N[0] == 0, "column 0 is full and must never be visited"
    assert np.argmax(root.N) == 4, f"with value 0 everywhere, the most-visited column should follow the prior (column 4), got {root.N}"


@test("puct_search: forced win and forced block", "terminal children: value -1 for the player to move after a winning move, 0 for a full board")
def _():
    rng = np.random.default_rng(2)
    w = board(["......."] * 4 + ["oo.....", "xxx...o"])
    root = S.puct_search(w, 1, uniform, 200, rng)
    assert np.argmax(root.N) == 3, f"x wins at once in column 3; visits {root.N}"
    blk = board(["......."] * 4 + ["oo.....", "xxx...."])
    root = S.puct_search(blk, -1, uniform, 400, rng)
    assert np.argmax(root.N) == 3, f"o must block in column 3; visits {root.N}"


@test("puct_search: root noise", "mix Dirichlet noise into the root prior over the legal columns only")
def _():
    rng = np.random.default_rng(3)
    b = board(["x......", "o......", "x......", "o......", "x......", "o......"])
    one = np.zeros(7); one[3] = 1
    roots = [S.puct_search(b, 1, lambda bb, p: (one, 0.0), 1, rng, noise=0.5) for _ in range(50)]
    P = np.array([r.P for r in roots])
    close(P.sum(1), np.ones(50), 1e-9, "the noisy prior sums to 1")
    assert np.all(P[:, 0] == 0), "the full column must keep prior 0"
    assert np.all(P[:, 3] >= 0.5 - 1e-12) and P[:, [1, 2, 4, 5, 6]].min() > 0, "P = 0.5 * prior + 0.5 * noise"


# ---------------------------------------------------------------- Part 3

@test("az_loss", "softmax over legal columns only; dlogits = (p - pi) / B, dv = -2 (z - v) / B")
def _():
    rng = np.random.default_rng(4)
    B = 5
    logits = rng.normal(0, 1, (B, 7)); v = np.tanh(rng.normal(0, 1, B)); z = rng.choice([-1.0, 0.0, 1.0], B)
    legal = np.ones((B, 7)); legal[0, 2] = 0; legal[3, [0, 6]] = 0
    pi = rng.dirichlet(np.ones(7), B) * legal; pi /= pi.sum(1, keepdims=True)
    loss, dl, dv = S.az_loss(logits, v, pi, z, legal)
    lg = np.where(legal > 0, logits, -np.inf); lg = lg - lg.max(1, keepdims=True)
    logp = np.where(legal > 0, lg - np.log(np.exp(lg).sum(1, keepdims=True)), 0)
    close(loss, np.mean((z - v) ** 2) - np.sum(pi * logp) / B, 1e-10, "loss")
    h = 1e-6
    for (i, j) in [(0, 0), (1, 3), (3, 5), (4, 6)]:
        lp, lm = logits.copy(), logits.copy(); lp[i, j] += h; lm[i, j] -= h
        num = (S.az_loss(lp, v, pi, z, legal)[0] - S.az_loss(lm, v, pi, z, legal)[0]) / (2 * h)
        close(dl[i, j], num, 1e-6, f"dlogits[{i}, {j}]")
    assert dl[0, 2] == 0 and dl[3, 0] == 0, "illegal columns get no gradient"
    vp = v.copy(); vp[2] += h; vm = v.copy(); vm[2] -= h
    close(dv[2], (S.az_loss(logits, vp, pi, z, legal)[0] - S.az_loss(logits, vm, pi, z, legal)[0]) / (2 * h), 1e-6, "dv[2]")


@test("self_play_game", "store (encode(board, player), N / sum N, result for that player, legal mask) and the mirror image of each")
def _():
    rng = np.random.default_rng(5)
    data = S.self_play_game(uniform, 20, rng)
    assert len(data) % 2 == 0 and len(data) >= 14, f"{len(data)} examples: expect two per position, at least 7 positions"
    for k in range(0, len(data), 2):
        x, pi, z, legal = data[k]; xm, pim, zm, lm = data[k + 1]
        assert x.shape == (84,) and pi.shape == (7,) and legal.shape == (7,), "shapes"
        close(pi.sum(), 1, 1e-9, "pi sums to 1")
        assert np.all(pi[legal == 0] == 0), "pi is zero on illegal columns"
        close(xm.reshape(2, 6, 7), x.reshape(2, 6, 7)[:, :, ::-1], 0, "the mirrored board")
        close(pim, pi[::-1], 0, "the mirrored policy"); assert zm == z, "the mirror has the same result"
    zs = [data[k][2] for k in range(0, len(data), 2)]
    pieces = [data[k][0].sum() for k in range(0, len(data), 2)]
    assert pieces == list(range(len(pieces))), "the k-th position has k pieces on the board"
    if zs[-1] != 0:
        assert zs[-1] == 1, "the player who made the last move won, so z = +1 at the last position"
        assert all(zs[k] == -zs[k + 1] for k in range(len(zs) - 1)), "z alternates in sign between consecutive positions"


@test("match and train", "match alternates who moves first; train runs self-play, keeps a buffer, and takes Adam steps on az_loss")
def _():
    rng = np.random.default_rng(6)
    sides = []
    def spy(b, p, r):
        if (b != 0).sum() <= 1:
            sides.append(p)
        return S.random_player(b, p, r)
    res = S.match(spy, S.random_player, 10, rng)
    assert sum(res) == 10, f"wins + draws + losses should be 10, got {res}"
    assert sorted(sides) == [-1] * 5 + [1] * 5, "the agent should play first in half of the games"
    w, d, l = S.match(S.tactician, S.random_player, 40, rng)
    assert w >= 30, f"the tactician beat random play only {w} times in 40; check the result bookkeeping"
    net = S.PolicyValueNet(rng, hidden=32)
    before = [p.copy() for p in net.params]
    losses = S.train(net, 2, 3, 8, rng, steps_per_iteration=30, batch=32)
    assert len(losses) == 2, "train returns one mean loss per iteration"
    assert any(np.abs(p - q).max() > 0 for p, q in zip(net.params, before)), "the network's parameters should change"


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
