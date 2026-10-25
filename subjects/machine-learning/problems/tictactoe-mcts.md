---
title: "Monte Carlo tree search plays tic-tac-toe"
kind: assignment
time: "2 hours"
after: "40"
---

# Monte Carlo tree search plays tic-tac-toe

AlphaGo's search is Monte Carlo tree search: play thousands of imaginary games from the
current position, keep statistics on which moves won, and spend more imagination on the
promising ones. Tonight you build it from scratch for tic-tac-toe, where you can also
solve the game exactly and grade the search against perfect play. You will watch MCTS go
from losing nine games in ten against a perfect opponent to never losing, purely by
thinking longer, and then break it by taking away its curiosity.

**Deliverable:** one script, `mcts.py`, and one plot: the share of draws against a perfect
opponent as a function of simulations per move, on a log axis.

## Before you start

- Lesson 40 (model-based RL and MCTS); lesson 41 if you want the AlphaZero version.
- Represent a board as a tuple of 9 cells, $+1$ for X, $-1$ for O and $0$ for empty. X moves
  first, so the player to move is X exactly when both have the same number of marks.
  Python's `random.Random(seed)` is enough; this is all plain Python.

## Step 1 · Solve the game

Enumerate every position reachable from the empty board by legal play (stop at a win or a
full board). Then write minimax with memoisation (`functools.lru_cache`): the value of a
position for X is $+1$, $0$ or $-1$ with perfect play from both sides.

**Predict first:** how many reachable positions are there: about 500, 5,000 or 50,000?

**You should see:** 5,478 reachable positions, 958 of them finished games. The empty board
is worth 0: tic-tac-toe is a draw. Every one of X's nine openings still draws with perfect
play. But if X opens in a corner, seven of O's eight replies lose; only the centre is safe.
That single position is your test case for Step 4.

## Step 2 · Baselines

A perfect player picks uniformly among the moves of best minimax value. Play 2,000 games
for each pairing.

**You should see:**

| X | O | X wins | draws | O wins |
|---|---|---|---|---|
| random | random | 57.6% | 13.3% | 29.0% |
| perfect | random | 96.2% | 3.8% | 0.0% |
| random | perfect | 0.0% | 23.8% | 76.2% |

Going first is worth a lot between random players, and a perfect player never loses.

## Step 3 · Build MCTS

Each simulation does four things, starting from a tree that contains only the current
position:

1. **Select.** From the root, while the node is fully expanded, move to the child with the
   largest $\frac{W}{N} + c\sqrt{\frac{\ln N_{\text{parent}}}{N}}$ (UCB, lesson 32), with $c = \sqrt2$.
   $W$ is the total reward of that child *for the player who made the move into it*.
2. **Expand.** If the node has untried moves, add one at random as a new child.
3. **Roll out.** From that child, play uniformly random moves to the end.
4. **Back up.** Walk back to the root, adding 1 to every $N$, and to each $W$ add 1 for a
   win by the player who moved into that node, ½ for a draw, 0 for a loss.

Finally play the root's most-visited child (not the highest $W/N$: visits are more robust).
Let MCTS play O against the perfect X, 100 games per setting (`random.Random(3)`).

**Predict first:** how many simulations per move until MCTS never loses?

**You should see:**

| simulations per move | 10 | 30 | 100 | 300 | 1,000 | 3,000 |
|---|---|---|---|---|---|---|
| draws against perfect X | 13% | 32% | 47% | 76% | 92% | **100%** |

With 10 simulations it barely beats random play; with 3,000 it never loses. Nothing about
the game was programmed in except the rules, and the search alone climbs to perfect play.
Against a random O, MCTS as X with only 100 simulations wins 96% of games, the same as the
perfect player. **Plot** the draw rate against simulations on a log axis.

Why does it need so many simulations for a game this small? The random rollouts are noisy
judges: a position that a perfect opponent wins by force may look fine if the rollouts
play randomly. AlphaZero replaces them with a value network, and its prior over moves
replaces the uniform expansion.

## Step 4 · Break it: no curiosity

Set the exploration constant $c = 0$, so selection is pure greed on $W/N$, and rerun the
300-simulation O against perfect X. Then go back to the corner opening from Step 1 and ask
each version of MCTS for its reply 200 times.

**You should see:** with $c = 0$, draws fall from 76% to 41%. After the corner opening,
the $c = \sqrt2$ search finds the only safe reply, the centre, 74% of the time; with
$c = 0$ it finds it only 31% of the time. Without the exploration bonus, the first move
that happens to win a few rollouts hogs all the remaining simulations, and the move that
actually saves the game is never examined properly. This is the bandit lesson again, now
at every node of a tree.

## Deliverable

`mcts.py` printing the tables above, and the plot. Add one sentence: what would you
change first to make MCTS strong with 100 simulations instead of 3,000?

## Stretch

- **Smarter rollouts.** Make the rollout policy take an immediate win when one exists and
  block an immediate loss. How many simulations does the 100%-draw level need now?
- **Reuse the tree.** Keep the subtree under the move actually played for the next turn
  instead of starting afresh. How much does it save?
- **Bigger board.** Try 4×4 with four in a row, where full minimax gets slow. Does MCTS
  still find the draw?
- **A prior.** Add the PUCT rule from lesson 41, $Q + c\,P(a)\frac{\sqrt{N_{\text{parent}}}}{1 + N}$,
  with $P$ uniform, then with $P$ putting extra weight on the centre and corners.

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/tictactoe_mcts.py`](../sims/tictactoe_mcts.py) prints every checkpoint on this page
in about 20 seconds. Its output:

```text
Step 1 · reachable positions: 5478 (of which 958 are finished games)
         minimax value of the empty board: 0 (a draw)
         X's opening moves, value with perfect play after them: [0, 0, 0, 0, 0, 0, 0, 0, 0]
         after X takes a corner (0), O's replies that lose: [1, 2, 3, 5, 6, 7, 8]; the only safe reply is the centre (4)
Step 2 · baselines over 2000 games
         random vs random:   X wins  57.6%   draws  13.3%   O wins  29.0%
         perfect vs random:  X wins  96.2%   draws   3.8%   O wins   0.0%
         random vs perfect:  X wins   0.0%   draws  23.8%   O wins  76.2%
Step 3 · MCTS as O against a perfect X (O can at best draw), 100 games per row
           10 simulations per move: X wins  87.0%   draws  13.0%   O wins   0.0%
           30 simulations per move: X wins  68.0%   draws  32.0%   O wins   0.0%
          100 simulations per move: X wins  53.0%   draws  47.0%   O wins   0.0%
          300 simulations per move: X wins  24.0%   draws  76.0%   O wins   0.0%
         1000 simulations per move: X wins   8.0%   draws  92.0%   O wins   0.0%
         3000 simulations per move: X wins   0.0%   draws 100.0%   O wins   0.0%
         MCTS as X against random O, 100 games:
           10 simulations per move: X wins  86.0%   draws   5.0%   O wins   9.0%
          100 simulations per move: X wins  96.0%   draws   4.0%   O wins   0.0%
Step 4 · the same 300-simulation MCTS as O against perfect X, with exploration c = 0
         c = 0:   X wins  59.0%   draws  41.0%   O wins   0.0%
         c = 1.41: after X's corner opening, O replies in the centre 74% of the time
         c = 0.00: after X's corner opening, O replies in the centre 31% of the time
```

With 100 games per row, each percentage is uncertain by about ±5 points.

</details>
