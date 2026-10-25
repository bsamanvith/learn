"""Autograder for assignment A9. Usage: python3 check.py [your_file.py]

Planning is checked against exact linear algebra and the Bellman equations, the update rules
against hand-computed values, and the learning loops by whether they learn what lessons 35
and 36 say they learn."""
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


def random_mdp(seed, S_=6, A=3):
    r = np.random.default_rng(seed)
    P = r.dirichlet(np.ones(S_) * 0.5, (A, S_))
    R = r.normal(0, 1, (A, S_))
    return P, R


class Script:
    """A fake environment that replays a fixed list of (next state, reward, done) transitions."""
    def __init__(self, n_states, episodes, n=None):
        self.n_states, self.n_actions, self.episodes = n_states, 1, episodes
        self.n = n_states if n is None else n
        self.e = -1

    def reset(self):
        self.e += 1; self.t = 0
        return self.episodes[self.e][0]

    def step(self, s, a, rng):
        tr = self.episodes[self.e][1][self.t]; self.t += 1
        return tr


# ---------------------------------------------------------------- Part 1

@test("stationary_distribution", "solve pi (P - I) = 0 together with sum(pi) = 1")
def _():
    r = np.random.default_rng(1)
    P = r.dirichlet(np.ones(7), 7)
    pi = S.stationary_distribution(P)
    close(pi.sum(), 1.0, 1e-9, "sum of pi")
    close(pi @ P, pi, 1e-9, "pi P - pi")
    A = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], float)   # a graph
    close(S.stationary_distribution(A / A.sum(1, keepdims=True)), A.sum(1) / A.sum(), 1e-9, "random walk on a graph (deg / 2|E|)")


@test("ucb1_choose", "unpulled arms first (lowest index), then mean + sqrt(2 ln t / n)")
def _():
    assert S.ucb1_choose(3, np.array([1., 0., 0.]), np.array([1., 0., 0.])) == 1, "an arm never pulled must be pulled first"
    counts, sums = np.array([10., 40., 5.]), np.array([6., 30., 2.])
    t = 55
    idx = sums / counts + np.sqrt(2 * np.log(t) / counts)
    assert S.ucb1_choose(t, counts, sums) == int(np.argmax(idx)), f"expected arm {int(np.argmax(idx))}, indices {np.round(idx, 3)}"
    assert S.ucb1_choose(t, np.array([10., 400., 5.]), np.array([6., 300., 2.])) == 2, "with 400 pulls of arm 1, arm 2's bonus should win"


@test("thompson_choose", "sample theta_a ~ Beta(1 + successes, 1 + failures) for every arm and take the argmax")
def _():
    rng = np.random.default_rng(0)
    picks = [S.thompson_choose(np.array([100., 100.]), np.array([50., 60.]), rng) for _ in range(4000)]
    f = np.mean(np.array(picks) == 1)
    assert 0.88 < f < 0.95, f"arm 1 (60/100) chosen {f:.3f} of the time against arm 0 (50/100); expected about 0.92"
    picks = [S.thompson_choose(np.array([0., 0., 0.]), np.array([0., 0., 0.]), rng) for _ in range(3000)]
    fr = np.bincount(picks, minlength=3) / 3000
    assert np.all(np.abs(fr - 1 / 3) < 0.04), f"with no data each arm should be chosen a third of the time, got {np.round(fr, 3)}"


# ---------------------------------------------------------------- Part 2

@test("policy_evaluation and q_from_v", "V = solve(I - gamma P_pi, r_pi); Q[s, a] = R[a, s] + gamma P[a, s] @ V")
def _():
    P, R = random_mdp(2)
    pi = np.random.default_rng(3).dirichlet(np.ones(3), 6)
    V = S.policy_evaluation(P, R, pi, 0.9)
    Q = S.q_from_v(P, R, V, 0.9)
    assert np.shape(Q) == (6, 3), f"Q should be (S, A) = (6, 3), got {np.shape(Q)}"
    close(np.sum(pi * Q, 1), V, 1e-9, "V(s) - sum_a pi(a|s) Q(s, a) (the Bellman expectation equation)")
    close(Q[2, 1], R[1, 2] + 0.9 * P[1, 2] @ V, 1e-12, "Q[2, 1]")


@test("greedy_policy", "one-hot rows; np.argmax already breaks ties towards the lowest index")
def _():
    pi = S.greedy_policy(np.array([[1., 3., 3.], [0., -1., -2.], [5., 5., 5.]]))
    close(pi, [[0, 1, 0], [1, 0, 0], [1, 0, 0]], 0, "greedy policy")


@test("value_iteration", "V <- max over a of R + gamma P V; stop when the largest change is below tol")
def _():
    P, R = random_mdp(4)
    V, pi, sweeps = S.value_iteration(P, R, 0.9, tol=1e-12)
    close(V, (R + 0.9 * P @ V).max(0), 1e-9, "Bellman optimality residual")
    close(S.policy_evaluation(P, R, pi, 0.9), V, 1e-8, "value of the returned policy minus V")
    assert 200 < sweeps < 400, f"{sweeps} sweeps; with gamma = 0.9 and tol 1e-12, expect a few hundred"
    g = S.GridWorld(S.GRID_4x3, slip=0.2, living=-0.04)
    V, pi, sweeps = S.value_iteration(g.P, g.R, 0.99)
    close(V[g.start], 0.69764, 1e-4, "V* at the start of the 4 x 3 world (slip 0.2, living -0.04, gamma 0.99)")


@test("policy_iteration", "evaluate exactly, take the greedy policy, stop when it no longer changes")
def _():
    P, R = random_mdp(5, S_=8, A=4)
    V, pi, steps = S.policy_iteration(P, R, 0.95)
    Vv, _, _ = S.value_iteration(P, R, 0.95, tol=1e-12)
    close(V, Vv, 1e-7, "policy iteration's V minus value iteration's")
    assert steps <= 10, f"{steps} improvement steps; policy iteration usually needs very few"
    close(S.greedy_policy(S.q_from_v(P, R, V, 0.95)), pi, 0, "the returned policy should be greedy with respect to its own value")


# ---------------------------------------------------------------- Part 3

@test("td0_prediction: the update", "V[s] += alpha (r + gamma V[s'] - V[s]) after every step; V[s'] counts as 0 when done")
def _():
    env = Script(4, [(0, [(1, 0.0, False), (2, 1.0, True)])], n=2)     # 0 -> 1 -> terminal 2
    V = S.td0_prediction(env, 0.5, 0.9, 1, np.random.default_rng(0), V0=0.4)
    close(V[:2], [0.4 + 0.5 * (0.9 * 0.4 - 0.4), 0.4 + 0.5 * (1 - 0.4)], 1e-12, "V after one scripted episode")


@test("mc_prediction: the update", "after the episode, V[S_t] += alpha (G_t - V[S_t]) with G_t = R_{t+1} + gamma G_{t+1}")
def _():
    env = Script(4, [(0, [(1, 0.5, False), (2, 1.0, True)])], n=2)
    V = S.mc_prediction(env, 0.5, 0.9, 1, np.random.default_rng(0), V0=0.4)
    close(V[:2], [0.4 + 0.5 * (0.5 + 0.9 - 0.4), 0.4 + 0.5 * (1 - 0.4)], 1e-12, "V after one scripted episode")


@test("prediction on the random walk", "if the update tests pass, check that each episode starts from env.reset() and runs until done")
def _():
    env = S.RandomWalk(5)
    for f, tol in [(S.td0_prediction, 0.03), (S.mc_prediction, 0.05)]:
        Vs = np.array([f(env, 0.02, 1.0, 400, np.random.default_rng(seed))[:5] for seed in range(10)])
        err = np.abs(Vs.mean(0) - env.true_values).max()
        assert err < tol, f"{f.__name__}: average over 10 runs is off the true values by {err:.3f}"


# ---------------------------------------------------------------- Part 4

@test("epsilon_greedy", "rng.random() < eps: random action from the valid ones; else a greedy one, ties broken at random")
def _():
    rng = np.random.default_rng(0)
    a = np.array([S.epsilon_greedy(np.array([0., 2., 2., 1.]), 0.2, rng) for _ in range(8000)])
    fr = np.bincount(a, minlength=4) / 8000
    close(fr, [0.05, 0.45, 0.45, 0.05], 0.02, "action frequencies with eps = 0.2 and a tie between actions 1 and 2")
    a = [S.epsilon_greedy(np.array([9., 0., 1., 5.]), 0.5, rng, valid=[1, 2]) for _ in range(2000)]
    assert set(a) <= {1, 2}, "only actions in `valid` may be chosen"
    assert 0.7 < np.mean(np.array(a) == 2) < 0.8, "among valid actions [1, 2], action 2 is greedy: expect 0.75 of the choices"


@test("the TD targets", "Expected SARSA: probabilities eps/A each, plus (1 - eps) split among the tied greedy actions")
def _():
    Q = np.array([[0., 0., 0., 0.], [1., 3., 2., 3.]])
    close(S.sarsa_target(Q, -1.0, 1, 2, False, 0.9), -1 + 0.9 * 2, 1e-12, "SARSA target")
    close(S.q_learning_target(Q, -1.0, 1, False, 0.9), -1 + 0.9 * 3, 1e-12, "Q-learning target")
    close(S.expected_sarsa_target(Q, -1.0, 1, False, 0.9, 0.1), -1 + 0.9 * 2.925, 1e-12, "Expected SARSA target")
    for f in (lambda: S.sarsa_target(Q, 2.0, 1, 2, True, 0.9), lambda: S.q_learning_target(Q, 2.0, 1, True, 0.9),
              lambda: S.expected_sarsa_target(Q, 2.0, 1, True, 0.9, 0.1)):
        close(f(), 2.0, 1e-12, "a target at a terminal transition (just r)")


@test("double_q_update", "the updated table chooses the next action, the other table evaluates it")
def _():
    Q1 = np.array([[0., 0.], [1., 5.]]); Q2 = np.array([[0., 0.], [4., 2.]])
    S.double_q_update(Q1, Q2, 0, 1, 1.0, 1, False, 0.5, 0.9, True)
    close(Q1[0, 1], 0.5 * (1 + 0.9 * 2), 1e-12, "Q1[0, 1] after updating Q1 (argmax by Q1 is action 1, valued by Q2 at 2)")
    close(Q2, [[0, 0], [4, 2]], 0, "Q2 must not change when Q1 is updated")
    S.double_q_update(Q1, Q2, 0, 0, 0.0, 1, False, 0.5, 0.9, False)
    close(Q2[0, 0], 0.5 * 0.9 * 1, 1e-12, "Q2[0, 0] after updating Q2 (argmax by Q2 is action 0, valued by Q1 at 1)")


@test("run_control on the cliff", "SARSA bootstraps from the action it will take; Q-learning from the greedy one")
def _():
    env = S.GridWorld(S.CLIFF, living=-1.0)
    out = {m: [S.run_control(env, m, 500, 0.5, 0.1, 1.0, np.random.default_rng(seed)) for seed in range(3)] for m in ("sarsa", "q_learning")}
    sarsa = np.mean([t[-100:].mean() for _, t, _ in out["sarsa"]]); ql = np.mean([t[-100:].mean() for _, t, _ in out["q_learning"]])
    assert sarsa > -32, f"SARSA's average reward over the last 100 episodes is {sarsa:.1f}; the safe path gives about -23"
    assert -60 < ql < -35, f"Q-learning's average reward over the last 100 episodes is {ql:.1f}; the edge path with exploration gives about -45"
    Q = out["q_learning"][0][0]
    s, n = env.start, 0
    while not env.terminal[s] and n < 50:
        s, _, _ = env.step(s, int(np.argmax(Q[s])), None); n += 1
    assert n == 13, f"Q-learning's greedy path takes {n} steps; the optimal path along the cliff takes 13"


@test("run_control: double Q-learning removes the bias", "for double_q, act eps-greedily on Q1 + Q2 and flip a fair coin for which table to update")
def _():
    env = S.MaxBias()
    left = {m: np.mean([(S.run_control(env, m, 20, 0.1, 0.1, 1.0, np.random.default_rng(seed), valid=env.valid)[2] == 0).mean()
                        for seed in range(100)]) for m in ("q_learning", "double_q")}
    assert left["q_learning"] > 0.6, f"Q-learning went left in {left['q_learning']:.2f} of its first 20 episodes; with the bias, expect about 0.8"
    assert left["double_q"] < left["q_learning"] - 0.2, f"double Q-learning went left in {left['double_q']:.2f} of its first 20 episodes, Q-learning in {left['q_learning']:.2f}"


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
