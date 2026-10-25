---
title: "The bandit casino"
kind: assignment
time: "90 min"
after: "32"
---

# The bandit casino

You walk into a casino with ten slot machines. Each pays out a random amount around its
own hidden average, and you have 1,000 pulls. Pull the machine that has paid best so far,
and you may never discover a better one; keep trying others, and you waste pulls on
losers. Tonight you build five gamblers, send each into 2,000 casinos, and measure who
walks out richest. One of them, Thompson sampling, does it by literally rolling dice
from its own beliefs.

**Deliverable:** one script, `casino.py`, and two plots: the fraction of casinos in which
each agent is on the best machine, against time; and cumulative regret on a log axis.

## Before you start

- Lesson 32 (bandits and exploration).
- The testbed (Sutton and Barto, §2.3): for each of $R = 2000$ casinos, draw ten true
  means $q^*(a) \sim N(0, 1)$; a pull of arm $a$ pays $N(q^*(a), 1)$. Use
  `rng = np.random.default_rng(0)` for the means and a second generator,
  `np.random.default_rng(1)`, for everything the agents and payouts draw.
- Vectorise over casinos: keep arrays of shape (2000, 10) for estimates $Q$ and counts
  $N$, and play all 2,000 casinos one pull at a time. 1,000 pulls then take a second or two.
- Break ties at random: `np.argmax` always picks the first maximum, which quietly biases a
  greedy agent towards arm 0.

**You should see:** the best arm's mean, averaged over casinos, is 1.535. That is the
most any gambler can earn per pull.

## Step 1 · The greedy gambler and the ε-greedy ones

Estimate each arm by the average of its payouts, $Q \leftarrow Q + (r - Q)/N$. The greedy
agent always pulls $\arg\max Q$; ε-greedy pulls a random arm with probability ε. Record
the average reward, whether the best arm was pulled, and the regret
$q^*(a^*) - q^*(a_t)$, all averaged over casinos.

**Predict first:** which earns more over 1,000 pulls, ε = 0.1 or ε = 0.01?

**You should see** (averages over the last 100 pulls, and total regret):

| agent | reward | on the best arm | regret after 100 | after 1,000 |
|---|---|---|---|---|
| greedy | 1.040 | 36.0% | 56.8 | 504.6 |
| ε = 0.01 | 1.301 | 59.1% | 55.6 | 343.6 |
| ε = 0.1 | 1.367 | 80.1% | 51.3 | 228.2 |

The greedy gambler locks onto whichever machine looked good first and is on the best one
in only 36% of casinos, forever. ε = 0.1 explores enough to find the best arm but keeps
wasting 10% of its pulls, so its regret keeps growing in a straight line; ε = 0.01 finds
the best arm more slowly but would overtake ε = 0.1 over a long enough game. Try
$T = 10{,}000$ to see it.

## Step 2 · Three smarter gamblers

1. **Optimism.** Greedy, but start every estimate at $Q_0 = 5$ and update with a constant
   step $\alpha = 0.1$. Every arm looks wonderful until it disappoints, so greedy is forced to
   try them all.
2. **UCB.** Pull $\arg\max_a\big(Q(a) + c\sqrt{\ln t/N(a)}\big)$ with $c = 2$, pulling each
   untried arm first. The bonus is large for arms you know little about.
3. **Thompson sampling.** Keep a Gaussian belief for each arm: with prior $N(0, 1)$ and
   $n$ payouts summing to $S$, the posterior is $N\big(\frac{S}{n + 1}, \frac1{n + 1}\big)$.
   Each pull, draw one sample from every arm's belief and pull the arm whose sample is largest.

**You should see:**

| agent | reward | on the best arm | regret after 100 | after 1,000 |
|---|---|---|---|---|
| optimistic $Q_0 = 5$ | 1.501 | 85.6% | 118.0 | 240.9 |
| UCB, $c = 2$ | 1.483 | 85.8% | 57.8 | 148.8 |
| Thompson | **1.518** | **91.6%** | **37.9** | **68.9** |

Optimism pays a heavy price up front (118 regret in the first 100 pulls, while it
works through its disappointments) and then plays well. UCB's regret curve bends over like
$\log t$. Thompson sampling wins on every column, with a seventh of greedy's regret, and has
no parameter to tune: its exploration shrinks exactly as fast as its uncertainty does.

**Plot 1:** percentage of casinos on the best arm against pull number, one curve per
agent (smooth over 20 pulls). **Plot 2:** cumulative regret against pull number with a log
$x$ axis: ε-greedy's curves bend upwards, UCB's and Thompson's flatten.

## Step 3 · Break it: a casino that changes

Halfway through, at pull 500, add $N(0, 1)$ noise to every true mean, so the best machine
changes. Rerun all six agents.

**Predict first:** which agents notice? Sample averages ($1/N$ step sizes) give the
thousandth payout a weight of $1/1000$, so they barely move.

**You should see** (the share of casinos on the *new* best arm over the last 100 pulls,
against the share on the old best arm just before the change):

| agent | before | after |
|---|---|---|
| greedy | 35.9% | 27.6% |
| ε = 0.01 | 49.2% | 47.8% |
| ε = 0.1 | 75.2% | 59.1% |
| optimistic, $\alpha = 0.1$ | 81.0% | 73.0% |
| UCB, $c = 2$ | 79.7% | **76.8%** |
| Thompson | **87.9%** | 68.8% |

Every agent loses ground, and the ranking changes. Thompson's posterior had become
confident, so it stopped exploring just when exploring mattered again. UCB recovers best,
because its bonus $\sqrt{\ln t/N}$ keeps growing for arms it has neglected. The constant
step $\alpha$ helps the optimistic agent too: it forgets old payouts. Greedy is on the new
best arm only when its locked-in arm happens to be it. For a casino that keeps
changing, use constant step sizes or a sliding window.

## Deliverable

`casino.py` printing both tables, and the two plots. Add one sentence: why does
Thompson sampling explore less and less without anyone telling it to?

## Stretch

- **Tune ε and c.** Plot average reward over all 1,000 pulls against ε (and against $c$) on
  a log axis, as in Sutton and Barto's figure 2.6. Which agent is least sensitive?
- **Bernoulli casino.** Make each machine pay 1 with probability $p_a$ and use Beta
  posteriors for Thompson. Compare with the Lai–Robbins lower bound from
  [problem set 8](../lessons/P8-problems-stage-8.html), problem 5.
- **A fair comparison with UCB's theory.** Problem 39 of the
  [hard ML volume](../lessons/H2-hard-problems-ml.html) bounds UCB's pulls of a bad arm by
  $8\ln T/\Delta^2$. How loose is that here?

## Reference solution

<details><summary>Show the reference solution</summary>

[`sims/bandit_casino.py`](../sims/bandit_casino.py) prints every checkpoint on this
page; run it with `--plot` for both plots. Its output:

```text
Step 0 · 2000 casinos × 10 arms; mean of the best arm E[max q*] = 1.535

         agent                         reward, last 100   optimal arm, last 100   total regret after 100 / 1000
         greedy                            1.040              36.0%                 56.8 /   504.6
         ε = 0.01                          1.301              59.1%                 55.6 /   343.6
         ε = 0.1                           1.367              80.1%                 51.3 /   228.2
         optimistic Q0 = 5, α = 0.1        1.501              85.6%                118.0 /   240.9
         UCB, c = 2                        1.483              85.8%                 57.8 /   148.8
         Thompson                          1.518              91.6%                 37.9 /    68.9

         (greedy is on the best arm in only 36.0% of casinos at pull 1,000)

Step 3 · the casino changes at pull 500; on the new best arm, last 100 pulls:
         greedy                          27.6%   (just before the change:  35.9% on the old best)
         ε = 0.01                        47.8%   (just before the change:  49.2% on the old best)
         ε = 0.1                         59.1%   (just before the change:  75.2% on the old best)
         optimistic Q0 = 5, α = 0.1      73.0%   (just before the change:  81.0% on the old best)
         UCB, c = 2                      76.8%   (just before the change:  79.7% on the old best)
         Thompson                        68.8%   (just before the change:  87.9% on the old best)
```

Your numbers will differ slightly if your agents consume random numbers in a different
order; differences of a percent or two are noise.

</details>
