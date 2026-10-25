---
title: "The ménage problem"
kind: hard
stars: 3
unlocks: "1.5"
uses: ["1.3", "1.4", "1.5"]
---

# The ménage problem

*French* ménage: *household*. In 1891 Édouard Lucas asked a dinner-party question that
looks like a small variation on the [hat-check problem](../text/1.5-inclusion-exclusion.md),
and turns out to need one more idea.

$n$ married couples sit at a round table, **alternating** woman, man, woman, man. The
women take their seats first. The men then sit down at random, one in each gap between
two women. What is the probability that **no man sits next to his partner**?

**Guess first.** In the hat-check problem each person has *one* hat that would be a
match, and the answer is about $1/e \approx 0.37$. Here each man has *two* forbidden
seats (either side of his partner). Is the answer about $0.37^2$? Something else? Does
it settle down as $n$ grows, as hat-check does?

> 🎛 **Play:** [the ménage problem](../widgets/menage.html). Reseat the men a few times
> for $n = 6$ and count how often nobody is beside their partner. Then compare the two
> plots under the table: how fast does each problem reach its limit?

## Hints

<details><summary>Hint 1 · simplify</summary>

Where the women sit doesn't matter: whatever the order, each man has exactly two
forbidden gaps. So fix the women. You are left with a uniformly random way to put
$n$ men into $n$ gaps, with two banned gaps per man.

</details>

<details><summary>Hint 2 · the tool</summary>

Inclusion–exclusion over the bad events "man $j$ sits in a gap next to his partner".
As in chapter 1.5, you need to know: in how many ways can you choose $k$ bad events
that **can all happen at once**, and then how many seatings realise them?

</details>

<details><summary>Hint 3 · the setup</summary>

There are $2n$ banned (gap, man) pairs. Write them in order round the table. Two
**neighbouring** pairs in this list share either a gap or a man, so they can't both
happen. Any two pairs that are **not** neighbours can. So the count you need is the
number of ways to choose $k$ points out of $2n$ points on a circle with no two
next to each other.

</details>

## Solution

<details><summary>Show the solution</summary>

**Answer.**

$$
P_n = \frac{U_n}{n!}, \qquad U_n = \sum_{k=0}^{n} (-1)^k\, \frac{2n}{2n-k}\binom{2n-k}{k}\,(n-k)!
$$

This is **Touchard's formula** (1934). It gives $P_3 = 1/6$, $P_4 = 1/12$,
$P_5 = 13/120 \approx 0.108$, $P_6 = 1/9$, and $P_n \to e^{-2} \approx 0.135$.

**Step 1: the forbidden pairs form a cycle.** Number the women $1, \dots, n$ round the
table and call gap $j$ the gap between woman $j$ and woman $j+1$ (with woman $n+1$
meaning woman 1). Man $j$ must not sit in gap $j-1$ or gap $j$. Write the $2n$
forbidden (gap, man) pairs in this order:

$$
(\text{gap } n, \text{man } 1),\ (\text{gap } 1, \text{man } 1),\ (\text{gap } 1, \text{man } 2),\ (\text{gap } 2, \text{man } 2),\ \dots,\ (\text{gap } n, \text{man } n), \text{ then back to the start.}
$$

Each pair shares a man or a gap with the pairs on either side of it, and with no
others. A set of forbidden pairs can all happen at once exactly when it uses no gap
twice and no man twice, which means **no two of them are neighbours in this cycle**.

**Step 2: inclusion–exclusion.** If we insist on $k$ compatible forbidden pairs, those
$k$ men are placed and the other $n-k$ men can go anywhere: $(n-k)!$ seatings. So

$$
U_n = \sum_{k=0}^{n} (-1)^k\, c(2n, k)\,(n-k)!,
$$

where $c(m, k)$ is the number of ways to choose $k$ of $m$ points on a circle with no
two adjacent.

**Step 3: a story proof for $c(m, k)$.** On a *line* of $m$ points, choosing $k$ with no
two adjacent is the same as placing $k$ chosen points among $m - k$ unchosen ones,
one per gap: $\binom{m-k+1}{k}$ ways (stars and bars, chapter 1.3). On a *circle*, ask
about point 1:

- **Point 1 chosen.** Then its two neighbours are out, and we choose $k-1$ more from
  a line of $m-3$ points: $\binom{m-k-1}{k-1}$.
- **Point 1 not chosen.** Cut the circle there: choose $k$ from a line of $m-1$
  points: $\binom{m-k}{k}$.

Adding, and using $\binom{m-k-1}{k-1} = \frac{k}{m-k}\binom{m-k}{k}$:

$$
c(m, k) = \binom{m-k}{k} + \frac{k}{m-k}\binom{m-k}{k} = \frac{m}{m-k}\binom{m-k}{k}.
$$

**Step 4: the limit.** The $k$-th term of $P_n = U_n/n!$ is

$$
(-1)^k\, \frac{2n}{2n-k}\binom{2n-k}{k}\, \frac{(n-k)!}{n!} \;\longrightarrow\; (-1)^k \frac{(2n)^k}{k!}\cdot\frac{1}{n^k} = \frac{(-2)^k}{k!},
$$

so $P_n \to \sum_k (-2)^k/k! = e^{-2}$. (Swapping the limit and the sum needs a
domination argument, which Part 6 will make routine.)

**Why $e^{-2}$, in one line.** Each man sits beside his partner with probability $2/n$,
so by linearity the expected number of couples sitting together is $n \cdot 2/n = 2$,
**exactly**, for every $n \ge 3$. Rare, nearly independent events with mean 2 behave
like a Poisson(2) count, and $P(\text{Poisson}(2) = 0) = e^{-2}$. Hat-check is the same
story with mean 1.

**Lucas's original count.** Lucas counted seatings of everyone, not a probability.
With $2n$ numbered seats, the women choose the odd or the even seats (2 ways) and an
order ($n!$), and then the men have $U_n$ good orders: $2 \cdot n! \cdot U_n$. For
$n = 5$ that is $3{,}120$ of the $28{,}800$ alternating seatings.

</details>

## Picture the answer

| $n$ | 3 | 4 | 5 | 6 | 8 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| $P_n$ | 0.1667 | 0.0833 | 0.1083 | 0.1111 | 0.1175 | 0.1212 | 0.1284 | 0.1326 | 0.1340 |
| $e^{-2}(1 - 1/n)$ | 0.0902 | 0.1015 | 0.1083 | 0.1128 | 0.1184 | 0.1218 | 0.1286 | 0.1326 | 0.1340 |

Two things to see in the widget's plots:

- **The limit is approached from below, slowly.** From about $n = 5$ on,
  $P_n \approx e^{-2}(1 - 1/n)$ to three decimals: the gap shrinks like $1/n$. Even
  with 100 couples the answer is still 1% below its limit.
- **Hat-check is far faster.** Its gap to $1/e$ is about $1/(n+1)!$, already below
  $10^{-4}$ at $n = 7$. Each hat-check term involves one person, and the terms
  stop overlapping very quickly. The ménage constraints come in a chain (each man's
  forbidden seats overlap his neighbours'), and those overlaps fade only like $1/n$.

## Wrong turns

- **"Each man avoids two of $n$ seats, independently: $(1 - 2/n)^n$."** It has the right
  limit, $e^{-2}$, but the wrong values: $0.078$ against the true $0.108$ at $n = 5$, and
  $0.107$ against $0.121$ at $n = 10$. The men aren't independent: once one man takes a
  gap, nobody else can, and a man who avoids his own forbidden gap makes the others'
  forbidden gaps *more* likely to be free.
- **Treating it as two hat-check problems: $(1/e)^2$.** That multiplies probabilities of
  events that are not independent, and also double-counts. The two forbidden seats per
  man live in *one* permutation.
- **Counting non-adjacent choices as on a line: $\binom{2n-k}{k}$.** This forgets that the
  table is round: the first and last forbidden pairs in the cycle, (gap $n$, man 1) and
  (gap $n$, man $n$), share gap $n$. It happens to give the right count for $n = 3$, then
  goes wrong: 3 instead of 2 for $n = 4$, and 16 instead of 13 for $n = 5$.
- **Trusting the formula without a small case.** For $n = 3$: $c(6,1) = 6$, $c(6,2) = 9$,
  $c(6,3) = 2$, so $U_3 = 1 \cdot 3! - 6 \cdot 2! + 9 \cdot 1! - 2 \cdot 0! = 6 - 12 + 9 - 2 = 1$. Check it
  by hand: with 3 couples, each man has only one allowed gap, so there is exactly one
  good order.

## Check by simulation

[`sims/menage.py`](../sims/menage.py) checks the circle lemma and Touchard's formula
against brute force for $n \le 8$, prints the table above, and seats 200,000 random
dinners for $n = 10$:

```text
Simulated n = 10, 200,000 random seatings: 0.1214  (exact 0.1212)
```

## Going further

- **The mixed table.** Drop the alternation: $2n$ people sit in random seats round the
  table. The expected number of couples together is still easy (linearity again: what
  is it?), and $P(\text{no couple together}) \to e^{-1}$. Why $e^{-1}$ now, not $e^{-2}$?
- **The Poisson picture.** Simulate the *number* of couples sitting together and
  compare it with Poisson(2). This is a preview of Part 3, where "many rare, weakly
  dependent events" becomes a theorem.
- The numbers $U_n = 1, 2, 13, 80, 579, 4738, \dots$ are the **ménage numbers**,
  sequence A000179 in the OEIS.
