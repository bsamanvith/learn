# Exercises

Practice is the core of maths learning here, not an appendix.

## Tiers

| Tier | Tag | Purpose | Typical count / chapter |
|---|---|---|---|
| Warm-up | 🟢 | One step from the definition. Can you apply it at all? | 3–5 |
| Core | 🟡 | The skills the chapter promises. Multi-step. | 5–8 |
| Stretch | 🔴 | Needs an idea, not just a procedure. Hints available. | 2–4 |
| Proof | 🟣 | Prove or disprove. Builds rigour. | 1–3 |
| Callback | 🔁 | From earlier chapters, mixed in (spacing + interleaving). | 2–3 |
| Simulate | 💻 | Check a result or explore numerically. | 1–2 |

## Rules for problems

1. **Every problem has a full solution.** No "left to the reader."
2. **Solutions explain the reasoning**, then the number. A solution that's just
   arithmetic teaches nothing.
3. **Name the wrong turn.** If there's a common mistake, the solution says what it is
   and why it fails.
4. **Multiple methods when they exist.** Counting vs. conditioning vs. symmetry vs.
   simulation. Seeing two routes to an answer is half the skill.
5. **Problems have names.** "The broken stick", "The hat-check problem" — names make
   them memorable and cross-referenceable.
6. **Visualise the answer when it helps**, not just the question. A plot of the answer
   as a parameter changes is often the real insight.
7. **Hints are graded:** Hint 1 nudges direction, Hint 2 names the tool, Hint 3 gives
   the setup.

## Format

```markdown
### 2.1.7 🟡 The two-child problem

A family has two children. At least one is a girl. What's the probability both are girls?

<details><summary>Hint 1</summary>List the sample space.</details>

<details><summary>Solution</summary>

**Answer: 1/3.**

Sample space: {GG, GB, BG, BB}, equally likely. Conditioning on "at least one girl"
removes BB, leaving three equally likely outcomes, one of which is GG.

**Tempting wrong answer: 1/2.** That treats "at least one is a girl" as if it said
"the *older* one is a girl" — a different, more informative event.

**Check by simulation:** see `sims/two_child.py`.
</details>
```

## Sources of problems

Original problems first. Classic problems are welcome and credited (e.g. "adapted from
Blitzstein & Hwang, Ex. 2.4"). Never copy problem text or solutions verbatim from a
copyrighted book — rewrite the setting and write our own solution.
