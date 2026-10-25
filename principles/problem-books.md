# Problem books

Chapter exercises ([exercises.md](exercises.md)) drill one chapter. A subject also needs
practice that **cuts across chapters**, practice that is **hard** enough to need an idea,
and practice where you **build something and watch it work**. Each subject's problem
book holds that. It has four shelves:

| Shelf | What it is | Size | Lives in |
|---|---|---|---|
| **Key books** | The classic external problem books for the subject, with one line on what each is good for and when to pick it up | 5–10 | `problems/index.json` → `books` |
| **Hard problems** | Named problems that need an idea from more than one chapter. A living list, graded ★ to ★★★★ | 20–40 | `problems/index.json` → `hard`, written ones in `problems/<id>.md` |
| **Assignments** | Small builds for one sitting (60–180 min) that end in something you made: a plot, a working program, a measured number | 8–15 | `problems/index.json` → `assignments`, written ones in `problems/<id>.md` |
| Chapter exercises | Already exist, per chapter | — | `exercises/` or the lesson notes |

The site shows the four shelves at `/<subject>/problems/`, and each written problem at
`/<subject>/problems/<id>/`.

## Keep it fun

- **Every problem has a story and a name.** "The ménage problem", not "Exercise 14".
- **Difficulty is a star rating**, not a tier code: ★ warm, ★★ needs care, ★★★ needs an
  idea, ★★★★ a boss problem that may take a few sittings. The problem book page lets
  you tick off what you've cleared (stored in your browser), so progress is visible.
- **Predict first.** A hard problem opens by asking for a gut guess. The solution
  comes back to that guess.
- **Locked, not hidden.** A problem whose chapters aren't written yet still appears
  in the list, marked with the chapter that unlocks it, as something to look forward to.

## Visualise every important idea

[visuals.md](visuals.md) applies in full. On top of it:

1. **Every hard problem and every assignment names its visual** in the registry, before
   anything else is written. If you can't name one, ask whether the problem is worth
   including.
2. **Visualise the answer, not only the setup.** For a problem with a parameter $n$,
   plot the answer against $n$ and show where it goes in the limit, and how fast.
3. **A written hard problem embeds a widget** where wiggling a parameter teaches
   something. Otherwise a static figure.
4. **An assignment's deliverable includes a plot** that the learner makes themselves.
   The assignment says what that plot should look like, so they can check it.

## Hard problems

Selection rules:

- **It needs an idea**, not only a longer calculation: a clever bijection, conditioning
  on the right thing, a symmetry, an invariant, a change of view.
- **It crosses chapters.** It says which chapter unlocks it (`unlocks`) and which
  earlier ideas it uses.
- **Draw on the plan first.** Each part of `PLAN.md` already lists signature problems.
  The hard list collects them and adds the harder relatives, rather than inventing a
  parallel list.
- **Classics are welcome and credited**, but problem text and solutions are always our
  own (see the copyright rule in [exercises.md](exercises.md)).

A written hard problem has this shape:

```markdown
---
title: "The ménage problem"
kind: hard
stars: 3
unlocks: "1.5"
uses: ["1.3", "1.4", "1.5"]
---

# The ménage problem

Story and statement. Ends with "Guess first: …".

> 🎛 **Play:** [name](../widgets/x.html). One line on what to try.

## Hints
<details><summary>Hint 1</summary>Direction.</details>
<details><summary>Hint 2</summary>The tool.</details>
<details><summary>Hint 3</summary>The setup.</details>

## Solution
<details><summary>Show the solution</summary> … the answer, with the reasoning … </details>

## Picture the answer
The answer as a function of the parameter, its limit, how fast it converges.

## Wrong turns
The tempting approaches, and exactly where they break.

## Check by simulation
A sim in sims/, with its output.

## Going further
A harder variant or the connection it opens.
```

## Assignments

An assignment is not a long exercise. It ends in **something you made**, and it lets you
**experience** an idea in full: you see the divergence, time the slow convergence,
or watch the estimate settle.

Rules:

1. **One sitting.** State the time (60–180 min) and the prerequisites.
2. **A stated deliverable**: a script and one or two plots, or a number with error bars.
3. **Checkpoints with numbers.** After each step, "you should see …" with values we
   computed, so the learner knows they are on track without opening the solution.
4. **Break it on purpose.** At least one step pushes past the edge (too big a step size,
   too few samples, a bad prior) so the failure is seen, not just read about.
5. **A reference solution** in `sims/`, folded away on the page, that reproduces every
   checkpoint.
6. **A stretch goal** for when it went well.

Shape:

```markdown
---
title: "Fit a line three ways, then break it"
kind: assignment
time: "90–120 min"
after: "02"
---

# Fit a line three ways, then break it

What you'll build, why it's worth an evening, and the deliverable.

## Before you start
Prerequisites and setup.

## Step 1 · …
What to do. **You should see:** checkpoint values.

## Step n · Break it
…

## Deliverable
The script and the plots, and what they should show.

## Stretch
…

## Reference solution
<details><summary>Show the reference solution</summary> … </details>
```

## Key books

A short, opinionated list of the external problem books per subject. For each: title,
author, level, and **when to pick it up** (e.g. "after Part 2, for conditioning
practice"). Prefer books with full solutions. Free and open books are marked as such.
We point to these books and never reproduce their problems.

## Registry

`subjects/<subject>/problems/index.json` is the source of truth. Every entry has a
`status` (`planned` → `written` → `keep` / `revise` / `drop` after the owner curates it)
and a `notes` field for the owner's verdict, as in `widgets/index.json`.
