# Page anatomy

The contract every chapter follows. Sections can be short, but none are skipped
without a reason.

```
1. Hook            a puzzle, paradox, bet, or real-world question
2. Intuition       a visual / interactive that makes the answer feel inevitable
3. Definition      the precise statement, with notation introduced once
4. Worked example  every step shown, with "why does this hold?" notes
5. Faded example   same shape, key steps left blank for you
6. Quick checks    1–3 tiny retrieval questions (answers hidden)
7. Theory          theorems + proofs (or derivations); story proof first if one exists
8. Pitfalls        the tempting wrong answer and why it's wrong
9. Simulation      code that checks the main result numerically (where relevant)
10. Problems        tiered — see exercises.md
11. Solutions       collapsed, full, explaining the reasoning not just the number
12. Connections     ← prerequisites it used, → where it's used next
```

## The three books per topic

Each topic folder holds three views of the same material:

| Folder | Role | Reads like |
|---|---|---|
| `concept/` | Intuition-first, visual, playful. Sections 1, 2, 6, 8. | A great popular-science chapter with sliders |
| `text/` | The rigorous textbook. Sections 3, 4, 5, 7, 9, 12. | Blitzstein & Hwang with better pictures |
| `exercises/` | Problems and solutions. Sections 10, 11 + callbacks. | A problem book you actually want to finish |

In practice a chapter can live as one file with all twelve sections; split into the
three folders when a chapter grows large. The contract is the sections, not the files.

## Front matter

Every chapter file starts with:

```yaml
---
title: Conditional probability
part: 2
chapter: "2.1"                 # quoted: YAML would read 2.10 as 2.1
prerequisites: ["1.2", "1.3"]
leads_to: ["2.2", "2.3", "5.4"]
status: draft   # idea | draft | reviewed | polished
---
```
