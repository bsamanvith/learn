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

## Where the sections live

| Folder | Holds | Why separate |
|---|---|---|
| `text/` | Sections 1–9 and 12: intuition and rigour in one flow | "Never only one" — the picture and the proof belong on the same page |
| `exercises/` | Sections 10–11: problems, then solutions at the end of the file | Solutions out of sight make you actually attempt the problem (retrieval practice) |
| `widgets/` | Interactive HTML for the chapter | Opened from links in `text/` |
| `sims/` | Python scripts that check results | Referenced from the text and solutions |

Each chapter is one file in `text/` and one matching file in `exercises/`, both named
`NN-topic.md` (e.g. `text/1.3-counting.md` ↔ `exercises/1.3-counting.md`).

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
