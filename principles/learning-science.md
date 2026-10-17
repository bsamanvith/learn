# Learning science

What the evidence says, and what we do about it. Ratings follow Dunlosky et al. (2013),
*Improving Students' Learning With Effective Learning Techniques*, Psychological Science
in the Public Interest 14(1), 4–58, plus later meta-analyses and the cognitive-load
literature on worked examples.

## Strong evidence — build these into every chapter

| Technique | What it is | How the repo uses it |
|---|---|---|
| **Retrieval practice** | Pulling an answer out of memory (testing yourself) beats re-reading it. | "Quick check" questions inline after each concept; problems before solutions; solutions collapsed by default. |
| **Spaced practice** | Revisiting material at growing intervals beats cramming. | Each chapter's problem set includes 2–3 "callback" problems from earlier chapters. A review deck per part. |
| **Worked → faded examples** | Novices learn faster from studying full solutions, then partially completed ones, then solving alone. | Every core idea: one fully worked example, one *faded* example (steps blanked), then problems. |

## Moderate evidence — use where it fits

| Technique | What it is | How the repo uses it |
|---|---|---|
| **Interleaving** | Mixing problem types forces you to choose the method, not just execute it. | Mixed problem sets at the end of each Part, not just each chapter. |
| **Self-explanation** | Explaining *why* each step works. | "Why does this step hold?" prompts inside worked examples. |
| **Elaborative interrogation** | Asking "why is this true?" and "what if it weren't?" | "What breaks if…" questions (drop independence, change a parameter). |
| **Dual coding** | Words + pictures together, each carrying meaning. | Every key result has a figure that encodes it (see [visuals.md](visuals.md)). |

## Weak evidence — avoid relying on these

Re-reading, highlighting, and summarising alone. They *feel* productive (fluency
illusion) but produce little durable learning. The repo should never be "just read it."

## The expertise-reversal effect

Worked examples help beginners but **slow down people who already know the material** —
the extra guidance becomes noise. So:

- Guidance fades within a chapter (worked → faded → independent).
- Stretch problems come with hints, not walkthroughs.
- Experienced readers can jump straight to the problems; the chapter must allow it.

## Desirable difficulty

Some struggle is the point. A problem you can't immediately solve but *can* solve
with effort is the best kind. Each problem tier (see [exercises.md](exercises.md)) aims
for that zone at a different level.

## Practical loop for a study session

1. Attempt the chapter's warm-up problems cold (retrieval, even before reading).
2. Read + play with the visuals.
3. Work the faded example without looking.
4. Do core problems. Check solutions only after a real attempt.
5. Next session: do the callback problems first.
