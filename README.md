# learn

Personal learning repo for science: probability, machine learning, physics, astronomy,
chemistry, biology and more — written to be **fun to read, deep enough to trust, and
impossible to finish without doing something.**

Every topic gets chapters that put intuition and rigour side by side, exercises with full
solutions, simulations that check the maths, and visuals you can play with.

**Status:** probability Parts 0–1 drafted (8 chapters, exercises, widgets, sims); Part 2 next.

## Layout

```
principles/            how we teach: philosophy, learning science, page anatomy,
                       exercises, visuals, how subjects differ, style guide
tools/                 shared tooling: widget template + JS helpers, figure style,
                       Python requirements, publishing decision
subjects/<subject>/
  README.md            notation, colour vocabulary, layout
  PLAN.md              syllabus: parts → chapters → concepts, visuals, problems
  text/                chapters: hook → intuition → definitions → proofs → pitfalls
  exercises/           problems + full solutions
  widgets/             interactive HTML
  sims/                Python simulations
  assets/              figures and images
```

## Subjects

| Subject | Status |
|---|---|
| [Probability](subjects/probability/) | Parts 0–1 drafted — see [PLAN.md](subjects/probability/PLAN.md) |
| Statistics, ML, physics, astronomy, chemistry, biology | later |

## Start here

1. [principles/philosophy.md](principles/philosophy.md)
2. [subjects/probability/PLAN.md](subjects/probability/PLAN.md)
3. Try the widget template: `python3 -m http.server 8000` → <http://localhost:8000/tools/widgets/_template.html>
