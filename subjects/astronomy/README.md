# Astronomy

The subject starts with practice: problem sets in `lessons/` (hand-written HTML, each an
`<article class="lesson problems">`), listed in `lessons.json`. The site gathers them into the
problem book at `/astronomy/problems/`, together with the named classics and key books in
`problems/index.json` (see [principles/problem-books.md](../../principles/problem-books.md)).

- Interactive figures live in `assets/scenes/` and register with the shared engine
  `assets/learn.js`. `assets/learn.js` and `assets/ml.css` are symlinks to the machine-learning
  copies, so every subject's hand-written pages look and behave the same.
- Problems are tagged key, warm-up, core and hard. Keep about five key problems per set.
- Lessons for each topic will come later; the problem sets stay as their practice.
