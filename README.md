# learn

Personal learning repo for science: machine learning, probability, statistics, physics,
astronomy and more. It's written to be **fun to read, deep enough to trust, and impossible to
finish without doing something.**

Every topic puts intuition and rigour side by side, and comes with full derivations, exercises
with solutions, simulations that check the maths, and figures you can play with.

**Website:** https://bsamanvith.github.io/learn/

**Status:**
- Machine learning: Stage 1 written (linear regression, gradient descent, logistic regression).
- Probability: Parts 0–1 drafted (8 chapters, exercises, widgets, sims).

## Layout

```
site/                  the website (Astro): one book per subject, a library home page, the Playground
subjects/index.json    every subject, in display order (slug, title, tagline)
subjects/<subject>/
  meta.json            book title, intro, part (or stage) titles and blurbs
  PLAN.md              syllabus
  text/                markdown chapters: hook → intuition → definitions → proofs → pitfalls
  exercises/           markdown problems + full solutions
  lessons/ + lessons.json   hand-written HTML lessons (machine learning uses these)
  pages/               hand-written HTML pages inside the site layout
  widgets/             standalone interactive HTML
  sims/                Python simulations
  assets/              figures, scripts and styles the subject needs
principles/            how we teach: philosophy, page anatomy, exercises, visuals, style guide
tools/                 shared widget template, theme tokens + JS helpers, figure style
```

A subject can be written as markdown (`text/`), as hand-crafted HTML (`lessons/`), or both. The
site turns either into the same book, with navigation, contents, prev/next, "Recently added" and
"Continue where you left off". Hand-written HTML is published as written. Write its links from
the site root (`/machine-learning/02-gradient-descent/`) and the site adds the `/learn` prefix.

## Run it locally

```
pnpm -C site install
pnpm -C site dev        # http://localhost:4321/learn/
```

`pnpm -C site build` writes the static site to `site/dist/`.

## Publishing

`.github/workflows/pages.yml` builds the site on every push to `main` and publishes it to
GitHub Pages. One-time setup: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The old `ml/*.html` addresses redirect to their new pages.

## Start here

1. [principles/philosophy.md](principles/philosophy.md)
2. [subjects/machine-learning/PLAN.md](subjects/machine-learning/PLAN.md) and
   [subjects/probability/PLAN.md](subjects/probability/PLAN.md)
