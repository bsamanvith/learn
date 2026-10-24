---
name: learn-and-create
description: The working loop for this repo, where the owner is both the learner and the co-author. Use when writing or revising a chapter, part, widget or exercise set, when the user shares an animation review or reading notes, or when they ask "what next?".
---

# Learn and create

The repo covers many subjects (probability is the first, not the only one). The owner is
**learning each subject while building it**. They are not an
editor reviewing finished text. They are the first student. Every part goes round this loop:

```
 write ─▶ verify ─▶ read & play ─▶ curate ─▶ revise ─▶ next part
 (Claude)  (Claude)   (user, site)   (user)    (Claude)
```

## 1. Write (Claude)

- Follow `principles/`: page anatomy, exercise tiers, style guide, visuals. Plan from
  `subjects/<subject>/PLAN.md`.
- Write for someone meeting the idea for the first time: hook first, intuition before
  formalism, worked → faded examples, no "clearly". Depth comes after, not instead.
- Each chapter: `text/<ch>-topic.md`, `exercises/<ch>-topic.md`, a widget where a picture
  beats prose, and a sim in `sims/`.
- Embed a widget with a blockquote that starts with 🎛 and links to `../widgets/x.html`;
  the site turns it into an inline, theme-synced embed. Register every new widget in
  `widgets/index.json` with `"status": "new"`.
- Widgets take colours only from `tools/widgets/tokens.css` (via `Learn.palette()`), and
  redraw in `Learn.onThemeChange`, so they work in light, paper and dark.

## 2. Verify (Claude), before handing over

- Every number: computed in Python. Every sim: run with `.venv/bin/python`.
- Site: `pnpm -C site build` passes with no `katex-error` in `site/dist`. Then look at
  the new pages in the browser (`preview_start` name `site`): one desktop and one phone
  width, both a light theme and dark, embeds interactive and not clipped.

## 3. Read and play (user)

Point them at the exact URLs: `http://localhost:4321/learn/<subject>/<id>/`, its
`/exercises/`, and `/animations/#<subject>-<widget>`. Suggest they try the quick checks and a few
exercises before opening the solutions. Don't race on to the next part unasked; the
point is that they learn it.

## 4. Curate (user)

On `/animations/` they mark each widget **keep / revise / drop** and write notes.
**Copy review as Markdown** gives a block like:

```
## Animation review (2026-10-18)
- **probability/widgets/birthday.html** (1.3, The birthday problem): **revise** — calendar too small on phone
```

They may also paste reading notes ("1.4 hockey stick didn't click"), or replace
`widgets/index.json` with the downloaded copy.

## 5. Revise (Claude)

- Treat each note as a request: fix, then say what changed. For **drop**, remove the
  embed from the chapter (keep the file until they confirm), and set the status in
  `index.json`.
- Write their verdicts and notes into `widgets/index.json` so the decision is recorded
  in the repo, not only in their browser.
- If something confused them, that is a signal about the text, not about them: add an
  intuition step, a picture or a worked example where it lost them.
- Re-verify (step 2), commit, then offer the next part.

## Where things are

| What | Where |
|---|---|
| Run the site | `pnpm -C site dev` → http://localhost:4321/learn/ |
| Chapter → site rules | `site/src/plugins/remark-learn.mjs` |
| Themes (one source) | `tools/widgets/tokens.css` |
| Site styles | `site/src/styles/global.css` |
| Subject list (home page) | `subjects/index.json` |
| Book title, intro, parts | `subjects/<subject>/meta.json` |
| Widget registry | `subjects/<subject>/widgets/index.json` |
| Plan + progress | `subjects/<subject>/PLAN.md` |
| Site routes | `/<subject>/`, `/<subject>/<id>/`, `/subjects/<subject>/widgets/x.html` |

## Starting a new subject

1. Add or update its entry in `subjects/index.json` (slug, title, mode, tagline).
2. Write `subjects/<slug>/PLAN.md`, then `meta.json` with the parts from the plan.
3. Write the first chapter in `text/`. The subject goes live on the home page as soon as
   it has one, and its widgets show up on `/animations/` once `widgets/index.json` exists.
