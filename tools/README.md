# Tools

Shared tooling for writing, visualising and publishing. Kept deliberately small until
a few chapters exist and we know what we actually need.

## What's decided

| Need | Choice | Why |
|---|---|---|
| Content format | Markdown (+ `.mdx` only for embedded widgets), KaTeX-compatible LaTeX | Readable on GitHub as-is; every site generator below can consume it |
| Static figures | Python + matplotlib (in `tools/figures/`), output SVG | Reproducible: figures are generated from code, never hand-edited |
| Simulations | Python + numpy, scripts next to the chapter (`sims/`) | Short, runnable checks of every result |
| Interactive widgets | Self-contained HTML + JS files (`tools/widgets/`) | Open in any browser, embed in any site generator later, no build step |
| Python env | `uv venv .venv` + `uv pip install -r tools/requirements.txt` | Fast; `.venv/` is gitignored |

### Widget pattern

Every widget follows [`widgets/_template.html`](widgets/_template.html):
controls (sliders, buttons) → a pure `draw(state)` function → SVG/canvas output, plus a
"simulate" button where randomness is involved. Shared helpers (RNG with seed,
histogram, axis drawing, palette) live in [`widgets/lib.js`](widgets/lib.js).

## Publishing: the site

Decided: **Astro**, in [`site/`](../site/). The chapter Markdown stays where it is and is never
edited for the site. [`site/src/plugins/remark-learn.mjs`](../site/src/plugins/remark-learn.mjs)
turns the repo's conventions into page structure:

- a `> 🎛 **Play:** [..](../widgets/x.html)` blockquote becomes an embedded, theme-synced widget;
- `## 1. Hook: …` style headings become styled anatomy sections with an eyebrow label;
- exercise `###` headings become problem cards linked to their `<details>` solutions;
- relative links to `text/`, `exercises/`, `sims/` and `widgets/` become site routes.

Themes (light, beige "paper", dark) live in one file, [`widgets/tokens.css`](widgets/tokens.css),
shared by the site and every widget. Widgets read `?theme=` and `?embed=1`, and `lib.js`
redraws them when the site switches theme. The **Animations** page (`/animations/`) shows every
widget from `subjects/<subject>/widgets/index.json` for review: verdict, notes, and export.

## For polished animations

[Manim](https://www.manim.community/) (3Blue1Brown-style video) for the few concepts
where a narrated animation beats a slider. Optional, added only when needed.

## Getting started

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r tools/requirements.txt   # for sims/
pnpm -C site install
pnpm -C site dev               # the site: http://localhost:4321
```

Widgets also work on their own without the site:

```bash
python3 -m http.server 8000   # then open http://localhost:8000/tools/widgets/_template.html
```
