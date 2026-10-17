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

## The open decision: how we publish

Right now everything reads fine on GitHub. When we want a real site with live widgets:

| Option | Strengths | Weaknesses |
|---|---|---|
| **Astro + MDX** (+ remark-math / KaTeX) | Most control over custom interactive widgets (islands: any JS framework); fast static site; looks however we want | We wire up math, cross-refs, sidebar ourselves (Starlight theme helps) |
| **MyST / Jupyter Book 2** | Built for textbooks: numbered theorems, cross-refs, exercises/solutions directives, executable notebooks, PDF export | Custom JS widgets are less natural; more "document" than "app" |
| **Quarto** | Excellent books + PDF; built-in Observable JS for reactive widgets; Python execution | Widget styling less flexible than Astro; another toolchain to install |

Leaning: **Astro + MDX** if animations/widgets are the star; **MyST** if the textbook +
exercise structure is the star. Our content format works with either, so this can be
decided after the first chapter.

## For polished animations

[Manim](https://www.manim.community/) (3Blue1Brown-style video) for the few concepts
where a narrated animation beats a slider. Optional, added only when needed.

## Getting started

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r tools/requirements.txt
python3 -m http.server 8000   # then open http://localhost:8000/tools/widgets/_template.html
```
