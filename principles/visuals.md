# Visuals

Pictures, diagrams, animations and interactive widgets.

## When a visual earns its place

A visual must answer a question you could write in one sentence:
*"Why does conditioning shrink the sample space?"*, *"What does the CLT look like as n grows?"*
If you can't write the question, the visual is decoration — cut it.

## Choosing the medium

| Medium | Use when | Tool (see [../tools/](../tools/)) |
|---|---|---|
| **Static figure** | One moment tells the story (a Venn diagram, a density curve) | matplotlib / SVG |
| **Interactive widget** | There's a parameter worth wiggling (n, p, a prior, a correlation) | HTML + JS widget |
| **Animation** | The *process* is the point (sampling, a random walk, a limit forming) | Widget with play button, or Manim for polished video |
| **Simulation** | You want the reader to *run the experiment* | Widget with "draw 1 / draw 1000" buttons |

Default to **interactive over animated**: an animation shows what we chose; a slider
lets the reader ask their own question.

## Design rules

1. **One idea per figure.** Two ideas → two figures.
2. **Before / after** beats a single state. Show the prior and the posterior side by side.
3. **Label directly** on the plot instead of using legends where possible.
4. **Same colour, same meaning**, across the whole subject. Define the palette per
   subject (e.g. in probability: event A always blue, event B always orange,
   intersection purple, complement grey).
5. **Show the numbers.** A slider should display its value; a histogram should show
   the theoretical curve overlaid.
6. **Works in every theme** (light, paper, dark: colours only from `tools/widgets/tokens.css`), and is readable on a phone.
7. **Accessible:** never encode meaning in colour alone; add shape or label.
8. **Visualise answers, not just questions** — e.g. plot P(at least one shared birthday)
   against group size, not just the formula.

## Recurring visual vocabulary for probability

- **Area = probability.** The unit square as the sample space; events as regions.
- **Probability trees** for sequential experiments and Bayes.
- **Grids of dots** (e.g. 1000 people) for base rates and conditional probability.
- **Histograms converging to densities** for limits.
- **Sample paths** for random walks, Markov chains, Brownian motion.

## Images from elsewhere

Only use images we've created or that are clearly licensed for reuse (credit them in
the caption). Store under `subjects/<subject>/assets/`.
