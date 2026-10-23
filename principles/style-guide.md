# Style guide

## Voice

- Talk to one reader, as "you". Curious, warm, a little playful. Never condescending.
- Short paragraphs. One idea per paragraph.
- Say *why* before *what*.
- Prefer concrete numbers before general symbols: do the 3-dice case, then n dice.
- Humour is welcome when it helps memory. Not at the cost of correctness.

## Maths

- LaTeX via `$...$` inline and `$$...$$` display (KaTeX-compatible: no exotic packages).
- Introduce every symbol in words the first time it appears in a chapter.
- Probability notation: $P(A)$, $P(A \mid B)$, $\mathbb{E}[X]$, $\operatorname{Var}(X)$,
  $X \sim \text{Bin}(n, p)$, $\mathbb{1}_A$ for indicators. Keep a notation table per
  subject in its README.
- Name theorems and problems; refer to them by name and number.

## Files

- Markdown (`.md`) by default; `.mdx` only when a page embeds a widget component.
- Filenames: `<chapter>-kebab-case.md`, e.g. `2.1-conditioning.md`.
- Code for simulations: Python, standard library + numpy + matplotlib unless there's a reason.

## Citations

Credit sources in a "Further reading" section at the end of each chapter. Paraphrase,
never copy text from copyrighted books.
