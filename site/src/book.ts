// Book-level metadata the chapter files don't carry. Part titles follow subjects/probability/PLAN.md.
export const PARTS: Record<number, { title: string; blurb: string }> = {
  0: { title: "What is probability, anyway?", blurb: "Three meanings, a 1654 letter that started it all, and why your gut gets it wrong." },
  1: { title: "Foundations", blurb: "Sample spaces, the axioms, counting, story proofs and inclusion–exclusion." },
  2: { title: "Conditional probability and independence", blurb: "The heart of the subject." },
  3: { title: "Discrete random variables", blurb: "" },
  4: { title: "Continuous random variables", blurb: "" },
  5: { title: "Joint distributions", blurb: "" },
  6: { title: "Inequalities and limit theorems", blurb: "" },
  7: { title: "Stochastic processes", blurb: "" },
  8: { title: "Measure-theoretic foundations", blurb: "" },
  9: { title: "Bridges", blurb: "" },
};

// Sort "1.10" after "1.9".
export const byChapter = (a: { data: { chapter: string } }, b: { data: { chapter: string } }) => {
  const [a1, a2] = a.data.chapter.split(".").map(Number);
  const [b1, b2] = b.data.chapter.split(".").map(Number);
  return a1 - b1 || a2 - b2;
};
