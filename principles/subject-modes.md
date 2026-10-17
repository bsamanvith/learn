# Subject modes

Different subjects are learned differently. The [page anatomy](page-anatomy.md) is the
common skeleton; this file says how each subject bends it. **First draft — revise as we
actually write each subject.**

| | Core loop | Main form of practice | Visual emphasis | "Proof" means |
|---|---|---|---|---|
| **Mathematics** | Puzzle → intuition → definition → theorem → proof → practice | Lots of problems, tiered; proofs | Geometric pictures, interactive parameters | Deductive proof |
| **Probability** (maths + a bit of stats) | Puzzle → intuition → model → compute → **simulate** | Problems + simulation checks | Area-as-probability, trees, convergence | Proof *and* simulation agreement |
| **Statistics** | Data → question → model → inference → check assumptions | Analyse real datasets; interpret, not just compute | Distributions of data and of estimators; resampling | Theorem + "does it hold on data?" |
| **AI / ML** | Problem → maths → code from scratch → experiment → compare to library | Derivations + implement-it exercises + experiments | Loss landscapes, decision boundaries, training curves | Derivation + reproducible experiment |
| **Physics** | Phenomenon → model → prediction → compare with measurement | Problems with units; estimation (Fermi) | Fields, trajectories, phase space, simulations | Derivation + dimensional analysis + experiment |
| **Astronomy** | Observation → scale → physical model → what we infer | Estimation, scale exercises, real data (light curves, spectra) | Scale (powers of ten), sky maps, timelines | Physical argument + observational evidence |
| **Chemistry** | Phenomenon → structure → mechanism → prediction | Mechanism problems, stoichiometry, structure recognition | 3D molecules, reaction coordinate diagrams, orbitals | Mechanism + energetics + experiment |
| **Biology** | Phenomenon → hierarchy (molecule → cell → organism → population) → mechanism → evolution | Recall (spaced!), case studies, mechanism explanation, data interpretation | Pathway diagrams, zoomable hierarchies, population dynamics | Experimental evidence + evolutionary logic |

## Notes per mode

- **Recall-heavy subjects** (biology, parts of chemistry) lean harder on spaced retrieval:
  flashcard-style quick checks matter more than proofs.
- **Derivation-heavy subjects** (maths, physics, ML theory) lean on worked → faded
  examples and proof problems.
- **Empirical subjects** (stats, ML, astronomy) always include a "try it on real data"
  section and a note on where the data came from.
- **Shared foundations** link across subjects instead of being duplicated: probability
  is a prerequisite for statistics, ML, statistical mechanics, population genetics.
  Write it once, link everywhere.
