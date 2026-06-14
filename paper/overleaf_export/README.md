# TC-GNN Paper — Overleaf Upload Package

## Quick start

1. Go to https://www.overleaf.com and create a new project
2. Click "Upload" and select `tcd_paper_overleaf.zip`
3. In the project menu (top-left), set:
   - **Compiler**: `pdflatex` for `main_en.tex`, `XeLaTeX` for `main_zh.tex`
4. Click "Recompile"
5. The PDF should appear in the right panel

## Files included

- `main_en.tex` — English manuscript (~13k words, IEEEtran TNNLS format)
- `main_zh.tex` — Chinese manuscript (~13k chars, ctex + IEEEtran)
- `refs.bib` — Bibliography stub (both .tex files use inline thebibliography)
- `LATEX_SETUP.md` — Detailed LaTeX toolchain documentation
- `figures/*.pdf` — 9 vector figures

## Compilation notes

- The English version uses standard `pdflatex` with `IEEEtran` document class
- The Chinese version needs `XeLaTeX` with `ctex` package installed
- Both use the `algorithm`, `algpseudocode`, `booktabs`, `multirow` packages
- Figures use `\includegraphics` with paths relative to the project root

## Figures (in order of appearance)

1. `architecture.pdf` — TC-GNN three-layer architecture
2. `case_studies.pdf` — Positive vs negative cycle visualization
3. `sensitivity.pdf` — Width × window heatmap
4. `isomorphism.pdf` — Trade-crypto structural mapping
5. `ablation.pdf` — 5-component ablation bar chart
6. `training_curves.pdf` — Loss and val AUC-PR curves
7. `results_comparison.pdf` — 8-model headline comparison
8. `bootstrap_ci.pdf` — Bootstrap 95% CI visualization
9. `stress_test.pdf` — Near-miss vs realistic AML stress test

## Reproducing experiments

See `../DELIVERABLES.md` for code/data reproduction instructions.
