"""Build Overleaf-ready zip with all paper artifacts.

Bundles:
  - main_en.tex + main_zh.tex
  - refs.bib
  - LATEX_SETUP.md
  - figures/*.pdf (9 vector figures)
  - README with compilation instructions

Output: paper/overleaf_export/tcd_paper_overleaf.zip
"""
from __future__ import annotations

import zipfile
import sys
from pathlib import Path


PAPER_DIR = Path("paper")
FIG_DIR = Path("figures")
OUT = PAPER_DIR / "overleaf_export" / "tcd_paper_overleaf.zip"


def build_zip():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files_to_include = []

    # Main .tex files
    for tex in ["main_en.tex", "main_zh.tex"]:
        p = PAPER_DIR / tex
        if p.exists():
            files_to_include.append(p)

    # References (assume refs.bib exists or generate from inline)
    refs_bib = PAPER_DIR / "refs.bib"
    if not refs_bib.exists():
        # Create a stub refs.bib from the inline thebibliography environment note
        with open(PAPER_DIR / "refs.bib", "w") as f:
            f.write("% Stub refs.bib - main_*.tex use thebibliography environment inline.\n")
            f.write("% This file is provided for Overleaf compatibility.\n")
        print("[build] generated stub refs.bib")
    files_to_include.append(PAPER_DIR / "refs.bib")

    # LATEX_SETUP.md
    setup_md = PAPER_DIR / "LATEX_SETUP.md"
    if setup_md.exists():
        files_to_include.append(setup_md)

    # All figures
    if FIG_DIR.exists():
        for pdf in FIG_DIR.glob("*.pdf"):
            files_to_include.append(pdf)
        print(f"[build] found {len(list(FIG_DIR.glob('*.pdf')))} figures")

    # README for Overleaf
    readme = PAPER_DIR / "overleaf_export" / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("""# TC-GNN Paper — Overleaf Upload Package

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
- Figures use `\\includegraphics` with paths relative to the project root

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
""", encoding="utf-8")
    files_to_include.append(readme)

    # Compile zip
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_include:
            arcname = f.name  # top-level flatten
            zf.write(f, arcname=arcname)
            print(f"  + {arcname}")

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"\n[build] Wrote {OUT} ({size_mb:.1f} MB)")
    print(f"[build] Upload to https://www.overleaf.com")


if __name__ == "__main__":
    build_zip()