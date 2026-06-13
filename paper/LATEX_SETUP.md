# LaTeX Toolchain Status (checked 2026-06-13)

Local `pdflatex` / `xelatex` / `bibtex` are **NOT installed** in this environment.

## Strategy

We will write correctly-formed `.tex` files and produce an Overleaf-ready zip
in `paper/overleaf_export/`. Final compilation happens in Overleaf (browser-based,
free, supports `xeCJK` for Chinese).

## To install local LaTeX (optional, for offline iteration)

```bash
# Option A: TinyTeX (lightweight, ~80MB)
curl -L https://yihui.org/tinytex/install-bin-windows.sh -o install-tinytex.sh
bash install-tinytex.sh

# Option B: MiKTeX (full features, ~500MB)
# Download from https://miktex.org/download and install

# After install, verify
pdflatex --version
xelatex --version
bibtex --version
```

## Overleaf export

After Phase 6 writes `main_en.tex` and Phase 7 writes `main_zh.tex`, run:

```bash
cd paper
python overleaf_export/build_zip.py
```

This produces `tcd_paper_overleaf.zip` ready for upload to overleaf.com.