# TCD Paper + TC-GNN Implementation

Bilingual (zh/en) TNNLS-submittable manuscript on **Temporal Cycle Detection (TCD)**
with full reproducible implementation and Elliptic2 experimental results.

## Project Structure

```
tradesupervi/
├── background.txt                 # 论文缘起
├── paperoutline.txt               # 论文大纲
├── docs/plans/                    # 实施计划 + 评审修订清单
├── src/temporal_cycle/            # 算法 + 模型实现
│   ├── indexing.py                # 时间排序邻接索引
│   ├── temporal_dfs.py            # Algorithm 1: 精确回溯
│   ├── temporal_sketch.py         # Algorithm 2: Count-Min 草图
│   ├── tc_gnn.py                  # TC-GNN 架构
│   └── losses.py                  # 约束正则化损失
├── tests/                         # TDD 测试
├── experiments/                   # 实验流水线
│   ├── data_loader.py
│   ├── train_tc_gnn.py
│   ├── eval_all.py
│   ├── ablation.py
│   ├── sensitivity.py
│   ├── case_study.py
│   └── baselines/                 # GCN/GAT/TGN/DCRNN/GLASS/XGB
├── data/elliptic2/                # 数据集 (gitignored raw)
├── paper/                         # 双语稿件
│   ├── main_en.tex                # 英文 IEEEtran
│   ├── main_zh.tex                # 中文 xeCJK
│   ├── refs.bib / refs_zh.bib
│   ├── abstract.tex
│   └── overleaf_export/           # Overleaf 可导入 zip
├── figures/                       # 论文插图
└── results/                       # 实验结果 CSV/JSON
```

## Reproducibility

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data (auto-download Elliptic2, fallback to Elliptic1/synthetic)
python data/download_elliptic2.py
python data/preprocess_elliptic2.py

# 3. Run tests
pytest tests/ -v

# 4. Run main experiments
python experiments/train_tc_gnn.py
python experiments/eval_all.py

# 5. Build paper
cd paper && latexmk -pdf main_en.tex && latexmk -xelatex main_zh.tex
```

## Status

See `docs/plans/2026-06-13-tcd-paper.md` for the full implementation plan with
TNNLS-review-driven revisions (W1–W7, m1–m10, Q1–Q6).