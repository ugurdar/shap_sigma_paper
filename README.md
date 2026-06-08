# SHAP-σ: A Label-Free Rashomon Pipeline for Multivariate Time-Series Anomaly Detection

Reference implementation of the conference paper: a **TELCO-only** study
over a self-tuned **K = 10** detector pool. The four numbered scripts in
`code/` reproduce every table and figure, in order.

## Repository layout

```
.
├── code/
│   ├── 1_singletons.py     # Table I  (10 self-tuned singletons)
│   ├── 2_pipeline.py       # Tables II & III + Fig. 1 data (the main pipeline)
│   ├── 3_figure.py         # Fig. 1  (joint-pair scatter)
│   └── 4_robustness.py     # 5-seed robustness (the 0.447±0.013 in the paper)
├── data/                   # not tracked; see "Data" below
│   ├── TELCO_data/
│   └── TELCO_labels/
├── results/                # one CSV per table/figure (see below)
├── paper/  (main.tex, main.pdf, references.bib, figs/)
└── requirements.txt
```

`results/` holds exactly the artifacts behind the paper:

| File | Used for |
|---|---|
| `telco_tuned_singleton.csv` | Table I |
| `telco_B_mixed_sweep.csv`   | Table II (K_R sensitivity sweep) |
| `telco_rule_perkpi.csv`     | Table III (label-free quality-rule set) |
| `telco_B_pair_landscape.csv`| Fig. 1 |
| `surrogate_fidelity.csv`    | surrogate R² (Sec. II / Limitations) |
| `telco_robustness.csv`      | 5-seed stability (`0.447±0.013`) |

## Quick start (reproduces the paper)

```bash
pip install -r requirements.txt
cd code

python 1_singletons.py    # Table I    -> results/telco_tuned_singleton.csv
python 2_pipeline.py      # Tables II/III, Fig.1 data, surrogate R2
python 3_figure.py        # Fig. 1     -> paper/figs/fig_jointpair_telco.{pdf,png}
python 4_robustness.py    # 5-seed stability -> results/telco_robustness.csv
```

All scripts read `../data` and write `../results`, so run them from `code/`.
`2_pipeline.py` is the main pipeline: Stage 1 MV-proxy quality (Eq. 1),
Stage 2 joint-subset selection, SHAP-σ gate. It fixes the Rashomon set
label-free by a variance-stability rule (above-median quality `q_d`),
which selects five detectors at `F1=0.460` (anchor LOF), and also emits
the `K_R` sensitivity sweep showing `F1` plateaus at `0.45–0.46` for
`K_R≥3`. Runtime ≈ 3–5 min per script (AutoEncoder fit + TreeSHAP over
10 detectors dominate; `4_robustness.py` repeats the fit over 5 seeds, ≈ 5×).

## Pipeline architecture (Sec. II of the paper)

Stage 1 — Label-free Mass-Volume proxy quality `q_d ∈ [0,1]`
(Eq. 1; LightGBM surrogate, three random low-dim subspaces, 2 000 uniform
background samples each, then min–max inversion). On TELCO this ranks LOF
(the best singleton) highest, so the anchor `a = argmax_d q_d` is LOF.

Stage 2 — Joint Rashomon-set selection over all `C(K, K_R)` subsets
(`K = 10`); three strategies (Quality, Diversity, Mixed). For `K_R = 2`
Mixed reduces to the closed-form joint pair score
`(q_i + q_j)/2 · (1 − ρ_{ij})` (Eq. 2). The reported sweep uses Mixed.

Stage 3 — LightGBM surrogate per selected detector, TreeSHAP
attributions, detector-wise global std-normalization (`/ (std(Φ_d^v)+1e-12)`).

Stage 4 — Cross-detector standard deviation `σ_{ij}` (Eq. 3), per-feature
0.75-quantile threshold `τ_j` on validation, gate `φ_{ij} = 1[σ_{ij} > τ_j]`.

Stage 5 — Anchor κ-sigma flag `f^{(a)}` (κ = 3) combined with the gate:
`Ŷ_{ij} = f^{(a)}_i ∧ φ_{ij}` (Eq. 5; per-KPI). Only this fusion rule is
reported (no per-series test-label peeking).

## Defaults

| Setting | Value |
|---|---|
| Detector pool (K = 10) | IForest, LOF, OCSVM, ECOD, COPOD, KNN, PCA, AutoEncoder, GMM, HBOS |
| Self-tuning | PCA 95% var; kNN/LOF k-distance elbow; IForest path-length plateau; GMM BIC; HBOS Sturges; OCSVM ν=0.05; AE bottleneck = tuned PCA dim |
| Set selection | label-free variance-stability rule: `q_d > median(q)` (→ 5 detectors); closed-form pair = `K_R=2` case |
| Anchor flag | κ-sigma with κ = 3 |
| σ-filter quantile | q = 0.75 |
| SHAP surrogate | LightGBM, 500 trees, depth 6, lr 0.05, num_leaves 31, subsample 0.8, colsample_bytree 0.8 |
| MV-proxy surrogate | LightGBM, 80 trees, depth 4, lr 0.1 |
| RNG seed | `SEED = 42` |

## Reproducibility

All randomness (MV-proxy subspace sampling, IForest/GMM/PCA states,
AutoEncoder init, LightGBM bootstrap/column subsampling) is fixed to
`SEED = 42`; ECOD/COPOD/HBOS/KNN/LOF are deterministic. Two clean runs
produce byte-identical CSVs. The headline is the label-free quality-rule
set, `F1 = 0.460`; `F1` stays in `0.45–0.46` for all `K_R≥3` and is
seed-stable (`0.447 ± 0.013` over five seeds, `4_robustness.py`).

## Data

Not bundled. The pipeline reads three files:

- `data/TELCO_data/TELCO_data_train.csv`  (columns `time, TS1…TS12`)
- `data/TELCO_data/TELCO_data_test.csv`
- `data/TELCO_labels/TELCO_labels_test.csv`  (columns `time, TS1…TS12`)

TELCO is the public multivariate cellular-KPI benchmark released with
DC-VAE (González-Vidal et al., 2023). The 80/20 train/validation split is
formed inside the scripts; place the official train/test files as above.

## Reproducing the paper PDF

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

LaTeX requires `IEEEtran` plus `booktabs, balance, algorithm, algorithmic,
multirow, array, xcolor`.
