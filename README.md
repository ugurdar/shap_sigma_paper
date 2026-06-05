# SHAP-σ: A Label-Free Rashomon Pipeline for Multivariate Time-Series Anomaly Detection

Reference implementation and IEEE conference paper.

## Repository layout

```
.
├── code/
│   ├── shap_sigma_pipeline.py    # Main pipeline (Stages 1–5 of the paper)
│   ├── singletons.py             # Per-detector singleton baselines
│   └── make_figures.py           # Reproduces all three figures
├── data/                         # (not tracked; see "Data" below)
│   ├── TELCO_data/
│   ├── TELCO_labels/
│   ├── SMD_data/
│   └── UCR_Anomaly_FullData/
├── results/
│   ├── standard_ksweep_all_datasets.csv
│   └── singletons_cross_dataset.csv
├── paper/
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   └── figs/
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# (1) Per-detector singleton baselines
python code/singletons.py
# -> results/singletons_cross_dataset.csv

# (2) Full SHAP-σ pipeline (TELCO + 28 SMD machines + 100 UCR base series)
python code/shap_sigma_pipeline.py
# -> results/standard_ksweep_all_datasets.csv

# (3) Reproduce the three paper figures
python code/make_figures.py
# -> paper/figs/{fig_singletons,fig_ksweep,fig_jointpair_telco}.{pdf,png}
```

Both pipelines write directly to `results/`; figures land in `paper/figs/`.
Runtime on a single workstation: TELCO ≈ 2 min, SMD ≈ 25 min,
UCR ≈ 35 min (set sizes K_R = 2…8, three strategies each).

## Pipeline architecture (Sec. II of the paper)

Stage 1 — Label-free Mass-Volume proxy quality `q_d ∈ [0,1]`
(Eq. 1; three random low-dim subspaces, 2 000 uniform background samples).

Stage 2 — Joint Rashomon-set selection over all `C(K, K_R)` subsets;
three strategies (Quality, Diversity, Mixed). For `K_R = 2` Mixed reduces
to the closed-form joint pair score
`(q_i + q_j)/2 · (1 − ρ_{ij})` (Eq. 2).

Stage 3 — LightGBM surrogate per selected detector, TreeSHAP
attributions, detector-wise global std-normalization
(divides by `std(Φ_d^v) + 10^{-12}`).

Stage 4 — Cross-detector standard deviation `σ_{ij}` (Eq. 3),
per-feature 0.75-quantile threshold `τ_j` on validation, point-level
OR-aggregate `φ_i^∨ = ⋁_j 1[σ_{ij} > τ_j]`.

Stage 5 — Anchor κ-sigma flag `f^{(a)}` (κ = 3) combined with the
attribution-evidence gate:
`ŷ_i = f^{(a)}_i ∧ φ_i^∨` (Eq. 5 — canonical).

The pipeline also reports four alternative fusion variants
(OR+σ, AND+σ, OR, AND) and `make_figures.py` highlights the canonical
anchor+σ as the headline.

## Defaults

| Setting | Value |
|---|---|
| Detector pool | IForest, LOF, ECOD, COPOD, KNN, PCA, GMM, HBOS |
| Default config | `(K_R = 2, S = Mixed)` |
| Anchor flag | κ-sigma with κ = 3 |
| σ-filter quantile | q = 0.75 |
| SHAP surrogate | LightGBM, 200 trees, depth 6, lr 0.05 |
| MV-proxy surrogate | LightGBM, 80 trees, depth 4, lr 0.1 |

## Datasets

- **TELCO** — multivariate cellular KPIs from an operational mobile
  network (González-Vidal et al., 2023). Place under
  `data/TELCO_data/` and `data/TELCO_labels/` with the column header
  `TS1, …, TS12`.
- **SMD** — Server Machine Dataset (Su et al., 2019). 28 machines
  under `data/SMD_data/{train, test, test_label}/<machine>.txt`.
- **UCR Anomaly Archive** — single-anomaly univariate series
  (Wu & Keogh, 2021). 250 files under
  `data/UCR_Anomaly_FullData/`. The pipeline auto-deduplicates to
  100 base series by stripping the `DISTORTED`/`NOISE` prefix and
  keeping the first file per base name.

Data is not bundled. Use the upstream sources cited above.

## Reproducing the paper

```bash
cd paper
pdflatex main.tex && pdflatex main.tex   # twice for references
```

LaTeX requires `IEEEtran` plus `booktabs, balance, algorithm,
algorithmic, multirow, array, xcolor`.

## Citation

```bibtex
@inproceedings{Dar2026shapsigma,
  author    = {Dar, Ugur and Malacarne, Sara and Cavus, Mustafa and Van den Abeele, Jeriek},
  title     = {{SHAP-$\sigma$: A Label-Free Rashomon Pipeline for Multivariate Time-Series Anomaly Detection}},
  booktitle = {Proc. IEEE Int. Conf.},
  year      = {2026},
}
```

## Funding

This work has received funding from the European Union via the
OC4-2025-TES-02 call issued and implemented by the ENFIELD project,
under grant agreement No 101120657.
