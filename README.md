# SHAP-σ: A Label-Free Rashomon Pipeline for Multivariate Time-Series Anomaly Detection

Reference implementation of the conference paper: a **label-free** anomaly
detection pipeline with **KPI-level localization**, evaluated as a
**TELCO-only** study over a self-tuned **K = 10** detector pool. Every
table and figure in the paper is reproduced by the numbered scripts in
`code/`, run in order.

## Method in one paragraph

A heterogeneous pool of 10 detectors is **self-tuned without labels**.
Each detector gets a label-free quality `q_d` from a Mass–Volume proxy
(Stage 1). A LightGBM **surrogate** is fit per detector and only those
with validation fidelity `R²_val ≥ 0.75` are kept (Stage 3 filter — on
TELCO this drops **LOF** at `R²=0.25` and **KNN** at `0.63`). A compact
**Rashomon set** is then chosen from the filtered pool by a **Mixed**
quality×diversity criterion (Stage 2); the highest-quality member is the
**anchor**. The anchor raises a 3-σ timestamp flag, and the flag is kept
for a KPI only where the **cross-detector standard deviation of SHAP
attributions** (the **SHAP-σ** gate, Stage 4–5) exceeds a validation
threshold. The set size `K_R` is a hyperparameter; we report the full
sweep and use the **`K_R = 2` pair ECOD+HBOS** (anchor ECOD) as the
default operating point. Choosing `K_R` label-free is left to future work.

Headline on TELCO (per-KPI macro): **F1 = 0.442, P = 0.647, R = 0.374**.

## Repository layout

```
.
├── code/
│   ├── 1_singletons.py        # Table I   — 10 self-tuned singletons
│   ├── 2_filter_select.py     # Tables II & III + Fig. (jointpair) data
│   ├── 3_aggregations.py      # Table III macro/micro/union + Table IV row
│   ├── 4_gate_ablation.py     # gate-variants data (SHAP-σ vs SHAP-mean)
│   ├── 5_fig_jointpair.py     # Fig. joint-pair landscape
│   └── 6_fig_gate_variants.py # Fig. SHAP-σ vs SHAP-mean across K_R
├── data/                      # not tracked; see "Data" below
│   ├── TELCO_data/
│   └── TELCO_labels/
├── results/                   # one CSV per table/figure (see below)
├── paper/                     # main.tex, main.pdf, references.bib, figs/
└── requirements.txt
```

## What each script produces

| Script | Paper artifact | Output in `results/` |
|---|---|---|
| `1_singletons.py`      | Table I (10 self-tuned singletons)          | `telco_tuned_singleton.csv` |
| `2_filter_select.py`   | Table II (K_R sweep), Table III (per-KPI K_R=2) | `telco_r075_mixed_sweep.csv`, `telco_r075_best_perkpi.csv`, `telco_B_pair_landscape.csv` |
| `3_aggregations.py`    | Table III macro/micro/union, Table IV (SHAP-σ row) | `telco_aggregations.csv` |
| `4_gate_ablation.py`   | gate-variants table & figure (SHAP-σ vs SHAP-mean) | `telco_gate_variants.csv` |
| `5_fig_jointpair.py`   | Fig. joint-pair landscape                   | `paper/figs/fig_jointpair_telco.{pdf,png}` |
| `6_fig_gate_variants.py` | Fig. SHAP-σ vs SHAP-mean across K_R        | `paper/figs/fig_gate_variants.{pdf,png}` |

The two figure scripts only read the CSVs above, so they are fast; rerun
them alone to restyle a figure without recomputing the pipeline.

## Quick start (reproduces the paper)

```bash
pip install -r requirements.txt
cd code

python 1_singletons.py        # Table I
python 2_filter_select.py     # Tables II/III + jointpair data
python 3_aggregations.py      # macro/micro/union (Tables III/IV)
python 4_gate_ablation.py     # SHAP-σ vs SHAP-mean sweep
python 5_fig_jointpair.py     # Fig. joint-pair landscape
python 6_fig_gate_variants.py # Fig. gate variants
```

All scripts read `../data` and write `../results` (and figures into
`../paper/figs`), so run them from `code/`. The pipeline scripts
(`1`, `2`, `4`) each take ≈ 3–5 min — the AutoEncoder fit and TreeSHAP
over 10 surrogates dominate. `SEED = 42`, validation quantile `q = 0.75`,
anchor threshold `κ = 3`, fidelity threshold `R²_val = 0.75` throughout.

On macOS, run with single-threaded BLAS to keep pyod/LightGBM stable:

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python 2_filter_select.py
```

## Data

`data/` is **not** tracked. Provide the TELCO benchmark as:

```
data/TELCO_data/TELCO_data_{train,test}.csv     # 12 KPI columns TS1..TS12
data/TELCO_labels/TELCO_labels_test.csv         # per-KPI 0/1 labels
```

Scripts use an 80/20 train/validation split of the training file; the
validation split calibrates every threshold label-free. Test labels are
used only for the final per-KPI evaluation.

## Paper

`paper/main.tex` builds the conference manuscript; `paper/main.pdf` is the
compiled version. Tables are inline; the two figures live in `paper/figs/`
and are produced by `5_fig_jointpair.py` and `6_fig_gate_variants.py`.

## Citation

If you use this code, please cite the SHAP-σ paper (see `paper/`). Code
and data pointers: <https://github.com/ugurdar/shap_sigma_paper>.

## Funding

This work is supported by the *Rashomon-based Uncertainty-Aware Explainable AI for Time Series* project, funded by the European Union via grant OC4-2025-TES-02 issued and implemented by the ENFIELD project under grant agreement No.~101120657.
