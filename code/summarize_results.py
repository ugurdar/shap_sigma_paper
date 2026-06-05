"""
Summarise results CSVs into the numbers Table I (singletons) and Table II
(SHAP-sigma vs singleton) of the paper require.

Run after shap_sigma_pipeline.py and singletons.py have finished.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
KSWEEP_CSV     = os.path.join(RES, "standard_ksweep_all_datasets.csv")
SINGLETONS_CSV = os.path.join(RES, "singletons_cross_dataset.csv")


def fmt(x, ndp=3):
    if pd.isna(x):
        return "---"
    s = f"{x:.{ndp}f}"
    # Drop leading zero (LaTeX paper convention: .438 not 0.438)
    return s[1:] if s.startswith("0.") else s


def latex_table_ii_rows(ks, sg):
    """Produce the four LaTeX rows of Table II based on the canonical CSVs."""
    rows = []

    # TELCO Def. row
    sub_t = ks[(ks["dataset"] == "TELCO") & (ks["K"] == 2) & (ks["strategy"] == "Mixed")]
    if len(sub_t):
        f1, p = sub_t["F1"].iloc[0], sub_t["P"].iloc[0]
        sg_t = sg[sg["dataset"] == "TELCO"]
        best = sg_t.groupby("detector")["F1_per_KPI_macro"].mean().sort_values(ascending=False)
        sing_f1 = best.iloc[0]
        sing_det = best.index[0]
        rows.append(("TELCO (Def.)", sing_det, sing_f1, "(2,M)", f1, np.nan, p, "---"))

    # SMD Def. row + Best row
    sub_smd_def = ks[(ks["dataset"] == "SMD") & (ks["K"] == 2) & (ks["strategy"] == "Mixed")]
    if len(sub_smd_def):
        # Singleton baseline
        sg_smd = sg[sg["dataset"] == "SMD"]
        sg_means = sg_smd.groupby("detector")["F1_point_level"].mean().sort_values(ascending=False)
        sing_det = sg_means.index[0]
        sing_f1 = sg_means.iloc[0]
        # Default pipeline means
        f1 = sub_smd_def["F1"].mean()
        pa = sub_smd_def["PA_F1"].mean()
        p_ = sub_smd_def["P"].mean()
        # Wilcoxon vs singleton
        pipe_per_series = sub_smd_def.set_index("series")["F1"]
        sing_per_series = sg_smd[sg_smd["detector"] == sing_det].set_index("series")["F1_point_level"]
        common = pipe_per_series.index.intersection(sing_per_series.index)
        try:
            stat, pval = wilcoxon(pipe_per_series[common].values,
                                  sing_per_series[common].values,
                                  alternative="two-sided", zero_method="wilcox")
            pval_str = f"{pval:.2g}"
        except Exception:
            pval_str = "---"
        rows.append((f"SMD ($n{{=}}{len(sub_smd_def)}$, Def.)",
                     sing_det, sing_f1, "(2,M)", f1, pa, p_, pval_str))

        # Best (K, S) row
        smd_means = ks[ks["dataset"] == "SMD"].groupby(["K", "strategy"])["F1"].mean().sort_values(ascending=False)
        best_K, best_S = smd_means.index[0]
        sub_best = ks[(ks["dataset"] == "SMD") & (ks["K"] == best_K) & (ks["strategy"] == best_S)]
        f1b = sub_best["F1"].mean()
        pab = sub_best["PA_F1"].mean()
        pb = sub_best["P"].mean()
        s_short = {"Quality": "Q", "Diversity": "D", "Mixed": "M"}[best_S]
        rows.append((f"SMD ($n{{=}}{len(sub_best)}$, Best$^{{\\dagger}}$)",
                     sing_det, sing_f1, f"({best_K},\\text{{{s_short}}})", f1b, pab, pb, "---"))

    # UCR Def. row
    sub_ucr = ks[(ks["dataset"] == "UCR") & (ks["K"] == 2) & (ks["strategy"] == "Mixed")]
    if len(sub_ucr):
        sg_ucr = sg[sg["dataset"] == "UCR"]
        sg_means = sg_ucr.groupby("detector")["F1_point_level"].mean().sort_values(ascending=False)
        sing_det = sg_means.index[0]
        sing_f1 = sg_means.iloc[0]
        f1 = sub_ucr["F1"].mean()
        pa = sub_ucr["PA_F1"].mean()
        p_ = sub_ucr["P"].mean()
        n_ucr = len(sub_ucr)
        # Wilcoxon
        pipe_per_series = sub_ucr.set_index("series")["F1"]
        sing_per_series = sg_ucr[sg_ucr["detector"] == sing_det].set_index("series")["F1_point_level"]
        common = pipe_per_series.index.intersection(sing_per_series.index)
        try:
            stat, pval = wilcoxon(pipe_per_series[common].values,
                                  sing_per_series[common].values,
                                  alternative="two-sided", zero_method="wilcox")
            pval_str = f"{pval:.2g}"
        except Exception:
            pval_str = "---"
        rows.append((f"UCR ($n{{=}}{n_ucr}$, Def.)",
                     sing_det, sing_f1, "(2,M)", f1, pa, p_, pval_str))

    # Print LaTeX
    print("\n" + "=" * 70)
    print("LATEX ROWS for Table II")
    print("=" * 70)
    for cfg, det, sf1, ks_str, f1, pa, p_, pval in rows:
        line = f"{cfg:<32} & {det:<4} & {fmt(sf1)} & ${ks_str}$ & \\textbf{{{fmt(f1)}}} & {fmt(pa)} & {fmt(p_)} & {pval} \\\\"
        print(line)


def main():
    ks = pd.read_csv(KSWEEP_CSV)
    sg = pd.read_csv(SINGLETONS_CSV)

    print("=" * 70)
    print("SINGLETON BASELINES (Fig. 1 + Table II singleton column)")
    print("=" * 70)
    for ds in ["TELCO", "SMD", "UCR"]:
        sub = sg[sg["dataset"] == ds]
        col = "F1_per_KPI_macro" if ds == "TELCO" else "F1_point_level"
        g = sub.groupby("detector")[col].mean().sort_values(ascending=False)
        best = g.index[0]
        n = sub["series"].nunique()
        print(f"\n{ds} (n={n}): best singleton = {best}, mean F1 = {g.iloc[0]:.4f}")
        print(g.round(4).to_string())

    print("\n" + "=" * 70)
    print("SHAP-SIGMA Default (K_R=2, Mixed) per dataset")
    print("=" * 70)
    for ds in ["TELCO", "SMD", "UCR"]:
        sub = ks[(ks["dataset"] == ds) & (ks["K"] == 2) & (ks["strategy"] == "Mixed")]
        n = sub["series"].nunique()
        print(f"\n{ds} (n={n}, K=2, Mixed):")
        for col in ["F1", "P", "R", "PA_F1"]:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    print(f"  mean {col}: {vals.mean():.4f}  std: {vals.std():.4f}")
        # Anchor distribution
        anchors = sub["anchor"].value_counts()
        print(f"  Selected anchors: {dict(anchors)}")
        # Best variant distribution
        bv = sub["best_variant"].value_counts()
        total = bv.sum()
        print(f"  Best fusion variant: " +
              ", ".join(f"{v}={c}/{total} ({100*c/total:.0f}%)" for v, c in bv.items()))

    print("\n" + "=" * 70)
    print("Per-dataset POST-HOC best (K_R, Strategy) — Tablo II Best^\\dagger row")
    print("=" * 70)
    for ds in ["TELCO", "SMD", "UCR"]:
        sub = ks[ks["dataset"] == ds]
        means = sub.groupby(["K", "strategy"])["F1"].mean().sort_values(ascending=False)
        print(f"\n{ds}: top 3 (K, strategy) by mean F1:")
        print(means.head(3).round(4).to_string())
        # Best
        best_K, best_S = means.index[0]
        best_sub = sub[(sub["K"] == best_K) & (sub["strategy"] == best_S)]
        print(f"  Best (K={best_K}, S={best_S}):")
        for col in ["F1", "P", "R", "PA_F1"]:
            if col in best_sub.columns:
                v = best_sub[col].dropna()
                if len(v) > 0:
                    print(f"    mean {col}: {v.mean():.4f}")

    print("\n" + "=" * 70)
    print("Wilcoxon: SHAP-sigma default vs uniformly-best singleton (per series)")
    print("=" * 70)
    for ds in ["SMD", "UCR"]:
        col_p = "F1_point_level"
        sg_ds = sg[sg["dataset"] == ds]
        # uniformly-best singleton determined by mean F1 across series
        best_det = sg_ds.groupby("detector")[col_p].mean().idxmax()
        sg_best = sg_ds[sg_ds["detector"] == best_det].set_index("series")[col_p]
        pipe = ks[(ks["dataset"] == ds) & (ks["K"] == 2) & (ks["strategy"] == "Mixed")]
        pipe = pipe.set_index("series")["F1"]
        # Aligned series
        common = sg_best.index.intersection(pipe.index)
        a = sg_best.loc[common].values
        b = pipe.loc[common].values
        # Wilcoxon two-sided on diff
        diff = b - a
        nz = diff[diff != 0]
        if len(nz) > 0:
            try:
                stat, pval = wilcoxon(b, a, alternative="two-sided", zero_method="wilcox")
                # Rank-biserial effect size
                wins = (b > a).sum()
                losses = (b < a).sum()
                rb = (wins - losses) / (wins + losses) if (wins + losses) > 0 else 0
                print(f"\n{ds}: best singleton {best_det} mean F1 = {a.mean():.4f}, "
                      f"pipeline mean F1 = {b.mean():.4f}")
                print(f"  Wilcoxon two-sided: p = {pval:.4g}, rank-biserial r = {rb:+.3f}")
                print(f"  wins/losses/ties = {wins}/{losses}/{len(common) - wins - losses}")
            except Exception as e:
                print(f"{ds}: wilcoxon failed: {e}")
        else:
            print(f"{ds}: no differences, no test.")


if __name__ == "__main__":
    main()
