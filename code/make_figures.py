"""
Reproduce the three figures in the paper:
  paper/figs/fig_singletons.pdf      (Fig. 1)
  paper/figs/fig_ksweep.pdf          (Fig. 2)
  paper/figs/fig_jointpair_telco.pdf (Fig. 3)

Inputs:
  results/standard_ksweep_all_datasets.csv  (from shap_sigma_pipeline.py)
  results/singletons_cross_dataset.csv      (from singletons.py)
  results/telco_all_k_combinations.csv      (TELCO K=2 sweep for joint-pair plot)

Output:
  paper/figs/*.pdf and *.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":           8,
    "axes.labelsize":      8.5,
    "axes.titlesize":      9,
    "xtick.labelsize":     7.5,
    "ytick.labelsize":     7.5,
    "legend.fontsize":     7.5,
    "legend.framealpha":   1.0,
    "legend.edgecolor":    "0.3",
    "legend.fancybox":     False,
    "lines.linewidth":     1.3,
    "lines.markersize":    4.5,
    "axes.linewidth":      0.6,
    "axes.edgecolor":      "0.2",
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linestyle":      ":",
    "grid.linewidth":      0.5,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "paper", "figs")
RES  = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

# Wong colour-blind-safe palette
COL  = {"Quality": "#0072B2", "Diversity": "#D55E00", "Mixed": "#009E73",
        "best":    "#009E73", "neutral":   "#BBBBBB"}
MARK = {"Quality": "o", "Diversity": "s", "Mixed": "D"}


# ============== Fig 1: Singletons ==============

def fig_singletons():
    """TELCO-only singleton bar chart."""
    sing = pd.read_csv(os.path.join(RES, "singletons_cross_dataset.csv"))
    DETS = ["ECOD", "LOF", "GMM", "KNN", "COPOD", "IF", "PCA", "HBOS"]

    fig, ax = plt.subplots(figsize=(3.4, 2.0),
                            gridspec_kw=dict(top=0.92, bottom=0.30,
                                              left=0.18, right=0.97))
    sub = sing[sing["dataset"] == "TELCO"]
    g = sub.groupby("detector")["F1_per_KPI_macro"].mean().reindex(DETS)
    best_det = g.idxmax()
    colors = [COL["best"] if d == best_det else COL["neutral"] for d in DETS]
    ax.bar(range(len(DETS)), g.values, color=colors,
            edgecolor="black", linewidth=0.5, zorder=3, width=0.7)

    # SHAP-σ reference line at F1=0.440
    ax.axhline(0.440, color="#D55E00", lw=1.0, ls="--", zorder=4,
                label=r"SHAP-$\sigma$ ($K_R{=}2$)")
    ax.text(len(DETS)-0.5, 0.445, "SHAP-$\\sigma$ 0.440", ha="right", va="bottom",
            fontsize=7, color="#D55E00")

    best_idx = DETS.index(best_det)
    ax.text(best_idx, g[best_det] + 0.015, f"{g[best_det]:.3f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color=COL["best"])
    ax.set_xticks(range(len(DETS)))
    ax.set_xticklabels(DETS, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Per-KPI macro $F_1$", fontsize=8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.tick_params(axis="y", direction="in", length=2.5, pad=2)
    ax.set_ylim(0, 0.50)
    for sp in ["top", "right"]:
        ax.spines[sp].set_linewidth(0.4)

    plt.savefig(os.path.join(OUT, "fig_singletons.pdf"))
    plt.savefig(os.path.join(OUT, "fig_singletons.png"), dpi=220)
    plt.close()


# ============== Fig 2: K-sweep ==============

def fig_ksweep():
    df = pd.read_csv(os.path.join(RES, "standard_ksweep_all_datasets.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), sharey=False,
                             gridspec_kw=dict(wspace=0.30, top=0.78, bottom=0.22,
                                              left=0.07, right=0.99))
    for i, ds in enumerate(["TELCO", "SMD", "UCR"]):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        for strat in ["Quality", "Diversity", "Mixed"]:
            g = sub[sub["strategy"] == strat].groupby("K")["F1"].mean().reset_index()
            ax.plot(g["K"], g["F1"], marker=MARK[strat], color=COL[strat],
                    markerfacecolor="white", markeredgewidth=1.2,
                    markeredgecolor=COL[strat], clip_on=False)
        ax.set_xlabel(r"Rashomon set size $K_R$")
        if i == 0:
            ax.set_ylabel("Mean macro $F_1$")
        n = sub["series"].nunique()
        ax.set_title(f"{ds} (n={n})")
        ax.set_xticks([2, 3, 4, 5, 6, 7, 8])
        ax.tick_params(direction="in", length=2.5, pad=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_linewidth(0.4)

    handles = [
        Line2D([0], [0], color=COL[s], marker=MARK[s],
               markerfacecolor="white", markeredgewidth=1.2,
               markeredgecolor=COL[s], lw=1.3,
               label=f"Strategy {s[0]} ({s})")
        for s in ["Quality", "Diversity", "Mixed"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.00), frameon=False,
               handletextpad=0.4, columnspacing=1.3)
    plt.savefig(os.path.join(OUT, "fig_ksweep.pdf"))
    plt.savefig(os.path.join(OUT, "fig_ksweep.png"), dpi=220)
    plt.close()


# ============== Fig 3: TELCO joint-pair scatter ==============

def fig_jointpair_telco():
    df2 = pd.read_csv(os.path.join(RES, "telco_all_k_combinations.csv"))
    k2 = df2[df2["K"] == 2].copy()

    # Empirical F1 oracle (highest F1 among 28 pairs)
    oracle = k2.sort_values("F1", ascending=False).iloc[0]
    # Joint-score top-3 (Mixed-strategy candidates by joint score Eq. 2)
    top_joint = k2.sort_values("q_times_div", ascending=False).head(3)

    fig = plt.figure(figsize=(3.4, 2.8))
    ax = fig.add_axes([0.18, 0.42, 0.78, 0.50])
    ax.scatter(k2["q_times_div"], k2["F1"], s=30, alpha=0.7,
               color=COL["neutral"], edgecolor="0.25", linewidth=0.4, zorder=2)
    # Highlight top-3 joint-score candidates (smaller markers, neutral)
    for _, r in top_joint.iloc[1:3].iterrows():
        ax.scatter(r["q_times_div"], r["F1"], s=55,
                   color="white", marker="o", edgecolor=COL["Quality"],
                   linewidth=1.3, zorder=4)
    # Joint-score maximiser = empirical F1 oracle (coincide on TELCO)
    ax.scatter(oracle["q_times_div"], oracle["F1"], s=140,
               color=COL["best"], marker="*", edgecolor="black",
               linewidth=0.8, zorder=6)

    ax.set_xlabel(r"Joint score $\frac{q_i+q_j}{2}\,(1-\rho_{ij})$", fontsize=8.5)
    ax.set_ylabel("Pipeline $F_1$ (per-KPI macro)", fontsize=8.5)
    ax.set_title("TELCO: all 28 candidate pairs ($K_R=2$)", fontsize=9)
    ax.tick_params(direction="in", length=2.5, pad=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_linewidth(0.4)

    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COL["neutral"], markeredgecolor="0.25",
               markersize=6, label="all 28 candidate pairs"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="white", markeredgecolor=COL["Quality"],
               markersize=7, markeredgewidth=1.3,
               label=f"runners-up by joint score"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor=COL["best"], markeredgecolor="black",
               markersize=11, label=f"joint-score max = oracle: {oracle['set']}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1,
               bbox_to_anchor=(0.55, -0.02), frameon=False,
               handletextpad=0.5, fontsize=7.5)
    plt.savefig(os.path.join(OUT, "fig_jointpair_telco.pdf"))
    plt.savefig(os.path.join(OUT, "fig_jointpair_telco.png"), dpi=220)
    plt.close()


if __name__ == "__main__":
    fig_singletons();        print("Saved fig_singletons")
    fig_ksweep();            print("Saved fig_ksweep")
    fig_jointpair_telco();   print("Saved fig_jointpair_telco")
