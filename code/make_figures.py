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
    sing = pd.read_csv(os.path.join(RES, "singletons_cross_dataset.csv"))
    DETS = ["ECOD", "LOF", "GMM", "KNN", "COPOD", "IF", "PCA", "HBOS"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95),
                             gridspec_kw=dict(wspace=0.30, top=0.78, bottom=0.30,
                                              left=0.07, right=0.99))
    for i, ds in enumerate(["TELCO", "SMD", "UCR"]):
        ax = axes[i]
        sub = sing[sing["dataset"] == ds]
        col = "F1_per_KPI_macro" if ds == "TELCO" else "F1_point_level"
        g = sub.groupby("detector")[col].mean().reindex(DETS)
        best_det = g.idxmax()
        colors = [COL["best"] if d == best_det else COL["neutral"] for d in DETS]
        ax.bar(range(len(DETS)), g.values, color=colors,
               edgecolor="black", linewidth=0.5, zorder=3, width=0.7)
        best_idx = DETS.index(best_det)
        ax.text(best_idx, g[best_det] + g.max() * 0.04, f"{g[best_det]:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=COL["best"])
        ax.set_xticks(range(len(DETS)))
        ax.set_xticklabels(DETS, rotation=45, ha="right", fontsize=7)
        if i == 0:
            ax.set_ylabel("Singleton $F_1$")
        n = sub["series"].nunique()
        lab = "per-KPI macro" if ds == "TELCO" else "point-level"
        ax.set_title(f"{ds} (n={n}, {lab})")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", which="both", bottom=False)
        ax.tick_params(axis="y", direction="in", length=2.5, pad=2)
        ax.set_ylim(0, g.max() * 1.20)
        for sp in ["top", "right"]:
            ax.spines[sp].set_linewidth(0.4)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COL["best"], ec="black", lw=0.5,
                      label="Best singleton per dataset"),
        plt.Rectangle((0, 0), 1, 1, color=COL["neutral"], ec="black", lw=0.5,
                      label="Other detectors"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.00), frameon=False,
               handletextpad=0.5, columnspacing=1.5)
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
    mixed_top = k2.sort_values("q_times_div", ascending=False).iloc[0]
    qual_top  = k2.sort_values("quality", ascending=False).iloc[0]

    fig = plt.figure(figsize=(3.4, 2.5))
    ax = fig.add_axes([0.18, 0.32, 0.78, 0.58])
    ax.scatter(k2["q_times_div"], k2["F1"], s=30, alpha=0.7,
               color=COL["neutral"], edgecolor="0.25", linewidth=0.4, zorder=2)
    ax.scatter(mixed_top["q_times_div"], mixed_top["F1"], s=110,
               color=COL["best"], marker="*", edgecolor="black",
               linewidth=0.7, zorder=5)
    ax.scatter(qual_top["q_times_div"], qual_top["F1"], s=60,
               color=COL["Quality"], marker="D", edgecolor="black",
               linewidth=0.6, zorder=4)
    ax.set_xlabel(r"Joint score $\frac{q_i+q_j}{2}\,(1-\rho_{ij})$", fontsize=8.5)
    ax.set_ylabel("Pipeline $F_1$ (per-KPI macro)", fontsize=8.5)
    ax.set_title("TELCO: all 28 candidate pairs (K_R=2)", fontsize=9)
    ax.tick_params(direction="in", length=2.5, pad=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_linewidth(0.4)

    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COL["neutral"], markeredgecolor="0.25",
               markersize=6, label="all 28 pairs"),
        Line2D([0], [0], marker="D", color="none",
               markerfacecolor=COL["Quality"], markeredgecolor="black",
               markersize=6, label=f"Quality pick: {qual_top['set']}"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor=COL["best"], markeredgecolor="black",
               markersize=9, label=f"Mixed pick = oracle: {mixed_top['set']}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1,
               bbox_to_anchor=(0.55, 0.00), frameon=False, handletextpad=0.5)
    plt.savefig(os.path.join(OUT, "fig_jointpair_telco.pdf"))
    plt.savefig(os.path.join(OUT, "fig_jointpair_telco.png"), dpi=220)
    plt.close()


if __name__ == "__main__":
    fig_singletons();        print("Saved fig_singletons")
    fig_ksweep();            print("Saved fig_ksweep")
    fig_jointpair_telco();   print("Saved fig_jointpair_telco")
