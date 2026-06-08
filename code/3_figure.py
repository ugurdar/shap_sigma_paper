"""Fig. 1 -- joint pair score vs. pipeline F1 for all K_R=2 pairs.
Reads results/telco_B_pair_landscape.csv (from 2_pipeline.py).
Green star = closed-form joint-score maximiser (LOF+ECOD).
Red diamond = label-aware F1-oracle pair (ECOD+HBOS) -- they do NOT coincide.
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES="../results"
OUT="../paper/figs"
plt.rcParams.update({"font.size":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,
                     "ytick.labelsize":7.5,"font.family":"serif"})
df=pd.read_csv(f"{RES}/telco_B_pair_landscape.csv")
jmax=df.sort_values("joint_pair_score",ascending=False).iloc[0]
oracle=df.sort_values("F1",ascending=False).iloc[0]

fig=plt.figure(figsize=(3.4,2.8)); ax=fig.add_axes([0.18,0.42,0.78,0.50])
ax.scatter(df["joint_pair_score"],df["F1"],s=30,alpha=0.7,color="0.6",
           edgecolor="0.25",linewidth=0.4,zorder=2)
# F1-oracle (label-aware) -- red diamond
ax.scatter(oracle["joint_pair_score"],oracle["F1"],s=90,color="#c0392b",marker="D",
           edgecolor="black",linewidth=0.7,zorder=5)
# closed-form joint-score maximiser -- green star
ax.scatter(jmax["joint_pair_score"],jmax["F1"],s=170,color="#27ae60",marker="*",
           edgecolor="black",linewidth=0.8,zorder=6)
ax.annotate(jmax["pair"],(jmax["joint_pair_score"],jmax["F1"]),fontsize=6.5,
            xytext=(-2,-9),textcoords="offset points",ha="right")
ax.annotate(oracle["pair"],(oracle["joint_pair_score"],oracle["F1"]),fontsize=6.5,
            xytext=(6,2),textcoords="offset points")
ax.set_xlabel(r"Joint pair score $\frac{q_i+q_j}{2}\,(1-\rho_{ij})$",fontsize=8.5)
ax.set_ylabel("Pipeline $F_1$ (per-KPI macro)",fontsize=8.5)
ax.set_title(r"TELCO: all 45 candidate pairs ($K_R{=}2$)",fontsize=9)
ax.tick_params(direction="in",length=2.5,pad=2)
for sp in ["top","right"]: ax.spines[sp].set_linewidth(0.4)
handles=[
    Line2D([0],[0],marker="o",color="none",markerfacecolor="0.6",markeredgecolor="0.25",markersize=6,label="all 45 pairs"),
    Line2D([0],[0],marker="*",color="none",markerfacecolor="#27ae60",markeredgecolor="black",markersize=11,label="joint-score max (LOF+ECOD)"),
    Line2D([0],[0],marker="D",color="none",markerfacecolor="#c0392b",markeredgecolor="black",markersize=7,label="$F_1$-oracle (ECOD+HBOS)"),
]
fig.legend(handles=handles,loc="lower center",ncol=1,bbox_to_anchor=(0.55,0.02),
           frameon=False,handletextpad=0.4,fontsize=6.8)
plt.savefig(f"{OUT}/fig_jointpair_telco.pdf"); plt.savefig(f"{OUT}/fig_jointpair_telco.png",dpi=220); plt.close()
print(f"saved fig_jointpair_telco | joint-max={jmax['pair']} F1={jmax.F1} | oracle={oracle['pair']} F1={oracle.F1}")
