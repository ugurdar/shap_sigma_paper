"""Joint-pair landscape figure (fidelity-filtered pipeline).
Reads results/telco_B_pair_landscape.csv (from 2_filter_select.py): all 45
K_R=2 pairs as joint score vs pipeline F1. Pairs containing a low-fidelity
detector (LOF or KNN, R^2_val<0.75) are greyed out as removed by the
filter; among the retained pairs the joint-score maximiser is ECOD+HBOS
(green star), the default operating point.
Writes paper/figs/fig_jointpair_telco.{pdf,png}.
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES="../results"; OUT="../paper/figs"
DROPPED={"LOF","KNN"}   # R^2_val < 0.75
plt.rcParams.update({"font.size":7,"axes.labelsize":7,"xtick.labelsize":6.5,
                     "ytick.labelsize":6.5,"font.family":"serif"})
df=pd.read_csv(f"{RES}/telco_B_pair_landscape.csv")
df["dropped"]=df["pair"].apply(lambda p: any(d in p.split("+") for d in DROPPED))
surv=df[~df["dropped"]]; rem=df[df["dropped"]]
selected=surv.sort_values("joint_pair_score",ascending=False).iloc[0]   # ECOD+HBOS

fig=plt.figure(figsize=(3.4,2.7)); ax=fig.add_axes([0.135,0.17,0.84,0.80])
ax.scatter(rem["joint_pair_score"],rem["F1"],s=26,marker="x",color="0.7",linewidth=0.7,zorder=2)
ax.scatter(surv["joint_pair_score"],surv["F1"],s=26,alpha=0.8,color="#4C72B0",
           edgecolor="0.2",linewidth=0.4,zorder=3)
ax.scatter(selected["joint_pair_score"],selected["F1"],s=180,color="#27ae60",marker="*",
           edgecolor="black",linewidth=0.8,zorder=6)
ax.annotate("selected pair:\nECOD+HBOS",(selected["joint_pair_score"],selected["F1"]),fontsize=6.3,
            xytext=(-52,2),textcoords="offset points",ha="right",va="center",color="#1e7a3a",
            arrowprops=dict(arrowstyle="->",color="#27ae60",lw=0.8,shrinkA=0,shrinkB=6))
ax.set_xlabel("Joint pair score",fontsize=8); ax.set_ylabel("$F_1$",fontsize=8)
ax.tick_params(direction="in",length=2.5,pad=2)
for sp in ["top","right"]: ax.spines[sp].set_linewidth(0.4)
xmin,xmax=df["joint_pair_score"].min(),df["joint_pair_score"].max()
ax.set_xlim(xmin-0.04,xmax+0.06); ax.set_ylim(-0.02,df["F1"].max()+0.07)
handles=[
    Line2D([0],[0],marker="x",color="0.7",markersize=6,linestyle="none",label="removed ($R^2_{\\mathrm{val}}{<}0.75$)"),
    Line2D([0],[0],marker="o",color="none",markerfacecolor="#4C72B0",markeredgecolor="0.2",markersize=6,label="retained ($R^2_{\\mathrm{val}}{\\geq}0.75$)"),
    Line2D([0],[0],marker="*",color="none",markerfacecolor="#27ae60",markeredgecolor="black",markersize=11,label="selected (ECOD+HBOS)"),
]
ax.legend(handles=handles,loc="center right",bbox_to_anchor=(0.995,0.32),frameon=True,framealpha=0.3,
          edgecolor="0.85",handletextpad=0.25,borderpad=0.35,labelspacing=0.3,markerscale=0.9,fontsize=5.2)
if os.path.isdir(OUT):
    plt.savefig(f"{OUT}/fig_jointpair_telco.pdf"); plt.savefig(f"{OUT}/fig_jointpair_telco.png",dpi=220)
plt.close()
print(f"saved fig_jointpair_telco | selected={selected['pair']} jp={selected.joint_pair_score} F1={selected.F1}")
