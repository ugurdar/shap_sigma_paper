"""Gate-signal figure: SHAP-sigma vs SHAP-mean across K_R, three panels
(F1 / Precision / Recall). Reads results/telco_gate_variants.csv (from
4_gate_ablation.py). Writes paper/figs/fig_gate_variants.{pdf,png}.
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":7.5,"font.family":"serif","axes.labelsize":8})

RES="../results"; OUT="../paper/figs"
df=pd.read_csv(f"{RES}/telco_gate_variants.csv").sort_values("K_R")
KR=df["K_R"].tolist()
sig ={"F1":df["sig_F1"].tolist(), "P":df["sig_P"].tolist(), "R":df["sig_R"].tolist()}
mean={"F1":df["mean_F1"].tolist(),"P":df["mean_P"].tolist(),"R":df["mean_R"].tolist()}
SIG="#1f4e79"; MEAN="#c0792b"

fig,axes=plt.subplots(1,3,figsize=(7.0,2.35))
for ax,metric,ttl in zip(axes,["F1","P","R"],["$F_1$","Precision","Recall"]):
    l1,=ax.plot(KR,sig[metric],"-o",color=SIG,ms=4,lw=1.4,label=r"SHAP-$\sigma$")
    l2,=ax.plot(KR,mean[metric],"--s",color=MEAN,ms=3.5,lw=1.4,label="SHAP-mean")
    ax.set_title(ttl,fontsize=8.5); ax.set_xlabel("$K_R$")
    ax.grid(True,color="0.9",lw=0.5); ax.set_xticks(KR)
    for sp in ["top","right"]: ax.spines[sp].set_linewidth(0.4)
    ax.tick_params(length=2.5)
fig.legend([l1,l2],[r"SHAP-$\sigma$","SHAP-mean"],loc="upper center",ncol=2,
           frameon=False,fontsize=8,bbox_to_anchor=(0.5,1.005),columnspacing=2.5)
fig.tight_layout(rect=[0,0,1,0.90],pad=0.6)
if os.path.isdir(OUT):
    fig.savefig(f"{OUT}/fig_gate_variants.pdf"); fig.savefig(f"{OUT}/fig_gate_variants.png",dpi=220)
plt.close(); print("saved fig_gate_variants")
