"""Macro / Micro / Union aggregation of the default K_R=2 operating point
(ECOD+HBOS, anchor ECOD) -- the last three rows of Table III and the
SHAP-sigma row of the headline Table IV.

  macro  unweighted mean of per-KPI P/R/F1
  micro  pool TP/FP/FN over all KPI entries, then P/R/F1
  union  per-timestamp OR across KPIs (point level): pred_i = any KPI
         flagged, true_i = any KPI anomalous

Outputs: results/telco_aggregations.csv
"""
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
import shap, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")

BASE="../data"; RES="../results"; SEED=42; K_SIGMA=3.0; Q=0.75; FEAT=[f"TS{i}" for i in range(1,13)]
df_tr=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_train.csv"); df_te=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_test.csv")
Xall=df_tr[FEAT].values.astype(np.float32); Xte=df_te[FEAT].values.astype(np.float32)
split=int(len(Xall)*0.8); Xtr,Xv=Xall[:split],Xall[split:]
sc=StandardScaler().fit(Xtr); Xtr_s,Xv_s,Xte_s=sc.transform(Xtr),sc.transform(Xv),sc.transform(Xte)
y=pd.read_csv(f"{BASE}/TELCO_labels/TELCO_labels_test.csv")[FEAT].values.astype(int)
n_b=int(np.log2(len(Xtr_s))+1)
dets={"ECOD":ECOD(),"HBOS":HBOS(n_bins=n_b)}
s_tr,s_v,s_t={},{},{}
for nm,m in dets.items():
    m.fit(Xtr_s); s_tr[nm]=m.decision_function(Xtr_s); s_v[nm]=m.decision_function(Xv_s); s_t[nm]=m.decision_function(Xte_s)
SHv,SHt={},{}
for d in dets:
    g=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,verbose=-1)
    g.fit(Xtr_s,s_tr[d]); ex=shap.TreeExplainer(g); SHv[d]=ex.shap_values(Xv_s); SHt[d]=ex.shap_values(Xte_s)
    nmv=SHv[d].std()+1e-12; SHv[d]/=nmv; SHt[d]/=nmv
anchor="ECOD"; fa=(s_t[anchor]>s_v[anchor].mean()+K_SIGMA*s_v[anchor].std()).astype(int)
SV=np.stack([SHv[d] for d in dets]); ST=np.stack([SHt[d] for d in dets])
sig_v=SV.std(0); sig_t=ST.std(0)
tau=np.array([np.quantile(sig_v[:,j],Q) for j in range(12)])
filt=np.stack([(sig_t[:,j]>tau[j]).astype(int) for j in range(12)],axis=1)
pred=(np.repeat(fa[:,None],12,axis=1)*filt).astype(int)

# macro
Fma=np.mean([f1_score(y[:,i],pred[:,i],zero_division=0) for i in range(12)])
Pma=np.mean([precision_score(y[:,i],pred[:,i],zero_division=0) for i in range(12)])
Rma=np.mean([recall_score(y[:,i],pred[:,i],zero_division=0) for i in range(12)])
# micro (pool all KPI entries)
TP=int(((y==1)&(pred==1)).sum()); FP=int(((y==0)&(pred==1)).sum()); FN=int(((y==1)&(pred==0)).sum())
Pmi=TP/(TP+FP); Rmi=TP/(TP+FN); Fmi=2*Pmi*Rmi/(Pmi+Rmi)
# union (point level)
pp=pred.any(1).astype(int); yp=y.any(1).astype(int)
Pu=precision_score(yp,pp,zero_division=0); Ru=recall_score(yp,pp,zero_division=0); Fu=f1_score(yp,pp,zero_division=0)

out=pd.DataFrame([
    {"agg":"macro","P":round(Pma,3),"R":round(Rma,3),"F1":round(Fma,3),"true":int(y.sum()),"pred":int(pred.sum())},
    {"agg":"micro","P":round(Pmi,3),"R":round(Rmi,3),"F1":round(Fmi,3),"true":int(y.sum()),"pred":int(pred.sum())},
    {"agg":"union","P":round(Pu,3), "R":round(Ru,3), "F1":round(Fu,3), "true":int(yp.sum()),"pred":int(pp.sum())},
])
out.to_csv(f"{RES}/telco_aggregations.csv",index=False)
print(out.to_string(index=False)); print("saved telco_aggregations.csv")
