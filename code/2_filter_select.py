"""Stage 1-4 of the SHAP-sigma pipeline, label-free end to end.

  1. Self-tune the K=10 detector pool and score train/val/test.
  2. Stage 1 quality q_d via the Mass-Volume proxy (Eq. 1).
  3. Stage 3 surrogate-fidelity filter: fit a LightGBM surrogate per
     detector and KEEP only those with validation R^2 >= 0.75
     (drops LOF and KNN on TELCO); their attributions are unreliable.
  4. Stage 2 Mixed joint-subset selection over the filtered pool, swept
     over the set size K_R (Table II). Selection uses only quality and
     diversity -- no labels.
  5. Stage 4-5 SHAP-sigma gate and anchor flag; the per-KPI breakdown of
     the default K_R=2 operating point ECOD+HBOS (Table III).

K_R is a hyperparameter; we report the whole sweep and use K_R=2 as the
default operating point. Choosing K_R label-free is left to future work.

Outputs (results/):
  telco_r075_mixed_sweep.csv   Table II  (K_R sensitivity sweep)
  telco_r075_best_perkpi.csv   Table III (per-KPI, default K_R=2 pair)
  telco_B_pair_landscape.csv   Fig. (all 45 K_R=2 pairs: joint score vs F1)
"""
import itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as SK_PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture as SK_GMM
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from pyod.models.pca import PCA as PCA_AD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.gmm import GMM as PYOD_GMM
from pyod.models.hbos import HBOS
import shap, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")

BASE="../data"; RES="../results"; SEED=42; K_SIGMA=3.0; Q=0.75; R2_MIN=0.75
MV_N_SUBSPACE=3; MV_N_UNIF=2000; LGB_TREES_MV=80
FEAT=[f"TS{i}" for i in range(1,13)]
df_tr=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_train.csv"); df_te=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_test.csv")
Xall=df_tr[FEAT].values.astype(np.float32); Xte=df_te[FEAT].values.astype(np.float32)
split=int(len(Xall)*0.8); Xtr,Xv=Xall[:split],Xall[split:]
sc=StandardScaler().fit(Xtr); Xtr_s,Xv_s,Xte_s=sc.transform(Xtr),sc.transform(Xv),sc.transform(Xte)
y_pf=pd.read_csv(f"{BASE}/TELCO_labels/TELCO_labels_test.csv")[FEAT].values.astype(int)
def macro(p): return (float(np.mean([f1_score(y_pf[:,i],p[:,i],zero_division=0) for i in range(12)])),
                      float(np.mean([precision_score(y_pf[:,i],p[:,i],zero_division=0) for i in range(12)])),
                      float(np.mean([recall_score(y_pf[:,i],p[:,i],zero_division=0) for i in range(12)])))
bcast=lambda f: np.repeat(f[:,None],12,axis=1); ksig=lambda st,sv:(st>sv.mean()+K_SIGMA*sv.std()).astype(int)

M=Xtr_s.shape[1]
n_pca=int(np.argmax(np.cumsum(SK_PCA(n_components=M).fit(Xtr_s).explained_variance_ratio_)>=0.95)+1)
nn=NearestNeighbors(n_neighbors=50).fit(Xtr_s); d_,_=nn.kneighbors(Xtr_s); mk=d_.mean(0)
yk=(mk-mk.min())/(mk.max()-mk.min()+1e-12); n_k=int(np.argmax(yk-np.linspace(0,1,50)))+1
grid=[50,100,150,200,300,500]; mp=[IForest(n_estimators=ne,random_state=SEED).fit(Xtr_s).decision_function(Xtr_s).std() for ne in grid]
ch=[abs(mp[i]-mp[i-1])/max(mp[i-1],1e-12) for i in range(1,len(mp))]; n_if=grid[next((i for i,c in enumerate(ch) if c<0.02),len(ch)-1)+1]
bb,bn=np.inf,2
for n in [2,3,4,5,6,8]:
    for ct in ['full','tied','diag','spherical']:
        try:
            b=SK_GMM(n_components=n,covariance_type=ct,random_state=SEED).fit(Xtr_s).bic(Xtr_s)
            if b<bb: bb,bn=b,n
        except: pass
n_b=int(np.log2(len(Xtr_s))+1)
dets={"IForest":IForest(n_estimators=n_if,random_state=SEED),"LOF":LOF(n_neighbors=n_k,novelty=True),
      "OCSVM":OCSVM(kernel="rbf",nu=0.05),"ECOD":ECOD(),"COPOD":COPOD(),"KNN":KNN(n_neighbors=n_k,method="mean"),
      "PCA":PCA_AD(n_components=n_pca),"AE":AutoEncoder(hidden_neuron_list=[64,32],epoch_num=50,batch_size=64,dropout_rate=0.1,random_state=SEED,verbose=0),
      "GMM":PYOD_GMM(n_components=bn,random_state=SEED),"HBOS":HBOS(n_bins=n_b)}
DETS=list(dets.keys()); s_tr,s_v,s_t={},{},{}
for nm,m in dets.items():
    m.fit(Xtr_s); s_tr[nm]=m.decision_function(Xtr_s); s_v[nm]=m.decision_function(Xv_s); s_t[nm]=m.decision_function(Xte_s)

# Stage 1: Mass-Volume proxy quality q_d
rng=np.random.default_rng(SEED); sub_dim=min(3,M); A_sub={d:[] for d in DETS}
for _ in range(MV_N_SUBSPACE):
    cols=rng.choice(M,sub_dim,replace=False); Xs=Xtr_s[:,cols]; lo,hi=Xs.min(0),Xs.max(0)
    if np.any(hi-lo==0): continue
    Xu=rng.uniform(lo,hi,size=(MV_N_UNIF,sub_dim))
    for d in DETS:
        g=lgb.LGBMRegressor(n_estimators=LGB_TREES_MV,learning_rate=0.1,max_depth=4,random_state=SEED,verbose=-1,n_jobs=1)
        g.fit(Xs,s_tr[d]); A_sub[d].append((g.predict(Xu)>np.quantile(s_tr[d],0.9)).mean())
A={d:float(np.mean(A_sub[d])) for d in DETS}; amax,amin=max(A.values()),min(A.values())
q={d:(amax-A[d])/(amax-amin+1e-12) for d in DETS}

# Stage 2: pairwise diversity 1 - Spearman
rho={}
for i,a in enumerate(DETS):
    for b_ in DETS[i+1:]:
        r,_=spearmanr(s_tr[a],s_tr[b_]); rho[(a,b_)]=r; rho[(b_,a)]=r

# Stage 3: surrogate + validation fidelity R^2 + TreeSHAP
SHv,SHt,R2={},{},{}
for d in DETS:
    g=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,verbose=-1)
    g.fit(Xtr_s,s_tr[d]); R2[d]=r2_score(s_v[d],g.predict(Xv_s))
    ex=shap.TreeExplainer(g); SHv[d]=ex.shap_values(Xv_s); SHt[d]=ex.shap_values(Xte_s); nm=SHv[d].std()+1e-12; SHv[d]/=nm; SHt[d]/=nm
flags={d:ksig(s_t[d],s_v[d]) for d in DETS}

def gate(members):
    """Stage 4-5: anchor (max q) k-sigma flag AND per-KPI sigma gate."""
    a=max(members,key=lambda d:q[d]); fa=flags[a]
    sv=np.stack([SHv[d] for d in members]); st=np.stack([SHt[d] for d in members])
    sig_v=sv.std(0); sig_t=st.std(0); tau=np.array([np.quantile(sig_v[:,j],Q) for j in range(12)])
    filt=np.stack([(sig_t[:,j]>tau[j]).astype(int) for j in range(12)],axis=1)
    return a,(bcast(fa)*filt).astype(int)
qbar=lambda S: np.mean([q[d] for d in S])
dbar=lambda S: np.mean([1-rho[(i,j)] for i,j in itertools.combinations(S,2)]) if len(S)>1 else 0.0

POOL=[d for d in DETS if R2[d]>=R2_MIN]
print("validation surrogate R2:", {d:round(R2[d],3) for d in DETS})
print(f"fidelity filter R2>={R2_MIN}: pool ({len(POOL)}) {POOL} | dropped {[d for d in DETS if d not in POOL]}\n")

# Table II: Mixed selection swept over K_R (label-free)
sweep=[]
for KR in range(2,len(POOL)+1):
    best=max(itertools.combinations(POOL,KR), key=lambda S: qbar(S)*dbar(S))
    a,pred=gate(best); f1,p,r=macro(pred)
    sweep.append({"K_R":KR,"members":"+".join(best),"anchor":a,"F1":round(f1,4),"P":round(p,4),"R":round(r,4)})
    print(f"K_R={KR} {'+'.join(best)} anchor={a} F1={f1:.4f} P={p:.4f} R={r:.4f}")
sw=pd.DataFrame(sweep); sw.to_csv(f"{RES}/telco_r075_mixed_sweep.csv",index=False)

# Table III: per-KPI breakdown of the DEFAULT K_R=2 operating point
defrow=sw[sw.K_R==2].iloc[0]; defset=tuple(defrow.members.split("+"))
a,pred=gate(defset); f1,p,r=macro(pred)
print(f"\n>>> default operating point: K_R=2 {defrow.members} anchor={a} F1={f1:.4f} P={p:.4f} R={r:.4f}")
rows=[]
for i,ts in enumerate(FEAT):
    yt=y_pf[:,i]; pp=pred[:,i]
    rows.append({"KPI":ts,"#true":int(yt.sum()),"#pred":int(pp.sum()),
                 "TP":int(((yt==1)&(pp==1)).sum()),"FP":int(((yt==0)&(pp==1)).sum()),"FN":int(((yt==1)&(pp==0)).sum()),
                 "F1":round(f1_score(yt,pp,zero_division=0),3),"P":round(precision_score(yt,pp,zero_division=0),3),"R":round(recall_score(yt,pp,zero_division=0),3)})
pk=pd.DataFrame(rows); pk.to_csv(f"{RES}/telco_r075_best_perkpi.csv",index=False)
print(pk.to_string(index=False)); print(f"macro F1={f1:.4f} P={p:.4f} R={r:.4f} #pred={pk['#pred'].sum()}")

# Fig data: all 45 K_R=2 pairs (joint score vs pipeline F1), dropped pairs included
land=[]
for i,j in itertools.combinations(DETS,2):
    _,pr=gate((i,j)); fj,_,_=macro(pr)
    land.append({"pair":f"{i}+{j}","joint_pair_score":round(qbar((i,j))*dbar((i,j)),4),"F1":round(fj,4)})
pd.DataFrame(land).to_csv(f"{RES}/telco_B_pair_landscape.csv",index=False)
print("\nsaved telco_r075_mixed_sweep.csv, telco_r075_best_perkpi.csv, telco_B_pair_landscape.csv")
