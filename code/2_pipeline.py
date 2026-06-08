"""
Main SHAP-sigma pipeline on the 10-detector self-tuned pool (Sec. II).

Implements the paper's Stage 1-2:
  - Stage 1: quality q_d via the surrogate Mass-Volume proxy A_d (Eq. Ad),
    LightGBM surrogate per subspace, 3 subspaces x 2000 uniform samples,
    sub_dim=min(3,M), then min-max invert.
  - Stage 2: JOINT selection over all C(K,K_R) subsets, Mixed strategy
    argmax qbar * deltabar; closed-form joint pair at K_R=2.
  - Anchor = argmax q_d in R; SHAP-sigma per-feature gate; per-KPI macro F1.

Outputs Table II (joint-subset Mixed sweep), the K_R=2 pair landscape
(for Fig 1), and the per-KPI breakdown at the peak K_R.
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
import shap, lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

BASE="../data"; RESULTS="../results"
SEED=42; K_SIGMA=3.0; Q=0.75
MV_N_SUBSPACE=3; MV_N_UNIF=2000; LGB_TREES_MV=80
FEAT=[f"TS{i}" for i in range(1,13)]

df_tr=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_train.csv"); df_te=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_test.csv")
Xall=df_tr[FEAT].values.astype(np.float32); Xte=df_te[FEAT].values.astype(np.float32)
split=int(len(Xall)*0.8); Xtr,Xv=Xall[:split],Xall[split:]
sc=StandardScaler().fit(Xtr); Xtr_s,Xv_s,Xte_s=sc.transform(Xtr),sc.transform(Xv),sc.transform(Xte)
lab=pd.read_csv(f"{BASE}/TELCO_labels/TELCO_labels_test.csv"); y_pf=lab[FEAT].values.astype(int); T=len(y_pf)

def macro(pred):
    F=[f1_score(y_pf[:,i],pred[:,i],zero_division=0) for i in range(12)]
    P=[precision_score(y_pf[:,i],pred[:,i],zero_division=0) for i in range(12)]
    R=[recall_score(y_pf[:,i],pred[:,i],zero_division=0) for i in range(12)]
    return float(np.mean(F)),float(np.mean(P)),float(np.mean(R))
def bcast(f): return np.repeat(f[:,None],12,axis=1)
def ksig(st,sv,k=K_SIGMA): return (st>sv.mean()+k*sv.std()).astype(int)

# ---- self-tune (same as telco_self_tuning_2tables) ----
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
DETS=list(dets.keys())
s_tr,s_v,s_t={},{},{}
for nm,m in dets.items():
    m.fit(Xtr_s); s_tr[nm]=m.decision_function(Xtr_s); s_v[nm]=m.decision_function(Xv_s); s_t[nm]=m.decision_function(Xte_s)
print("tuned & fit ok", flush=True)

# ---- Stage 1: q_d via Eq(Ad) surrogate MV proxy ----
rng=np.random.default_rng(SEED); sub_dim=min(3,M)
A_sub={d:[] for d in DETS}
for _ in range(MV_N_SUBSPACE):
    cols=rng.choice(M,sub_dim,replace=False); Xs=Xtr_s[:,cols]; lo,hi=Xs.min(0),Xs.max(0)
    if np.any(hi-lo==0): continue
    Xu=rng.uniform(lo,hi,size=(MV_N_UNIF,sub_dim))
    for d in DETS:
        surr=lgb.LGBMRegressor(n_estimators=LGB_TREES_MV,learning_rate=0.1,max_depth=4,random_state=SEED,verbose=-1,n_jobs=1)
        surr.fit(Xs,s_tr[d]); su=surr.predict(Xu); th=np.quantile(s_tr[d],0.9)
        A_sub[d].append((su>th).mean())
A={d:float(np.mean(A_sub[d])) for d in DETS}
amax,amin=max(A.values()),min(A.values())
q={d:(amax-A[d])/(amax-amin+1e-12) for d in DETS}
print("quality q_d:", {d:round(q[d],3) for d in sorted(q,key=lambda x:-q[x])}, flush=True)

# ---- Spearman rho ----
rho={}
for i,a in enumerate(DETS):
    for b_ in DETS[i+1:]:
        r,_=spearmanr(s_tr[a],s_tr[b_]); rho[(a,b_)]=r; rho[(b_,a)]=r

# ---- SHAP per detector (+ surrogate fidelity R^2) ----
SHv,SHt,R2={},{},{}
for d in DETS:
    g=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,verbose=-1)
    g.fit(Xtr_s,s_tr[d]); R2[d]=round(float(r2_score(s_v[d],g.predict(Xv_s))),3); ex=shap.TreeExplainer(g)
    SHv[d]=ex.shap_values(Xv_s); SHt[d]=ex.shap_values(Xte_s); nmv=SHv[d].std()+1e-12; SHv[d]/=nmv; SHt[d]/=nmv
pd.DataFrame([{"detector":d,"R2_val":R2[d]} for d in DETS]).to_csv(f"{RESULTS}/surrogate_fidelity.csv",index=False)
print("shap ok | surrogate R2_val median =",round(float(np.median(list(R2.values()))),3), flush=True)

flags={d:ksig(s_t[d],s_v[d]) for d in DETS}

def gate_eval(members):
    a=max(members,key=lambda d:q[d])           # anchor = argmax q_d
    fa=flags[a]
    sv=np.stack([SHv[d] for d in members]); st=np.stack([SHt[d] for d in members])
    sig_v=sv.std(0); sig_t=st.std(0)
    tau=np.array([np.quantile(sig_v[:,j],Q) for j in range(12)])
    filt=np.stack([(sig_t[:,j]>tau[j]).astype(int) for j in range(12)],axis=1)
    f1,p,r=macro((bcast(fa)*filt).astype(int))
    return a,f1,p,r

def qbar(S): return np.mean([q[d] for d in S])
def dbar(S):
    pairs=list(itertools.combinations(S,2))
    return np.mean([1-rho[(i,j)] for i,j in pairs]) if pairs else 0.0

# ---- Stage 2: joint-subset Mixed sweep ----
sweep=[]
for KR in range(2,len(DETS)+1):
    best=None
    for S in itertools.combinations(DETS,KR):
        sc_mixed=qbar(S)*dbar(S)
        if best is None or sc_mixed>best[0]: best=(sc_mixed,S)
    S=best[1]; a,f1,p,r=gate_eval(S)
    sweep.append({"K_R":KR,"members":"+".join(S),"anchor":a,"mixed_score":round(best[0],4),"F1":round(f1,4),"P":round(p,4),"R":round(r,4)})
    print(f"K_R={KR} mixed={'+'.join(S)} anchor={a} F1={f1:.4f}", flush=True)
sw=pd.DataFrame(sweep); sw.to_csv(f"{RESULTS}/telco_B_mixed_sweep.csv",index=False)

# ---- K_R=2 pair landscape (Fig 1) ----
pairs=[]
for i,j in itertools.combinations(DETS,2):
    jp=0.5*(q[i]+q[j])*(1-rho[(i,j)])
    a,f1,p,r=gate_eval((i,j))
    pairs.append({"pair":f"{i}+{j}","joint_pair_score":round(jp,4),"F1":round(f1,4),"anchor":a})
pl=pd.DataFrame(pairs).sort_values("joint_pair_score",ascending=False)
pl.to_csv(f"{RESULTS}/telco_B_pair_landscape.csv",index=False)
jmax=pl.iloc[0]; oracle=pl.sort_values("F1",ascending=False).iloc[0]
print(f"\njoint-pair MAX: {jmax['pair']} (score={jmax.joint_pair_score}, F1={jmax.F1})")
print(f"F1 ORACLE pair: {oracle['pair']} (F1={oracle.F1})")
print(f"coincide: {jmax['pair']==oracle['pair']}")

# ---- label-free quality-rule set + per-KPI (Table III) ----
# Variance-stability rule: keep detectors with above-median quality (no labels).
qmed=float(np.median(list(q.values())))
members=tuple(sorted([d for d in DETS if q[d]>qmed],key=lambda d:-q[d]))
a=max(members,key=lambda d:q[d]); fa=flags[a]
sv=np.stack([SHv[d] for d in members]); st=np.stack([SHt[d] for d in members]); sig_v=sv.std(0); sig_t=st.std(0)
tau=np.array([np.quantile(sig_v[:,j],Q) for j in range(12)]); filt=np.stack([(sig_t[:,j]>tau[j]).astype(int) for j in range(12)],axis=1)
pred=(bcast(fa)*filt).astype(int)
rows=[]
for i,ts in enumerate(FEAT):
    yt=y_pf[:,i]; pp=pred[:,i]; tp=int(((yt==1)&(pp==1)).sum()); fp=int(((yt==0)&(pp==1)).sum()); fn=int(((yt==1)&(pp==0)).sum())
    rows.append({"KPI":ts,"#true":int(yt.sum()),"#pred":int(pp.sum()),"TP":tp,"FP":fp,"FN":fn,
                 "F1":round(f1_score(yt,pp,zero_division=0),3),"P":round(precision_score(yt,pp,zero_division=0),3),"R":round(recall_score(yt,pp,zero_division=0),3)})
pk=pd.DataFrame(rows); pk.to_csv(f"{RESULTS}/telco_rule_perkpi.csv",index=False)
mF,mP,mR=macro(pred)
print(f"\nLABEL-FREE RULE (q>median={qmed:.3f}) set={'+'.join(members)} anchor={a}")
print(pk.to_string(index=False)); print(f"macro F1={mF:.4f} P={mP:.4f} R={mR:.4f} #pred={pk['#pred'].sum()}")
print("\n=== K_R SENSITIVITY SWEEP ==="); print(sw.to_string(index=False))
print("saved telco_B_mixed_sweep.csv, telco_B_pair_landscape.csv, telco_rule_perkpi.csv, surrogate_fidelity.csv")
