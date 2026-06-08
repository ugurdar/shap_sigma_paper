"""
5-seed robustness of the SHAP-sigma headline at the peak K_R=7 set
(anchor LOF), with Q fixed at the paper default 0.75 (no test-set tuning).
Reports mean +/- std over seeds {42,7,123,2024,1} for SHAP-sigma and, for
reference, the anchor and simple gate baselines (vote>=2/3, per-KPI SHAP
vote). The paper cites SHAP-sigma = 0.447 +/- 0.013 from here.
Output: results/telco_robustness.csv
"""
import itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as SK_PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture as SK_GMM
from sklearn.metrics import precision_score, recall_score, f1_score
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
BASE="../data"; RESULTS="../results"; K_SIGMA=3.0; Q=0.75
MV_N_SUBSPACE=3; MV_N_UNIF=2000; LGB_TREES_MV=80; FEAT=[f"TS{i}" for i in range(1,13)]
SEEDS=[42,7,123,2024,1]
df_tr=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_train.csv"); df_te=pd.read_csv(f"{BASE}/TELCO_data/TELCO_data_test.csv")
Xall=df_tr[FEAT].values.astype(np.float32); Xte=df_te[FEAT].values.astype(np.float32)
split=int(len(Xall)*0.8); Xtr,Xv=Xall[:split],Xall[split:]
sc=StandardScaler().fit(Xtr); Xtr_s,Xv_s,Xte_s=sc.transform(Xtr),sc.transform(Xv),sc.transform(Xte)
lab=pd.read_csv(f"{BASE}/TELCO_labels/TELCO_labels_test.csv"); y_pf=lab[FEAT].values.astype(int); T=len(y_pf)
def macroF(p): return float(np.mean([f1_score(y_pf[:,i],p[:,i],zero_division=0) for i in range(12)]))
def bcast(f): return np.repeat(f[:,None],12,axis=1)
def ksig(st,sv,k=K_SIGMA): return (st>sv.mean()+k*sv.std()).astype(int)
M=Xtr_s.shape[1]

def build(seed):
    n_pca=int(np.argmax(np.cumsum(SK_PCA(n_components=M).fit(Xtr_s).explained_variance_ratio_)>=0.95)+1)
    nn=NearestNeighbors(n_neighbors=50).fit(Xtr_s); d_,_=nn.kneighbors(Xtr_s); mk=d_.mean(0)
    yk=(mk-mk.min())/(mk.max()-mk.min()+1e-12); n_k=int(np.argmax(yk-np.linspace(0,1,50)))+1
    grid=[50,100,150,200,300,500]; mp=[IForest(n_estimators=ne,random_state=seed).fit(Xtr_s).decision_function(Xtr_s).std() for ne in grid]
    ch=[abs(mp[i]-mp[i-1])/max(mp[i-1],1e-12) for i in range(1,len(mp))]; n_if=grid[next((i for i,c in enumerate(ch) if c<0.02),len(ch)-1)+1]
    bb,bn=np.inf,2
    for n in [2,3,4,5,6,8]:
        for ct in ['full','tied','diag','spherical']:
            try:
                b=SK_GMM(n_components=n,covariance_type=ct,random_state=seed).fit(Xtr_s).bic(Xtr_s)
                if b<bb: bb,bn=b,n
            except: pass
    n_b=int(np.log2(len(Xtr_s))+1)
    dets={"IForest":IForest(n_estimators=n_if,random_state=seed),"LOF":LOF(n_neighbors=n_k,novelty=True),
          "OCSVM":OCSVM(kernel="rbf",nu=0.05),"ECOD":ECOD(),"COPOD":COPOD(),"KNN":KNN(n_neighbors=n_k,method="mean"),
          "PCA":PCA_AD(n_components=n_pca),"AE":AutoEncoder(hidden_neuron_list=[64,32],epoch_num=50,batch_size=64,dropout_rate=0.1,random_state=seed,verbose=0),
          "GMM":PYOD_GMM(n_components=bn,random_state=seed),"HBOS":HBOS(n_bins=n_b)}
    DETS=list(dets.keys()); s_tr,s_v,s_t={},{},{}
    for nm,m in dets.items():
        m.fit(Xtr_s); s_tr[nm]=m.decision_function(Xtr_s); s_v[nm]=m.decision_function(Xv_s); s_t[nm]=m.decision_function(Xte_s)
    rng=np.random.default_rng(seed); sub_dim=min(3,M); A_sub={d:[] for d in DETS}
    for _ in range(MV_N_SUBSPACE):
        cols=rng.choice(M,sub_dim,replace=False); Xs=Xtr_s[:,cols]; lo,hi=Xs.min(0),Xs.max(0)
        if np.any(hi-lo==0): continue
        Xu=rng.uniform(lo,hi,size=(MV_N_UNIF,sub_dim))
        for d in DETS:
            g=lgb.LGBMRegressor(n_estimators=LGB_TREES_MV,learning_rate=0.1,max_depth=4,random_state=seed,verbose=-1,n_jobs=1)
            g.fit(Xs,s_tr[d]); A_sub[d].append((g.predict(Xu)>np.quantile(s_tr[d],0.9)).mean())
    A={d:float(np.mean(A_sub[d])) for d in DETS}; amax,amin=max(A.values()),min(A.values())
    q={d:(amax-A[d])/(amax-amin+1e-12) for d in DETS}
    rho={}
    for i,a in enumerate(DETS):
        for b_ in DETS[i+1:]:
            r,_=spearmanr(s_tr[a],s_tr[b_]); rho[(a,b_)]=r; rho[(b_,a)]=r
    SHv,SHt={},{}
    for d in DETS:
        g=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=seed,verbose=-1)
        g.fit(Xtr_s,s_tr[d]); ex=shap.TreeExplainer(g); SHv[d]=ex.shap_values(Xv_s); SHt[d]=ex.shap_values(Xte_s)
        nmv=SHv[d].std()+1e-12; SHv[d]/=nmv; SHt[d]/=nmv
    flags={d:ksig(s_t[d],s_v[d]) for d in DETS}
    return DETS,q,rho,SHv,SHt,flags

def mixed_set(DETS,q,rho,KR):
    def qb(S): return np.mean([q[d] for d in S])
    def db(S):
        pr=list(itertools.combinations(S,2)); return np.mean([1-rho[(i,j)] for i,j in pr]) if pr else 0.0
    best=None
    for S in itertools.combinations(DETS,KR):
        sm=qb(S)*db(S)
        if best is None or sm>best[0]: best=(sm,S)
    return best[1]

res={k:[] for k in ["anchor","SHAPsigma","vote>=2","vote>=3","G1 k=2","G1 k=3","G1 k=4"]}
for seed in SEEDS:
    DETS,q,rho,SHv,SHt,flags=build(seed)
    S=mixed_set(DETS,q,rho,7); a=max(S,key=lambda d:q[d]); fa=flags[a]; K=len(S)
    AV=np.stack([np.abs(SHv[d]) for d in S]); AT=np.stack([np.abs(SHt[d]) for d in S])
    SVr=np.stack([SHv[d] for d in S]); STr=np.stack([SHt[d] for d in S])
    sigv=SVr.std(0); sigt=STr.std(0); tau=np.array([np.quantile(sigv[:,j],Q) for j in range(12)])
    shapσ=(bcast(fa)*np.stack([(sigt[:,j]>tau[j]).astype(int) for j in range(12)],1)).astype(int)
    votes=np.sum([flags[d] for d in S],0)
    th=np.stack([[np.quantile(AV[d,:,j],Q) for j in range(12)] for d in range(K)])
    cnt=np.zeros((T,12),int)
    for d in range(K): cnt+=(AT[d]>th[d][None,:]).astype(int)
    res["anchor"].append(macroF(bcast(fa)))
    res["SHAPsigma"].append(macroF(shapσ))
    res["vote>=2"].append(macroF((bcast(fa)*bcast((votes>=2).astype(int))).astype(int)))
    res["vote>=3"].append(macroF((bcast(fa)*bcast((votes>=3).astype(int))).astype(int)))
    for k in [2,3,4]:
        res[f"G1 k={k}"].append(macroF((bcast(fa)*(cnt>=k).astype(int)).astype(int)))
    print(f"seed {seed}: anchor={res['anchor'][-1]:.3f} SHAPσ={res['SHAPsigma'][-1]:.3f} vote2={res['vote>=2'][-1]:.3f} G1k3={res['G1 k=3'][-1]:.3f}", flush=True)

print("\n===== VERIFY (Q=0.75 fixed, K_R=7, anchor LOF, 5 seeds) =====")
for k,v in res.items():
    print(f"  {k:<14} F1 = {np.mean(v):.4f} +/- {np.std(v):.4f}")
pd.DataFrame(res).to_csv(f"{RESULTS}/telco_robustness.csv",index=False)
print("saved telco_robustness.csv")
