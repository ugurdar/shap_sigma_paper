"""Gate-signal ablation across Rashomon-set sizes K_R = 2..8 on the
fidelity-filtered pool (R^2_val >= 0.75 -> 8 detectors, anchor ECOD).

For each K_R the same per-detector TreeSHAP attributions feed two KPI
gates, under the identical anchor and q=0.75 validation threshold:

  SHAP-sigma : std_d(phi)   cross-detector disagreement  (the paper)
  SHAP-mean  : mean_d|phi|   cross-detector magnitude     (alternative)

(The OR and AND of the two gates are also recorded for completeness.)
This is the source of the gate-variants table and figure: SHAP-sigma is
consistently more precise across K_R, while SHAP-mean trades precision
for a little recall and degrades at large K_R.

Output: results/telco_gate_variants.csv
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

BASE="../data"; RES="../results"
SEED=42; K_SIGMA=3.0; Q=0.75; R2_MIN=0.75; FEAT=[f"TS{i}" for i in range(1,13)]
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
rng=np.random.default_rng(SEED); sub_dim=min(3,M); A_sub={d:[] for d in DETS}
for _ in range(3):
    cols=rng.choice(M,sub_dim,replace=False); Xs=Xtr_s[:,cols]; lo,hi=Xs.min(0),Xs.max(0)
    if np.any(hi-lo==0): continue
    Xu=rng.uniform(lo,hi,size=(2000,sub_dim))
    for d in DETS:
        g=lgb.LGBMRegressor(n_estimators=80,learning_rate=0.1,max_depth=4,random_state=SEED,verbose=-1,n_jobs=1)
        g.fit(Xs,s_tr[d]); A_sub[d].append((g.predict(Xu)>np.quantile(s_tr[d],0.9)).mean())
A={d:float(np.mean(A_sub[d])) for d in DETS}; amax,amin=max(A.values()),min(A.values())
q={d:(amax-A[d])/(amax-amin+1e-12) for d in DETS}
rho={}
for i,a in enumerate(DETS):
    for b_ in DETS[i+1:]:
        r,_=spearmanr(s_tr[a],s_tr[b_]); rho[(a,b_)]=r; rho[(b_,a)]=r
SHv,SHt,R2={},{},{}
for d in DETS:
    g=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=SEED,verbose=-1)
    g.fit(Xtr_s,s_tr[d]); R2[d]=r2_score(s_v[d],g.predict(Xv_s))
    ex=shap.TreeExplainer(g); SHv[d]=ex.shap_values(Xv_s); SHt[d]=ex.shap_values(Xte_s); nm=SHv[d].std()+1e-12; SHv[d]/=nm; SHt[d]/=nm
flags={d:ksig(s_t[d],s_v[d]) for d in DETS}
POOL=[d for d in DETS if R2[d]>=R2_MIN]
qbar=lambda S: np.mean([q[d] for d in S]); dbar=lambda S: np.mean([1-rho[(i,j)] for i,j in itertools.combinations(S,2)]) if len(S)>1 else 0.0

def masks(members):
    SV=np.stack([SHv[d] for d in members]); ST=np.stack([SHt[d] for d in members])
    def gm(av,at):
        tau=np.array([np.quantile(av[:,j],Q) for j in range(12)])
        return np.stack([(at[:,j]>tau[j]) for j in range(12)],axis=1)
    return gm(SV.std(0),ST.std(0)), gm(np.abs(SV).mean(0),np.abs(ST).mean(0))

rows=[]
for KR in range(2,len(POOL)+1):
    S=max(itertools.combinations(POOL,KR), key=lambda s: qbar(s)*dbar(s))
    a=max(S,key=lambda d:q[d]); A_=bcast(flags[a]); gs,gm=masks(S)
    out={"K_R":KR,"members":"+".join(S),"anchor":a}
    for tag,msk in {"sig":gs,"mean":gm,"or":gs|gm,"and":gs&gm}.items():
        f1,p,r=macro((A_*msk).astype(int)); out[f"{tag}_F1"]=round(f1,3); out[f"{tag}_P"]=round(p,3); out[f"{tag}_R"]=round(r,3)
    rows.append(out)
    print(f"K_R={KR} {a:<5} sigma={out['sig_F1']}/{out['sig_P']} mean={out['mean_F1']}/{out['mean_P']}")
pd.DataFrame(rows).to_csv(f"{RES}/telco_gate_variants.csv",index=False)
print("saved telco_gate_variants.csv")
