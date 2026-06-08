"""
Table I -- per-KPI macro F1/P/R of the 10 self-tuned singleton detectors
on TELCO. Each detector is tuned label-free (PCA 95% var, kNN/LOF
k-distance elbow, IForest path-length plateau, GMM BIC, HBOS Sturges,
OCSVM nu=0.05, AE bottleneck = tuned PCA dim) and flagged by a 3-sigma
rule broadcast across the 12 KPI streams.
Output: results/telco_tuned_singleton.csv
"""
import os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as SK_PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture as SK_GMM
from sklearn.metrics import precision_score, recall_score, f1_score, auc as sk_auc
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

BASE = "../data"; RESULTS = "../results"
TRAIN_DATA = f"{BASE}/TELCO_data/TELCO_data_train.csv"
TEST_DATA  = f"{BASE}/TELCO_data/TELCO_data_test.csv"
LABELS     = f"{BASE}/TELCO_labels/TELCO_labels_test.csv"

K_SIGMA, Q = 3.0, 0.75
FEAT = [f"TS{i}" for i in range(1, 13)]
SEED = 42

print("Loading...", flush=True)
df_tr = pd.read_csv(TRAIN_DATA); df_te = pd.read_csv(TEST_DATA)
X_tr_all = df_tr[FEAT].values.astype(np.float32); X_te = df_te[FEAT].values.astype(np.float32)
split = int(len(X_tr_all) * 0.8)
X_tr = X_tr_all[:split]; X_v = X_tr_all[split:]
sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr); X_v_s = sc.transform(X_v); X_te_s = sc.transform(X_te)
lab = pd.read_csv(LABELS); y_pf = lab[FEAT].values.astype(int); T = len(y_pf)

def macro(pred_pf):
    P, R, F = [], [], []
    for i in range(12):
        P.append(precision_score(y_pf[:, i], pred_pf[:, i], zero_division=0))
        R.append(recall_score(y_pf[:, i], pred_pf[:, i], zero_division=0))
        F.append(f1_score(y_pf[:, i], pred_pf[:, i], zero_division=0))
    return float(np.mean(F)), float(np.mean(P)), float(np.mean(R))
def broadcast(flag): return np.repeat(flag[:, None], 12, axis=1)
def kf(s_t, s_v, k=K_SIGMA): return (s_t > s_v.mean() + k*s_v.std()).astype(int)

# === Self-tuning ===
M = X_tr_s.shape[1]
pca_full = SK_PCA(n_components=M).fit(X_tr_s)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_pca = int(np.argmax(cumvar >= 0.95) + 1)

MAX_K = 50
nn = NearestNeighbors(n_neighbors=MAX_K).fit(X_tr_s)
distances, _ = nn.kneighbors(X_tr_s)
mean_kth = distances.mean(axis=0)
x_norm = np.linspace(0, 1, MAX_K)
y_norm = (mean_kth - mean_kth.min()) / (mean_kth.max() - mean_kth.min() + 1e-12)
elbow_idx = int(np.argmax(y_norm - x_norm))
n_knn = elbow_idx + 1; n_lof = elbow_idx + 1

n_estimators_grid = [50, 100, 150, 200, 300, 500]
mean_paths = []
for n_est in n_estimators_grid:
    iforest_tmp = IForest(n_estimators=n_est, random_state=SEED); iforest_tmp.fit(X_tr_s)
    mean_paths.append(iforest_tmp.decision_function(X_tr_s).std())
changes = [abs(mean_paths[i] - mean_paths[i-1]) / max(mean_paths[i-1], 1e-12) for i in range(1, len(mean_paths))]
plateau_idx = next((i for i, c in enumerate(changes) if c < 0.02), len(changes)-1)
n_if = n_estimators_grid[plateau_idx + 1]

best_bic, best_n, best_ct = np.inf, None, None
for n in [2, 3, 4, 5, 6, 8]:
    for ct in ['full', 'tied', 'diag', 'spherical']:
        try:
            g = SK_GMM(n_components=n, covariance_type=ct, random_state=SEED).fit(X_tr_s)
            bic = g.bic(X_tr_s)
            if bic < best_bic: best_bic, best_n, best_ct = bic, n, ct
        except: pass
n_gmm = best_n
n_bins_hbos = int(np.log2(len(X_tr_s)) + 1)

print(f"Tuned configs: PCA={n_pca}, KNN={n_knn}, LOF={n_lof}, IF={n_if}, "
       f"GMM={n_gmm}/{best_ct}, HBOS={n_bins_hbos}", flush=True)

# === Train tuned dets ===
dets = {
    "IsolationForest": IForest(n_estimators=n_if, random_state=SEED),
    "LOF":             LOF(n_neighbors=n_lof, novelty=True),
    "OCSVM":           OCSVM(kernel="rbf", nu=0.05),
    "ECOD":            ECOD(),
    "COPOD":           COPOD(),
    "KNN":             KNN(n_neighbors=n_knn, method="mean"),
    "PCA":             PCA_AD(n_components=n_pca),
    "AutoEncoder":     AutoEncoder(hidden_neuron_list=[64,32], epoch_num=50, batch_size=64,
                                    dropout_rate=0.1, random_state=SEED, verbose=0),
    "GMM":             PYOD_GMM(n_components=n_gmm, random_state=SEED),
    "HBOS":            HBOS(n_bins=n_bins_hbos),
}
DETS = list(dets.keys())
print("\nTraining tuned detectors...", flush=True)
s_tr = {}; s_v = {}; s_t = {}
for name, m in dets.items():
    m.fit(X_tr_s)
    s_tr[name] = m.decision_function(X_tr_s)
    s_v[name]  = m.decision_function(X_v_s)
    s_t[name]  = m.decision_function(X_te_s)
    print(f"  {name} OK", flush=True)

# === EM/MV for tuned ===
def em_curve(s_X, s_U, vol, t_grid):
    EM = np.zeros_like(t_grid); n, nu = len(s_X), len(s_U)
    for u in np.unique(s_X):
        EM = np.maximum(EM, (s_X > u).sum()/n - t_grid * (s_U > u).sum()/nu * vol)
    return EM
def mv_curve(s_X, s_U, vol, ag):
    n, nu = len(s_X), len(s_U); ss = np.sort(s_X)[::-1]
    mv = np.zeros_like(ag); cpt, mass = 0, 0.0; u = ss[0]
    for i, a in enumerate(ag):
        while mass < a and cpt < n - 1: cpt += 1; u = ss[cpt]; mass = cpt / n
        mv[i] = (s_U >= u).sum() / nu * vol
    return mv

FAST = [d for d in DETS if d not in ("OCSVM", "AutoEncoder")]
em_s = {d: [] for d in FAST}; mv_s = {d: [] for d in FAST}
rng = np.random.default_rng(SEED)
N_SUB, D_SUB, N_UNIF, N_TR = 10, min(5, M), 50_000, min(5000, len(X_tr_s))
print("\nEM/MV...", flush=True)
for k_sub in range(N_SUB):
    cols = rng.choice(M, size=D_SUB, replace=False)
    Xs = X_tr_s[:, cols]
    li, lu = Xs.min(0), Xs.max(0); vol = float(np.prod(lu-li))
    if vol == 0: continue
    Xu = rng.uniform(li, lu, size=(N_UNIF, D_SUB))
    idx = rng.choice(len(Xs), size=N_TR, replace=False); Xs_sub = Xs[idx]
    tg = np.linspace(0, 0.9/vol, 500); ag = np.linspace(0.9, 0.999, 500)
    # rebuild small detectors per subspace
    sub_dets = {
        "IsolationForest": IForest(n_estimators=n_if, random_state=SEED),
        "LOF":             LOF(n_neighbors=n_lof, novelty=True),
        "ECOD":            ECOD(), "COPOD":           COPOD(),
        "KNN":             KNN(n_neighbors=n_knn, method="mean"),
        "PCA":             PCA_AD(n_components=min(n_pca, D_SUB)),
        "GMM":             PYOD_GMM(n_components=n_gmm, random_state=SEED),
        "HBOS":            HBOS(n_bins=n_bins_hbos),
    }
    for d in FAST:
        try:
            m = sub_dets[d]; m.fit(Xs)
            sX = m.decision_function(Xs_sub); sU = m.decision_function(Xu)
            em_s[d].append(float(sk_auc(tg, em_curve(sX, sU, vol, tg))))
            mv_s[d].append(float(sk_auc(ag, mv_curve(sX, sU, vol, ag))))
        except Exception:
            em_s[d].append(np.nan); mv_s[d].append(np.nan)
    print(f"  subset {k_sub+1}/{N_SUB}", flush=True)
emmv = pd.DataFrame([{"detector": d,
                       "EM": float(np.nanmean(em_s[d])) if d in em_s else np.nan,
                       "MV": float(np.nanmean(mv_s[d])) if d in mv_s else np.nan} for d in DETS])
emmv["EM_rk"] = emmv["EM"].rank(ascending=False, method='min', na_option='bottom').fillna(len(DETS)).astype(int)
emmv["MV_rk"] = emmv["MV"].rank(ascending=True, method='min', na_option='bottom').fillna(len(DETS)).astype(int)
emmv["avg_rk"] = (emmv["EM_rk"] + emmv["MV_rk"]) / 2
em_rk_dict = dict(zip(emmv["detector"], emmv["EM_rk"]))
anchor = emmv.sort_values("avg_rk").iloc[0]["detector"]
print(f"\nAnchor: {anchor}", flush=True)

# Canonical order
cands = []
for d in DETS:
    if d == anchor: continue
    r, _ = spearmanr(s_tr[anchor], s_tr[d])
    cands.append((d, (1 - r) * (11 - int(em_rk_dict[d]))))
cands.sort(key=lambda x: -x[1])
ordered = [anchor] + [d for d, _ in cands]
print(f"Order: {ordered}", flush=True)

# === SHAP ===
print("\nSHAP...", flush=True)
SHAP_v, SHAP_t = {}, {}
for d in DETS:
    surr = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1)
    surr.fit(X_tr_s, s_tr[d])
    expl = shap.TreeExplainer(surr)
    SHAP_v[d] = expl.shap_values(X_v_s); SHAP_t[d] = expl.shap_values(X_te_s)
    nm = SHAP_v[d].std() + 1e-12; SHAP_v[d] /= nm; SHAP_t[d] /= nm

# === TABLO 1 — Singleton (canonical sırada) ===
flags = {d: kf(s_t[d], s_v[d]) for d in DETS}
singleton_rows = []
for i, d in enumerate(ordered, start=1):
    f1, p, r = macro(broadcast(flags[d]))
    singleton_rows.append({"K": i, "Dedektör": d,
                            "Singleton F1": round(f1, 4),
                            "Singleton P": round(p, 4),
                            "Singleton R": round(r, 4)})
df1 = pd.DataFrame(singleton_rows)
df1.to_csv(f"{RESULTS}/telco_tuned_singleton.csv", index=False)
with open(f"{RESULTS}/telco_tuned_singleton.tex", "w") as fh:
    fh.write("\\begin{tabular}{rlrrr}\n\\toprule\n")
    fh.write("K & Dedektör & Singleton F1 & Singleton P & Singleton R \\\\\n\\midrule\n")
    for _, r in df1.iterrows():
        fh.write(f"{r['K']} & {r['Dedektör']} & {r['Singleton F1']:.4f} & {r['Singleton P']:.4f} & {r['Singleton R']:.4f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print("\nTABLO 1 — Singleton (tuned, canonical sıralama):", flush=True)
print(df1.to_string(index=False), flush=True)

# === TABLO 2 — Pipeline 3 fusion ===
f_anchor = flags[anchor]
pipeline_rows = []
for K_size in range(1, len(DETS)+1):
    members = ordered[:K_size]
    if K_size == 1:
        f1_s, p_s, r_s = macro(broadcast(f_anchor))
        f1_or, p_or, r_or = f1_s, p_s, r_s
        f1_and, p_and, r_and = f1_s, p_s, r_s
    else:
        sv = np.stack([SHAP_v[d] for d in members])
        st = np.stack([SHAP_t[d] for d in members])
        sig_v = np.std(sv, axis=0); sig_t = np.std(st, axis=0)
        tau = np.array([np.quantile(sig_v[:, j], Q) for j in range(12)])
        filt = np.zeros((T, 12), dtype=int)
        for j in range(12):
            filt[:, j] = (sig_t[:, j] > tau[j]).astype(int)
        f1_s, p_s, r_s = macro((broadcast(f_anchor) * filt).astype(int))
        or_flag = np.zeros(T, dtype=int); and_flag = np.ones(T, dtype=int)
        for d in members:
            or_flag = or_flag | flags[d]
            and_flag = and_flag & flags[d]
        f1_or, p_or, r_or = macro(broadcast(or_flag).astype(int))
        f1_and, p_and, r_and = macro(broadcast(and_flag).astype(int))
    pipeline_rows.append({
        "K": K_size, "Üyeler": ", ".join(members),
        "F1 (anchor + σ)": round(f1_s, 4), "P (anchor + σ)": round(p_s, 4), "R (anchor + σ)": round(r_s, 4),
        "F1 (OR)": round(f1_or, 4), "P (OR)": round(p_or, 4), "R (OR)": round(r_or, 4),
        "F1 (AND)": round(f1_and, 4), "P (AND)": round(p_and, 4), "R (AND)": round(r_and, 4),
    })
print(f"\nSaved: {RESULTS}/telco_tuned_singleton.csv (+ .tex)  [Table I]")
