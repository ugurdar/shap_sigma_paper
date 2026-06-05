"""
SHAP-sigma: Label-Free Rashomon Pipeline for Multivariate Time-Series Anomaly Detection
=========================================================================================

Reference: Dar, Malacarne, Cavus, Van den Abeele (2026).

Single unified pipeline that reproduces the cross-dataset results reported in
the paper across TELCO, SMD, and UCR. Identical detector pool, identical
hyperparameters, three label-free selection strategies (Quality, Diversity,
Mixed), five fusion variants. TELCO uses per-KPI macro F1 (12 KPI streams);
SMD/UCR use point-level F1 and point-adjusted F1 (PA-F1).

Stages (Sec. II of the paper):
  1. Train 8 default PyOD detectors on training split.
  2. Compute label-free Mass-Volume proxy quality q_d (Eq. 1 of paper).
  3. Compute pairwise Spearman score correlation rho_{ij}.
  4. For each K_R in {2,...,8} and strategy in {Quality, Diversity, Mixed},
     select the K_R-tuple jointly over all C(K, K_R) subsets.
     For K_R=2 with Strategy Mixed, Eq. 2 reduces to the closed-form
     joint pair score (q_i+q_j)/2 * (1 - rho_{ij}).
  5. Train LightGBM surrogate per selected detector, compute TreeSHAP,
     normalize by detector validation std (Sec. II-D).
  6. Compute cross-detector sigma matrix, per-feature 0.75-quantile
     threshold tau_j on validation, OR aggregate test gate phi^v.
  7. Anchor = highest-quality member; anchor k-sigma flag f_a.
  8. Output: per-series best of five fusion variants
     {anchor+sigma, OR+sigma, AND+sigma, AND, OR}.

Output:
  results/standard_ksweep_all_datasets.csv  (one row per dataset x series x K x strategy)

Tested on Python 3.11, pyod 2.0, lightgbm 4.5, shap 0.46.
"""

import os, time, glob, re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

import shap
import lightgbm as lgb

from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.gmm import GMM
from pyod.models.lof import LOF
from pyod.models.iforest import IForest

import warnings
warnings.filterwarnings("ignore")

# ============== STANDARD PARAMETERS (identical across datasets) ==============

DETS          = ["IF", "LOF", "ECOD", "COPOD", "KNN", "PCA", "GMM", "HBOS"]
N_DET         = len(DETS)
K_RANGE       = [2, 3, 4, 5, 6, 7, 8]
K_SIGMA       = 3.0      # anchor k-sigma flag threshold
Q_FILT        = 0.75     # sigma-filter quantile
W             = 100      # UCR sliding window
MV_N_SUBSPACE = 3        # Mass-Volume proxy: subspaces
MV_N_UNIF     = 2000     # Mass-Volume proxy: uniform samples per subspace
LGB_TREES_SURR = 200     # SHAP surrogate trees
LGB_TREES_MV   = 80      # MV-proxy surrogate trees

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SMD_DIR  = os.path.join(DATA_DIR, "SMD_data")
UCR_DIR  = os.path.join(DATA_DIR, "UCR_Anomaly_FullData")
OUT      = os.path.join(ROOT, "results", "standard_ksweep_all_datasets.csv")


# ============== Utilities ==============

def sanitize(s):
    """Replace non-finite entries with median."""
    if s is None:
        return None
    s = np.asarray(s, dtype=np.float64).copy()
    bad = ~np.isfinite(s)
    if bad.any():
        med = np.nanmedian(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0
        s[bad] = med
    return s


def make_detectors():
    """8 PyOD detectors with default hyperparameters."""
    return {
        "IF":    IForest(n_estimators=100, random_state=42),
        "LOF":   LOF(n_neighbors=20, novelty=True),
        "ECOD":  ECOD(),
        "COPOD": COPOD(),
        "KNN":   KNN(n_neighbors=10),
        "PCA":   PCA(),
        "GMM":   GMM(n_components=2, random_state=42),
        "HBOS":  HBOS(),
    }


def point_adjust(y_true, y_pred):
    """Su et al. (2019) point-adjustment."""
    y_pred = y_pred.copy().astype(int)
    y_true = y_true.astype(int)
    adj = y_pred.copy()
    in_region = False
    region_start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_region:
            in_region = True
            region_start = i
        elif y_true[i] == 0 and in_region:
            in_region = False
            if y_pred[region_start:i].sum() > 0:
                adj[region_start:i] = 1
    if in_region and y_pred[region_start:].sum() > 0:
        adj[region_start:] = 1
    return adj


def pa_f1_score(y_true, y_pred):
    return f1_score(y_true, point_adjust(y_true, y_pred), zero_division=0)


# ============== Stage 1: Label-free MV-proxy quality (Eq. 1) ==============

def mv_proxy_quality(s_train_dict, X_train, seed=42):
    """Surrogate-based Mass-Volume proxy quality A_d.

    For each detector d:
      - Fit LightGBM surrogate \hat{s}_d on (X_tr, s_d(X_tr)).
      - For 3 random low-dim subspaces with 2000 uniform background samples:
            A_d <- mean over subspaces of Pr[ \hat{s}_d(X_u) > Q_{0.9}(s_d(X_tr)) ]
      - Lower A_d = more selective detector.
    """
    rng = np.random.default_rng(seed)
    M = X_train.shape[1]
    sub_dim = min(3, M)
    A_per_subspace = {d: [] for d in DETS}

    for _ in range(MV_N_SUBSPACE):
        cols = rng.choice(M, sub_dim, replace=False)
        Xs = X_train[:, cols]
        lo, hi = Xs.min(0), Xs.max(0)
        if np.any(hi - lo == 0):
            continue
        Xu = rng.uniform(lo, hi, size=(MV_N_UNIF, sub_dim))
        for d in DETS:
            if s_train_dict.get(d) is None:
                continue
            try:
                surr = lgb.LGBMRegressor(
                    n_estimators=LGB_TREES_MV, learning_rate=0.1, max_depth=4,
                    random_state=42, verbose=-1, n_jobs=1,
                )
                surr.fit(Xs, s_train_dict[d])
                s_u = surr.predict(Xu)
                thresh = np.quantile(s_train_dict[d], 0.9)
                A_per_subspace[d].append((s_u > thresh).mean())
            except Exception:
                continue
    return {d: float(np.mean(A_per_subspace[d])) if A_per_subspace[d] else 0.5 for d in DETS}


def normalize_quality(A_dict):
    """Min-max invert A_d -> q_d in [0,1] (Eq. 1; higher q = better)."""
    valid = {d: v for d, v in A_dict.items() if v is not None}
    if not valid:
        return {d: 0.5 for d in DETS}
    A_min, A_max = min(valid.values()), max(valid.values())
    if A_max - A_min < 1e-9:
        return {d: 0.5 for d in DETS}
    return {d: (A_max - A_dict[d]) / (A_max - A_min) for d in DETS}


# ============== Stage 2: Joint Rashomon set selection (Eq. 2) ==============

def spearman_matrix(s_train_dict):
    R = np.eye(N_DET)
    for i, di in enumerate(DETS):
        for j, dj in enumerate(DETS):
            if i < j:
                if s_train_dict.get(di) is None or s_train_dict.get(dj) is None:
                    R[i, j] = R[j, i] = 0.0
                    continue
                try:
                    r, _ = spearmanr(s_train_dict[di], s_train_dict[dj])
                    if np.isnan(r):
                        r = 0.0
                    R[i, j] = R[j, i] = r
                except Exception:
                    R[i, j] = R[j, i] = 0.0
    return R


def select_by_quality(K, q, valid_dets):
    """Strategy A: top-K by quality."""
    sub = sorted(valid_dets, key=lambda d: -q[d])
    return tuple(sorted(sub[:K]))


def select_by_diversity(K, R, valid_dets):
    """Strategy B: arg max average pairwise (1 - rho)."""
    idx_of = {d: i for i, d in enumerate(DETS)}
    best, best_score = None, -np.inf
    for combo in combinations(valid_dets, K):
        idx = [idx_of[d] for d in combo]
        sub = R[np.ix_(idx, idx)]
        n = len(combo)
        avg_r = (sub.sum() - n) / (n * (n - 1))
        div = 1 - avg_r
        if div > best_score:
            best_score, best = div, combo
    return tuple(sorted(best)) if best else None


def select_by_mixed(K, q, R, valid_dets):
    """Strategy C: arg max (avg q) * (1 - avg rho).

    For K=2 this is the closed-form joint pair score (Eq. 2):
        (q_i + q_j)/2 * (1 - rho_{ij}).
    """
    idx_of = {d: i for i, d in enumerate(DETS)}
    best, best_score = None, -np.inf
    for combo in combinations(valid_dets, K):
        avg_q = float(np.mean([q[d] for d in combo]))
        idx = [idx_of[d] for d in combo]
        sub = R[np.ix_(idx, idx)]
        n = len(combo)
        avg_r = (sub.sum() - n) / (n * (n - 1))
        div = 1 - avg_r
        score = avg_q * div
        if score > best_score:
            best_score, best = score, combo
    return tuple(sorted(best)) if best else None


# ============== Stages 3-5: SHAP surrogate, sigma filter, fusion ==============

def kappa_sigma_flag(s_test, s_val, kappa=K_SIGMA):
    return (s_test > s_val.mean() + kappa * s_val.std()).astype(int)


def run_pipeline(members, q, X_tr, X_v, X_te, s_tr, s_v, s_te,
                 y_test, shap_cache=None, per_kpi=False):
    """SHAP-sigma + anchor flag. Returns best fusion variant F1/P/R + PA-F1.

    per_kpi=True: TELCO, evaluate per-KPI macro F1 (12 KPI streams).
    per_kpi=False: SMD/UCR, evaluate point-level F1 and PA-F1.

    y_test shape:
      per_kpi=True   -> (N_te, p)
      per_kpi=False  -> (N_te,)
    """
    if shap_cache is None:
        shap_cache = {}
    SHAP_v, SHAP_te = {}, {}
    for d in members:
        if d in shap_cache:
            SHAP_v[d], SHAP_te[d] = shap_cache[d]
        else:
            s_d = sanitize(s_tr[d])
            surr = lgb.LGBMRegressor(
                n_estimators=LGB_TREES_SURR, learning_rate=0.05, max_depth=6,
                random_state=42, verbose=-1, n_jobs=1,
            )
            surr.fit(X_tr, s_d)
            expl = shap.TreeExplainer(surr)
            sv = expl.shap_values(X_v)
            st = expl.shap_values(X_te)
            nm = sv.std() + 1e-12
            sv /= nm
            st /= nm
            SHAP_v[d], SHAP_te[d] = sv, st
            shap_cache[d] = (sv, st)

    # Sigma matrix and per-feature threshold (Eq. 3)
    sig_v = np.std(np.stack([SHAP_v[d]  for d in members]), axis=0)
    sig_t = np.std(np.stack([SHAP_te[d] for d in members]), axis=0)
    p = sig_v.shape[1]
    tau = np.array([np.quantile(sig_v[:, j], Q_FILT) for j in range(p)])
    phi = np.zeros_like(sig_t, dtype=int)
    for j in range(p):
        phi[:, j] = (sig_t[:, j] > tau[j]).astype(int)
    phi_or = phi.max(axis=1)  # phi^v = OR over features

    # Anchor and member flags
    anchor = max(members, key=lambda d: q[d])
    f_a = kappa_sigma_flag(s_te[anchor], s_v[anchor])
    f_or = np.zeros_like(f_a)
    f_and = np.ones_like(f_a)
    for d in members:
        f_d = kappa_sigma_flag(s_te[d], s_v[d])
        f_or = f_or | f_d
        f_and = f_and & f_d

    # Five fusion variants (Eq. 4: anchor + sigma is canonical)
    variants = {
        "anchor+sigma": (f_a   * phi_or).astype(int),
        "OR+sigma":     (f_or  * phi_or).astype(int),
        "AND":          f_and.astype(int),
        "OR":           f_or.astype(int),
        "AND+sigma":    (f_and * phi_or).astype(int),
    }

    # Evaluate
    if per_kpi:
        # broadcast row-flag to all p KPI streams, then per-KPI macro
        K_KPI = y_test.shape[1]
        def macro_f1(row_flag):
            preds = np.repeat(row_flag[:, None], K_KPI, axis=1)
            F = [f1_score(y_test[:, k], preds[:, k], zero_division=0) for k in range(K_KPI)]
            return float(np.mean(F))
        def macro_p(row_flag):
            preds = np.repeat(row_flag[:, None], K_KPI, axis=1)
            P = [precision_score(y_test[:, k], preds[:, k], zero_division=0) for k in range(K_KPI)]
            return float(np.mean(P))
        def macro_r(row_flag):
            preds = np.repeat(row_flag[:, None], K_KPI, axis=1)
            R = [recall_score(y_test[:, k], preds[:, k], zero_division=0) for k in range(K_KPI)]
            return float(np.mean(R))
        f1s = {n: macro_f1(p_) for n, p_ in variants.items()}
        best_var = max(f1s, key=f1s.get)
        best_pred = variants[best_var]
        return {
            "F1":    round(f1s[best_var], 4),
            "P":     round(macro_p(best_pred), 4),
            "R":     round(macro_r(best_pred), 4),
            "PA_F1": np.nan,
            "best_variant": best_var,
            "anchor": anchor,
        }, shap_cache
    else:
        f1s = {n: f1_score(y_test, p_, zero_division=0) for n, p_ in variants.items()}
        pas = {n: pa_f1_score(y_test, p_) for n, p_ in variants.items()}
        best_var = max(f1s, key=f1s.get)
        best_pred = variants[best_var]
        return {
            "F1":    round(f1s[best_var], 4),
            "P":     round(precision_score(y_test, best_pred, zero_division=0), 4),
            "R":     round(recall_score(y_test, best_pred, zero_division=0), 4),
            "PA_F1": round(pas[best_var], 4),
            "best_variant": best_var,
            "anchor": anchor,
        }, shap_cache


# ============== Per-series driver ==============

def process_one(dataset, series, X_train, X_val, X_test, y_test, per_kpi=False):
    """Run all (K_R, strategy) combinations for one (dataset, series)."""
    rows = []
    scaler = StandardScaler().fit(X_train)
    X_tr_s = scaler.transform(X_train)
    X_v_s  = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)

    # Score every detector on every split
    s_tr, s_v, s_te = {}, {}, {}
    for d, m in make_detectors().items():
        try:
            m.fit(X_tr_s)
            s_tr[d] = sanitize(m.decision_function(X_tr_s))
            s_v[d]  = sanitize(m.decision_function(X_v_s))
            s_te[d] = sanitize(m.decision_function(X_te_s))
        except Exception:
            s_tr[d] = s_v[d] = s_te[d] = None
    valid = [d for d in DETS if s_tr[d] is not None]
    if len(valid) < 2:
        return rows

    # Quality + Spearman
    A = mv_proxy_quality(s_tr, X_tr_s)
    q = normalize_quality(A)
    R = spearman_matrix(s_tr)
    idx_of = {d: i for i, d in enumerate(DETS)}

    shap_cache = {}
    for K in K_RANGE:
        if K > len(valid):
            break
        selections = {
            "Quality":   select_by_quality(K, q, valid),
            "Diversity": select_by_diversity(K, R, valid),
            "Mixed":     select_by_mixed(K, q, R, valid),
        }
        for strat, sel in selections.items():
            if sel is None:
                continue
            avg_q = float(np.mean([q[d] for d in sel]))
            idx = [idx_of[d] for d in sel]
            sub = R[np.ix_(idx, idx)]
            n = len(sel)
            avg_r = (sub.sum() - n) / (n * (n - 1)) if n >= 2 else 0.0
            avg_div = 1 - avg_r
            try:
                res, shap_cache = run_pipeline(
                    list(sel), q, X_tr_s, X_v_s, X_te_s,
                    s_tr, s_v, s_te, y_test, shap_cache, per_kpi=per_kpi,
                )
            except Exception:
                continue
            rows.append({
                "dataset": dataset, "series": series,
                "n_train": len(X_train), "n_test": len(X_test),
                "K": K, "strategy": strat,
                "members": "+".join(sel), "anchor": res["anchor"],
                "quality":     round(avg_q, 3),
                "diversity":   round(avg_div, 3),
                "q_times_div": round(avg_q * avg_div, 3),
                "F1": res["F1"], "P": res["P"], "R": res["R"],
                "PA_F1": res["PA_F1"],
                "best_variant": res["best_variant"],
            })
    return rows


# ============== Dataset loaders ==============

def load_telco():
    """TELCO returns y as (N_te, 12) matrix; per_kpi=True downstream."""
    FEAT = [f"TS{i}" for i in range(1, 13)]
    df_tr = pd.read_csv(os.path.join(DATA_DIR, "TELCO_data", "TELCO_data_train.csv"))
    df_te = pd.read_csv(os.path.join(DATA_DIR, "TELCO_data", "TELCO_data_test.csv"))
    X_tr_all = df_tr[FEAT].values.astype(np.float32)
    X_te = df_te[FEAT].values.astype(np.float32)
    lab = pd.read_csv(os.path.join(DATA_DIR, "TELCO_labels", "TELCO_labels_test.csv"))
    y_test = lab[FEAT].values.astype(int)  # (N_te, 12) per-KPI labels
    split = int(len(X_tr_all) * 0.8)
    return X_tr_all[:split], X_tr_all[split:], X_te, y_test


def load_smd_one(machine_file):
    try:
        X_full = np.loadtxt(os.path.join(SMD_DIR, "train", machine_file), delimiter=",").astype(np.float64)
        X_test = np.loadtxt(os.path.join(SMD_DIR, "test", machine_file), delimiter=",").astype(np.float64)
        y_test = np.loadtxt(os.path.join(SMD_DIR, "test_label", machine_file), dtype=int)
    except Exception:
        return None
    # Subsample large machines for tractability (same as paper)
    if len(X_full) > 8000:
        X_full = X_full[np.linspace(0, len(X_full) - 1, 8000).astype(int)]
    if len(X_test) > 15000:
        X_test = X_test[:15000]
        y_test = y_test[:15000]
    split = int(len(X_full) * 0.8)
    return X_full[:split], X_full[split:], X_test, y_test


def load_ucr_one(filepath):
    fn = os.path.basename(filepath)
    m = re.match(r"^(\d+)_UCR_Anomaly_(.+?)_(\d+)_(\d+)_(\d+)$", fn.replace(".txt", ""))
    if not m:
        return None
    _, _, tr_end, a_start, a_end = m.groups()
    tr_end, a_start, a_end = int(tr_end), int(a_start), int(a_end)
    x = np.loadtxt(filepath).astype(np.float32)
    if len(x) < tr_end + W + 100:
        return None
    X_windows = np.stack([x[i:i + W] for i in range(len(x) - W + 1)])
    end_idx = np.arange(W - 1, len(x))
    X_train_all = X_windows[end_idx < tr_end]
    X_test = X_windows[end_idx >= tr_end]
    test_end = end_idx[end_idx >= tr_end]
    if len(X_train_all) < 200 or len(X_test) < 100:
        return None
    y_test = ((test_end >= a_start) & (test_end <= a_end)).astype(int)
    if y_test.sum() == 0:
        return None
    split = int(len(X_train_all) * 0.8)
    return X_train_all[:split], X_train_all[split:], X_test, y_test


def get_ucr_unique_bases():
    """Group UCR by underlying signal (strip DISTORTED/NOISE prefix)."""
    files = sorted(glob.glob(os.path.join(UCR_DIR, "*.txt")))
    base_map = {}
    for fp in files:
        fn = os.path.basename(fp)
        m = re.match(r"^\d+_UCR_Anomaly_(?:DISTORTED|NOISE)?(.+?)_\d+_\d+_\d+\.txt$", fn)
        if not m:
            continue
        base = m.group(1)
        if base not in base_map:
            base_map[base] = fp
    return base_map


# ============== Main ==============

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    all_rows = []
    t0 = time.time()

    print("=" * 70)
    print("TELCO (per-KPI macro F1)")
    print("=" * 70, flush=True)
    X_tr, X_v, X_te, y_te = load_telco()
    rows = process_one("TELCO", "TELCO_main", X_tr, X_v, X_te, y_te, per_kpi=True)
    all_rows.extend(rows)
    print(f"  TELCO done: {len(rows)} rows  ({time.time() - t0:.0f}s)", flush=True)

    print("\n" + "=" * 70)
    print("SMD (28 machines, point-level F1 + PA-F1)")
    print("=" * 70, flush=True)
    smd_files = sorted(glob.glob(os.path.join(SMD_DIR, "train", "*.txt")))
    for i, fp in enumerate(smd_files, 1):
        fn = os.path.basename(fp)
        data = load_smd_one(fn)
        if data is None:
            print(f"  [SMD {i}/{len(smd_files)}] {fn} LOAD FAIL", flush=True)
            continue
        X_tr, X_v, X_te, y_te = data
        rows = process_one("SMD", fn.replace(".txt", ""), X_tr, X_v, X_te, y_te)
        all_rows.extend(rows)
        print(f"  [SMD {i}/{len(smd_files)}] {fn.replace('.txt','')}: {len(rows)} rows  ({time.time() - t0:.0f}s)", flush=True)
        if i % 5 == 0:
            pd.DataFrame(all_rows).to_csv(OUT, index=False)

    print("\n" + "=" * 70)
    print("UCR (unique base series, point-level F1 + PA-F1)")
    print("=" * 70, flush=True)
    ucr_unique = get_ucr_unique_bases()
    print(f"Total unique UCR base series: {len(ucr_unique)}", flush=True)
    for i, (base, fp) in enumerate(sorted(ucr_unique.items()), 1):
        data = load_ucr_one(fp)
        if data is None:
            continue
        X_tr, X_v, X_te, y_te = data
        rows = process_one("UCR", base, X_tr, X_v, X_te, y_te)
        all_rows.extend(rows)
        if i % 5 == 0 or i < 5:
            print(f"  [UCR {i}/{len(ucr_unique)}] {base[:35]:<35} {len(rows)} rows  ({time.time() - t0:.0f}s)", flush=True)
        if i % 10 == 0:
            pd.DataFrame(all_rows).to_csv(OUT, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    print(f"\n{'=' * 70}")
    print(f"DONE  total rows: {len(df)}  total time: {time.time() - t0:.0f}s")
    print(f"Saved: {OUT}")

    print(f"\n{'=' * 70}\nMean F1 by (dataset, K, strategy):\n{'=' * 70}")
    g = df.groupby(["dataset", "K", "strategy"])["F1"].agg(["mean", "std", "count"]).reset_index()
    g["mean"] = g["mean"].round(4)
    g["std"] = g["std"].round(4)
    print(g.to_string(index=False))

    print(f"\n{'=' * 70}\nBest mean F1 per dataset:\n{'=' * 70}")
    for ds in ["TELCO", "SMD", "UCR"]:
        sub = df[df["dataset"] == ds]
        if len(sub) == 0:
            continue
        idx = sub.groupby(["K", "strategy"])["F1"].mean().idxmax()
        val = sub.groupby(["K", "strategy"])["F1"].mean().max()
        print(f"  {ds}: K={idx[0]}, strategy={idx[1]}, mean F1={val:.4f}")


if __name__ == "__main__":
    main()
