"""
Singleton (per-detector standalone) F1 baselines on TELCO, SMD, UCR.

Same protocol as shap_sigma_pipeline.py: 8 detectors, k-sigma flag (kappa=3),
no SHAP, no ensemble. Reports point-level F1, PA-F1 (SMD/UCR), and per-KPI
macro F1 (TELCO).

Output: results/singletons_cross_dataset.csv
"""
import os, glob, re, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

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

DETS    = ["IF", "LOF", "ECOD", "COPOD", "KNN", "PCA", "GMM", "HBOS"]
K_SIGMA = 3.0
W       = 100

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SMD_DIR  = os.path.join(DATA_DIR, "SMD_data")
UCR_DIR  = os.path.join(DATA_DIR, "UCR_Anomaly_FullData")
OUT      = os.path.join(ROOT, "results", "singletons_cross_dataset.csv")


def sanitize(s):
    s = np.asarray(s, dtype=np.float64).copy()
    bad = ~np.isfinite(s)
    if bad.any():
        med = np.nanmedian(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0
        s[bad] = med
    return s


def make_detectors():
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
    y_pred = y_pred.copy().astype(int)
    y_true = y_true.astype(int)
    adj = y_pred.copy()
    in_region = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_region:
            in_region = True
            start = i
        elif y_true[i] == 0 and in_region:
            in_region = False
            if y_pred[start:i].sum() > 0:
                adj[start:i] = 1
    if in_region and y_pred[start:].sum() > 0:
        adj[start:] = 1
    return adj


def pa_f1_score(y, p):
    return f1_score(y, point_adjust(y, p), zero_division=0)


def kappa_sigma_flag(s_test, s_val):
    return (s_test > s_val.mean() + K_SIGMA * s_val.std()).astype(int)


def run_telco():
    FEAT = [f"TS{i}" for i in range(1, 13)]
    df_tr = pd.read_csv(os.path.join(DATA_DIR, "TELCO_data", "TELCO_data_train.csv"))
    df_te = pd.read_csv(os.path.join(DATA_DIR, "TELCO_data", "TELCO_data_test.csv"))
    X_full = df_tr[FEAT].values.astype(np.float32)
    X_test = df_te[FEAT].values.astype(np.float32)
    lab = pd.read_csv(os.path.join(DATA_DIR, "TELCO_labels", "TELCO_labels_test.csv"))
    y_pf = lab[FEAT].values.astype(int)
    y_point = y_pf.max(axis=1)
    split = int(len(X_full) * 0.8)
    X_tr, X_v = X_full[:split], X_full[split:]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_v_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_v), scaler.transform(X_test)
    rows = []
    for d, m in make_detectors().items():
        try:
            m.fit(X_tr_s)
            s_v = sanitize(m.decision_function(X_v_s))
            s_t = sanitize(m.decision_function(X_te_s))
            flag = kappa_sigma_flag(s_t, s_v)
            pred_pf = np.repeat(flag[:, None], 12, axis=1)
            f1_kpi = np.mean([f1_score(y_pf[:, j], pred_pf[:, j], zero_division=0) for j in range(12)])
            p_kpi  = np.mean([precision_score(y_pf[:, j], pred_pf[:, j], zero_division=0) for j in range(12)])
            r_kpi  = np.mean([recall_score(y_pf[:, j], pred_pf[:, j], zero_division=0) for j in range(12)])
            rows.append({
                "dataset": "TELCO", "series": "TELCO_main", "detector": d,
                "F1_per_KPI_macro": round(f1_kpi, 4),
                "P_per_KPI": round(p_kpi, 4),
                "R_per_KPI": round(r_kpi, 4),
                "F1_point_level": round(f1_score(y_point, flag, zero_division=0), 4),
                "P_point": round(precision_score(y_point, flag, zero_division=0), 4),
                "R_point": round(recall_score(y_point, flag, zero_division=0), 4),
                "PA_F1": float("nan"),
            })
        except Exception as e:
            print(f"  TELCO {d} FAIL: {e}")
    return rows


def run_smd_one(machine_file):
    try:
        X_full = np.loadtxt(os.path.join(SMD_DIR, "train", machine_file), delimiter=",").astype(np.float64)
        X_test = np.loadtxt(os.path.join(SMD_DIR, "test", machine_file), delimiter=",").astype(np.float64)
        y_test = np.loadtxt(os.path.join(SMD_DIR, "test_label", machine_file), dtype=int)
    except Exception:
        return []
    if len(X_full) > 8000:
        X_full = X_full[np.linspace(0, len(X_full) - 1, 8000).astype(int)]
    if len(X_test) > 15000:
        X_test = X_test[:15000]
        y_test = y_test[:15000]
    split = int(len(X_full) * 0.8)
    X_tr, X_v = X_full[:split], X_full[split:]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_v_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_v), scaler.transform(X_test)
    rows = []
    for d, m in make_detectors().items():
        try:
            m.fit(X_tr_s)
            s_v = sanitize(m.decision_function(X_v_s))
            s_t = sanitize(m.decision_function(X_te_s))
            flag = kappa_sigma_flag(s_t, s_v)
            rows.append({
                "dataset": "SMD", "series": machine_file.replace(".txt", ""), "detector": d,
                "F1_per_KPI_macro": float("nan"), "P_per_KPI": float("nan"), "R_per_KPI": float("nan"),
                "F1_point_level": round(f1_score(y_test, flag, zero_division=0), 4),
                "P_point": round(precision_score(y_test, flag, zero_division=0), 4),
                "R_point": round(recall_score(y_test, flag, zero_division=0), 4),
                "PA_F1": round(pa_f1_score(y_test, flag), 4),
            })
        except Exception:
            continue
    return rows


def run_ucr_one(filepath):
    fn = os.path.basename(filepath)
    m = re.match(r"^(\d+)_UCR_Anomaly_(.+?)_(\d+)_(\d+)_(\d+)$", fn.replace(".txt", ""))
    if not m:
        return []
    _, name, tr_end, a_start, a_end = m.groups()
    tr_end, a_start, a_end = int(tr_end), int(a_start), int(a_end)
    x = np.loadtxt(filepath).astype(np.float32)
    if len(x) < tr_end + W + 100:
        return []
    X_w = np.stack([x[i:i + W] for i in range(len(x) - W + 1)])
    end_idx = np.arange(W - 1, len(x))
    X_full = X_w[end_idx < tr_end]
    X_test = X_w[end_idx >= tr_end]
    test_end = end_idx[end_idx >= tr_end]
    if len(X_full) < 200 or len(X_test) < 100:
        return []
    y_test = ((test_end >= a_start) & (test_end <= a_end)).astype(int)
    if y_test.sum() == 0:
        return []
    split = int(len(X_full) * 0.8)
    X_tr, X_v = X_full[:split], X_full[split:]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_v_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_v), scaler.transform(X_test)
    base = re.sub(r"^(DISTORTED|NOISE)", "", name)
    rows = []
    for d, mdl in make_detectors().items():
        try:
            mdl.fit(X_tr_s)
            s_v = sanitize(mdl.decision_function(X_v_s))
            s_t = sanitize(mdl.decision_function(X_te_s))
            flag = kappa_sigma_flag(s_t, s_v)
            rows.append({
                "dataset": "UCR", "series": base, "detector": d,
                "F1_per_KPI_macro": float("nan"), "P_per_KPI": float("nan"), "R_per_KPI": float("nan"),
                "F1_point_level": round(f1_score(y_test, flag, zero_division=0), 4),
                "P_point": round(precision_score(y_test, flag, zero_division=0), 4),
                "R_point": round(recall_score(y_test, flag, zero_division=0), 4),
                "PA_F1": round(pa_f1_score(y_test, flag), 4),
            })
        except Exception:
            continue
    return rows


def get_ucr_unique_bases():
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


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    all_rows = []
    t0 = time.time()

    print("TELCO...", flush=True)
    all_rows.extend(run_telco())

    print("SMD (28 machines)...", flush=True)
    smd_files = sorted(glob.glob(os.path.join(SMD_DIR, "train", "*.txt")))
    for i, fp in enumerate(smd_files, 1):
        all_rows.extend(run_smd_one(os.path.basename(fp)))
        if i % 5 == 0:
            print(f"  [SMD {i}/{len(smd_files)}] ({time.time() - t0:.0f}s)", flush=True)

    print("UCR (100 unique base series)...", flush=True)
    ucr_uniq = get_ucr_unique_bases()
    for i, (base, fp) in enumerate(sorted(ucr_uniq.items()), 1):
        all_rows.extend(run_ucr_one(fp))
        if i % 10 == 0:
            print(f"  [UCR {i}/{len(ucr_uniq)}] ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")

    print(f"\n{'=' * 80}\nSINGLETON F1 SUMMARY\n{'=' * 80}")
    for ds in ["TELCO", "SMD", "UCR"]:
        sub = df[df["dataset"] == ds]
        if len(sub) == 0:
            continue
        n_series = sub["series"].nunique()
        col = "F1_per_KPI_macro" if ds == "TELCO" else "F1_point_level"
        print(f"\n{ds} ({n_series} series):")
        g = sub.groupby("detector")[col].agg(["mean", "std", "count"]).round(4)
        print(g.sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main()
