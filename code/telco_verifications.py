"""
TELCO verifications with the new canonical pipeline:
  (1) Truncated 25,143-timestamp window F1 (matches Malacarne2026 evaluation).
  (2) Filter quantile sweep q in {0.5, 0.75, 0.9} on the ECOD+HBOS pair.
  (3) All 28 candidate pairs at K_R=2: joint pair score vs pipeline F1
      (input data for Fig. 3 fig_jointpair_telco).

Outputs:
  results/telco_truncated_window.csv
  results/telco_quantile_sweep.csv
  results/telco_all_k_combinations.csv  (regenerated)
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shap_sigma_pipeline as sp


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

DETS = sp.DETS


def telco_pair_pipeline(anchor, partner, X_tr, X_v, X_te, y_pf,
                        s_tr, s_v, s_te, q_filt=0.75):
    """Run SHAP-sigma pipeline for a (anchor, partner) pair on TELCO,
    return F1 (per-KPI macro) of the canonical anchor+sigma variant."""
    import shap, lightgbm as lgb
    K_KPI = y_pf.shape[1]
    SHAP_v, SHAP_te = {}, {}
    for d in (anchor, partner):
        surr = lgb.LGBMRegressor(
            n_estimators=sp.LGB_TREES_SURR, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=sp.SEED, verbose=-1, n_jobs=1,
        )
        surr.fit(X_tr, sp.sanitize(s_tr[d]))
        expl = shap.TreeExplainer(surr)
        sv = expl.shap_values(X_v); st = expl.shap_values(X_te)
        nm = sv.std() + 1e-12
        sv /= nm; st /= nm
        SHAP_v[d], SHAP_te[d] = sv, st

    sig_v = np.std(np.stack([SHAP_v[anchor], SHAP_v[partner]]), axis=0)
    sig_t = np.std(np.stack([SHAP_te[anchor], SHAP_te[partner]]), axis=0)
    p = sig_v.shape[1]
    tau = np.array([np.quantile(sig_v[:, j], q_filt) for j in range(p)])
    phi = np.zeros_like(sig_t, dtype=int)
    for j in range(p):
        phi[:, j] = (sig_t[:, j] > tau[j]).astype(int)

    f_a = sp.kappa_sigma_flag(s_te[anchor], s_v[anchor])
    f_a_b = np.repeat(f_a[:, None], K_KPI, axis=1)
    pred = (f_a_b * phi).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    F1 = np.mean([f1_score(y_pf[:, k], pred[:, k], zero_division=0) for k in range(K_KPI)])
    P  = np.mean([precision_score(y_pf[:, k], pred[:, k], zero_division=0) for k in range(K_KPI)])
    R  = np.mean([recall_score(y_pf[:, k], pred[:, k], zero_division=0) for k in range(K_KPI)])
    return float(F1), float(P), float(R)


def main():
    # Load TELCO once
    X_tr, X_v, X_te, y_pf = sp.load_telco()
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_v_s, X_te_s = sc.transform(X_tr), sc.transform(X_v), sc.transform(X_te)

    # Detector scores
    s_tr, s_v, s_te = {}, {}, {}
    for d, m in sp.make_detectors().items():
        m.fit(X_tr_s)
        s_tr[d] = sp.sanitize(m.decision_function(X_tr_s))
        s_v[d]  = sp.sanitize(m.decision_function(X_v_s))
        s_te[d] = sp.sanitize(m.decision_function(X_te_s))

    # Quality + Spearman
    A = sp.mv_proxy_quality(s_tr, X_tr_s)
    q = sp.normalize_quality(A)
    R = sp.spearman_matrix(s_tr)
    idx_of = {d: i for i, d in enumerate(DETS)}

    # ============== (1) Truncated window ==============
    print("=" * 70)
    print("(1) TELCO truncated 25,143-timestamp window")
    print("=" * 70)
    n_trunc = 25143
    # Slice test data and labels
    X_te_trunc = X_te_s[:n_trunc]
    y_pf_trunc = y_pf[:n_trunc]
    s_te_trunc = {d: s_te[d][:n_trunc] for d in DETS}
    # Run pair pipeline on ECOD+HBOS with q=0.75
    f1, p, r = telco_pair_pipeline(
        "ECOD", "HBOS", X_tr_s, X_v_s, X_te_trunc, y_pf_trunc,
        s_tr, s_v, s_te_trunc, q_filt=0.75,
    )
    print(f"  ECOD+HBOS truncated window: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    pd.DataFrame([{"window": "truncated_25143", "anchor": "ECOD",
                   "partner": "HBOS", "F1": round(f1, 4),
                   "P": round(p, 4), "R": round(r, 4)}]).to_csv(
        os.path.join(RES, "telco_truncated_window.csv"), index=False)

    # ============== (2) Quantile sweep q in {0.5, 0.75, 0.9} ==============
    print("\n" + "=" * 70)
    print("(2) Filter quantile sweep on ECOD+HBOS (full window)")
    print("=" * 70)
    rows = []
    for q_filt in [0.5, 0.75, 0.9]:
        f1, p, r = telco_pair_pipeline(
            "ECOD", "HBOS", X_tr_s, X_v_s, X_te_s, y_pf,
            s_tr, s_v, s_te, q_filt=q_filt,
        )
        print(f"  q={q_filt}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
        rows.append({"q": q_filt, "F1": round(f1, 4), "P": round(p, 4), "R": round(r, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "telco_quantile_sweep.csv"), index=False)

    # ============== (3) All 28 candidate pairs at K_R=2 ==============
    print("\n" + "=" * 70)
    print("(3) All 28 candidate pairs at K_R=2 (input for Fig. 3)")
    print("=" * 70)
    rows = []
    for i, j in combinations(range(len(DETS)), 2):
        di, dj = DETS[i], DETS[j]
        avg_q = (q[di] + q[dj]) / 2
        r_ij = R[i, j]
        div = 1 - r_ij
        joint = avg_q * div
        f1, p, r = telco_pair_pipeline(
            di, dj, X_tr_s, X_v_s, X_te_s, y_pf,
            s_tr, s_v, s_te, q_filt=0.75,
        )
        rows.append({
            "set": f"{di}+{dj}", "K": 2,
            "anchor": di if q[di] >= q[dj] else dj,
            "partner": dj if q[di] >= q[dj] else di,
            "quality": round(avg_q, 4),
            "diversity": round(div, 4),
            "q_times_div": round(joint, 4),
            "F1": round(f1, 4), "P": round(p, 4), "R": round(r, 4),
        })
        print(f"  {di}+{dj}: q={avg_q:.3f}, div={div:.3f}, joint={joint:.3f}, F1={f1:.4f}")
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    df.to_csv(os.path.join(RES, "telco_all_k_combinations.csv"), index=False)
    print(f"\n  Saved {RES}/telco_all_k_combinations.csv ({len(df)} pairs)")
    print(f"  Top by F1:    {df.iloc[0]['set']} (F1={df.iloc[0]['F1']:.4f})")
    print(f"  Top by joint: {df.sort_values('q_times_div', ascending=False).iloc[0]['set']}")


if __name__ == "__main__":
    main()
