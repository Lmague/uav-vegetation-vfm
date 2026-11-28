#!/usr/bin/env python3
"""Réplication COMPLÈTE 100 % DÉPLOYABLE (8 vraies voisines, zéro masque d'annotation)
de la table 8 backbones (`multibackbone_smoothing_fullgraph.csv`), `peft_pairs_fullgraph.csv`
et `strata_fullgraph.json`.

Même protocole exact que `context_multibackbone` :
  - 8 backbones (5 gelés + 3 LoRA seed 0)
  - Graphe COMPLET : 80 088 tuiles annotées + 40 076 tuiles auxiliaires = 120 164 nœuds
    (degré moyen test = 8.00 exactement — zéro fuite de masque d'annotation)
  - Grilles étendues : beta ∈ {0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 16}, iters ∈ {1, 2, 3, 5, 8, 12}
  - Bras par backbone : R0, R0+MF (uniforme), R0+ICM, R0+MF_Wconf (pondéré par confiance^4),
    R3 (moyenne 3×3 sur les 8 vraies voisines), R3+MF, R3+ICM
  - Sondes canoniques : StandardScaler + make_canonical_lr (lbfgs, C ∈ {1e-4..10}), BLAS mono-thread
  - Bootstrap hiérarchique apparié (10 000 réplicats, ortho puis tuile, seed 7) pour tous les deltas
    + les 3 paires PEFT vs gelé.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import make_canonical_lr

STRIDE = 112
K = 11
KEEP8 = [0, 2, 4, 5, 6, 8, 9, 10]
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
BETAS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0)
ITERS = (1, 2, 3, 5, 8, 12)
OFFS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

BACKBONES = [
    "dinov3_vitb16_lvd",
    "dinov3_vith16plus_lvd",
    "dinov3_vitl16_lvd",
    "dinov3_vitl16_lvd_lora_s0",
    "simdinov2_vitb16",
    "simdinov2_vitb16_lora_r8a16_s0",
    "simdinov2_vitl16",
    "simdinov2_vitl16_lora_s0",
]

PEFT_PAIRS = [
    ("dinov3_vitl16_lvd", "dinov3_vitl16_lvd_lora_s0", "DINOv3-L"),
    ("simdinov2_vitb16", "simdinov2_vitb16_lora_r8a16_s0", "SimDINOv2-B"),
    ("simdinov2_vitl16", "simdinov2_vitl16_lora_s0", "SimDINOv2-L"),
]


def f1_11(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    d = cm.sum(1) + cm.sum(0)
    f = np.where(d > 0, 2.0 * tp / np.maximum(d, 1.0), np.nan)
    return float(f[~np.isnan(f)].mean())


def f1_8(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    d = cm.sum(1) + cm.sum(0)
    f = np.where(d > 0, 2.0 * tp / np.maximum(d, 1.0), 0.0)
    return float(f[KEEP8].mean())


def build_adj(pos: list[tuple]) -> sp.csr_matrix:
    idx = {p: i for i, p in enumerate(pos)}
    rows, cols = [], []
    for i, (o, r, c) in enumerate(pos):
        for dr, dc in OFFS:
            j = idx.get((o, r + STRIDE * dr, c + STRIDE * dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
    n = len(pos)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))


def mean_features_full(X: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    A1 = A.copy()
    A1.setdiag(1.0)
    cnt = np.asarray(A1.sum(1)).ravel()
    return (np.asarray(A1 @ X) / cnt[:, None]).astype(np.float32)


def onehot(labels: np.ndarray) -> np.ndarray:
    O = np.zeros((len(labels), K), np.float32)
    O[np.arange(len(labels)), labels] = 1.0
    return O


def run_mf(logp: np.ndarray, W: sp.csr_matrix, beta: float, iters: int, init: np.ndarray) -> np.ndarray:
    Q = onehot(init)
    for _ in range(iters):
        Q = W @ Q
        rs = Q.sum(1, keepdims=True)
        Q = Q / np.maximum(rs, 1e-9)
        Z = logp + beta * Q
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        Q = (E / E.sum(1, keepdims=True)).astype(np.float32)
    return Q.argmax(1).astype(np.int64)


def run_icm(logp: np.ndarray, A: sp.csr_matrix, beta: float, iters: int, init: np.ndarray) -> np.ndarray:
    L = init.copy()
    for _ in range(iters):
        cnt = A @ onehot(L)
        L = np.argmax(logp + beta * cnt, axis=1).astype(np.int64)
    return L


def conf_weighted_adj(A: sp.csr_matrix, probs: np.ndarray, gamma: float = 4.0) -> sp.csr_matrix:
    """Pondère chaque arête i<-j par (max_c P_j)^gamma (zéro apprentissage, déployable)."""
    w_node = (probs.max(axis=1) ** gamma).astype(np.float32)
    W = A.multiply(w_node[None, :]).tocsr()
    return W


def eval_backbone(task: tuple) -> dict:
    bb, split_json, canonical_mb_dir, aux_dir, aux_index, out_dir = task
    t0 = time.time()
    tiles = json.load(open(split_json))["tiles"]
    aux_list = json.load(open(aux_index))

    X_lab = np.load(_os.path.join(canonical_mb_dir, bb, "features_tile.npy")).astype(np.float32)
    X_aux = np.load(_os.path.join(aux_dir, f"{bb}_aux_224.npy")).astype(np.float32)
    assert X_lab.shape[0] == len(tiles) and X_aux.shape[0] == len(aux_list)
    assert X_lab.shape[1] == X_aux.shape[1]

    n_lab = len(tiles)
    y_lab = np.array([t["label"] for t in tiles], dtype=np.int64)
    s_lab = np.array([t["split"] for t in tiles])
    s_aux = np.array([a["split"] for a in aux_list])
    s_full = np.concatenate([s_lab, s_aux])
    pos_full = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles] + [
        (a["ortho"], a["row"], a["col"]) for a in aux_list
    ]

    X_full = np.concatenate([X_lab, X_aux], axis=0)
    A = build_adj(pos_full)

    tr_lab = np.where(s_lab == "train")[0]
    va_lab = np.where(s_lab == "val")[0]
    te_lab = np.where(s_lab == "test")[0]

    va_all = np.where(s_full == "val")[0]
    te_all = np.where(s_full == "test")[0]
    va_mask = va_all < n_lab
    te_mask = te_all < n_lab

    A_va = A[va_all][:, va_all].tocsr()
    A_te = A[te_all][:, te_all].tocsr()
    mean_deg_te = float(np.asarray(A_te.sum(1)).ravel()[te_mask].mean())
    assert abs(mean_deg_te - 8.0) < 0.05, f"Unexpected degree {mean_deg_te}"

    Xm_full = mean_features_full(X_full, A)

    preds_dir = _os.path.join(out_dir, bb, "preds")
    _os.makedirs(preds_dir, exist_ok=True)
    rec = {"backbone": bb, "dim": int(X_full.shape[1]), "mean_deg_test": round(mean_deg_te, 3)}

    for arm_name, Fmat in (("R0", X_full), ("R3", Xm_full)):
        sc = StandardScaler().fit(Fmat[tr_lab])
        Ftr = sc.transform(Fmat[tr_lab]).astype(np.float32)
        Fva = sc.transform(Fmat[va_all]).astype(np.float32)
        Fte = sc.transform(Fmat[te_all]).astype(np.float32)

        best = (-1.0, None, None)
        for C in C_GRID:
            clf = make_canonical_lr(C=C, max_iter=2000, random_state=42)
            clf.fit(Ftr, y_lab[tr_lab])
            v = f1_11(y_lab[va_lab], clf.predict(Fva[va_mask]))
            if v >= best[0]:
                best = (v, C, clf)
        v_base, best_C, clf = best

        P_va = clf.predict_proba(Fva).astype(np.float32)
        P_te = clf.predict_proba(Fte).astype(np.float32)
        logp_va = np.log(np.maximum(P_va, 1e-12)).astype(np.float32)
        logp_te = np.log(np.maximum(P_te, 1e-12)).astype(np.float32)
        init_va = P_va.argmax(1)
        init_te = P_te.argmax(1)

        te_base = f1_11(y_lab[te_lab], init_te[te_mask])
        te_base_8 = f1_8(y_lab[te_lab], init_te[te_mask])
        np.save(_os.path.join(preds_dir, f"{arm_name}_base.npy"), init_te[te_mask].astype(np.uint8))

        # Uniform Mean-Field on full 8-neighbor graph
        best_mf = (-1.0, None, None)
        for b in BETAS:
            for it in ITERS:
                Lv = run_mf(logp_va, A_va, b, it, init_va)
                v = f1_11(y_lab[va_lab], Lv[va_mask])
                if v > best_mf[0]:
                    best_mf = (v, b, it)
        Lt_mf = run_mf(logp_te, A_te, best_mf[1], best_mf[2], init_te)[te_mask]
        np.save(_os.path.join(preds_dir, f"{arm_name}_MF.npy"), Lt_mf.astype(np.uint8))

        # ICM on full 8-neighbor graph
        best_icm = (-1.0, None, None)
        for b in BETAS:
            for it in ITERS:
                Lv = run_icm(logp_va, A_va, b, it, init_va)
                v = f1_11(y_lab[va_lab], Lv[va_mask])
                if v > best_icm[0]:
                    best_icm = (v, b, it)
        Lt_icm = run_icm(logp_te, A_te, best_icm[1], best_icm[2], init_te)[te_mask]
        np.save(_os.path.join(preds_dir, f"{arm_name}_ICM.npy"), Lt_icm.astype(np.uint8))

        # Confidence-weighted Mean-Field (gamma=4, deployable)
        W_va = conf_weighted_adj(A_va, P_va, gamma=4.0)
        W_te = conf_weighted_adj(A_te, P_te, gamma=4.0)
        best_w = (-1.0, None, None)
        for b in BETAS:
            for it in ITERS:
                Lv = run_mf(logp_va, W_va, b, it, init_va)
                v = f1_11(y_lab[va_lab], Lv[va_mask])
                if v > best_w[0]:
                    best_w = (v, b, it)
        Lt_w = run_mf(logp_te, W_te, best_w[1], best_w[2], init_te)[te_mask]
        np.save(_os.path.join(preds_dir, f"{arm_name}_MF_Wconf.npy"), Lt_w.astype(np.uint8))

        rec[arm_name] = {
            "C": best_C,
            "val": round(v_base, 6),
            "test": round(te_base, 6),
            "test_8": round(te_base_8, 6),
            "mf_beta": best_mf[1],
            "mf_iters": best_mf[2],
            "mf_val": round(best_mf[0], 6),
            "mf_test": round(f1_11(y_lab[te_lab], Lt_mf), 6),
            "mf_test_8": round(f1_8(y_lab[te_lab], Lt_mf), 6),
            "icm_beta": best_icm[1],
            "icm_iters": best_icm[2],
            "icm_test": round(f1_11(y_lab[te_lab], Lt_icm), 6),
            "wconf_beta": best_w[1],
            "wconf_iters": best_w[2],
            "wconf_val": round(best_w[0], 6),
            "wconf_test": round(f1_11(y_lab[te_lab], Lt_w), 6),
            "wconf_test_8": round(f1_8(y_lab[te_lab], Lt_w), 6),
        }

    rec["runtime_s"] = round(time.time() - t0, 1)
    with open(_os.path.join(out_dir, bb, "ablation_fullgraph.json"), "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def paired_boot(y: np.ndarray, ortho: np.ndarray, pa: np.ndarray, pb: np.ndarray, n_boot: int = 10000, seed: int = 7):
    names = sorted(set(ortho))
    groups = {o: np.where(ortho == o)[0] for o in names}
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        idx = np.concatenate([groups[o][rng.randint(0, len(groups[o]), len(groups[o]))] for o in pick])
        d[i] = f1_11(y[idx], pa[idx]) - f1_11(y[idx], pb[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    pt = f1_11(y, pa) - f1_11(y, pb)
    return float(pt), float(lo), float(hi), float(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--canonical-mb-dir", default=None)
    ap.add_argument("--aux-dir", default=None)
    ap.add_argument("--aux-index", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-only", action="store_true",
                    help="Recompute bootstrap CSVs from saved out-dir/<bb>/{ablation_fullgraph.json,preds/*.npy}.")
    args = ap.parse_args()

    _os.makedirs(args.out_dir, exist_ok=True)
    records = {}
    if args.boot_only:
        for bb in BACKBONES:
            with open(_os.path.join(args.out_dir, bb, "ablation_fullgraph.json")) as f:
                records[bb] = json.load(f)
        print(f"[mb-full] --boot-only: loaded {len(records)} saved backbone records", flush=True)
    else:
        for flag, val in (("--canonical-mb-dir", args.canonical_mb_dir),
                          ("--aux-dir", args.aux_dir),
                          ("--aux-index", args.aux_index)):
            if not val:
                ap.error(f"{flag} is required unless --boot-only is set")
        tasks = [
            (bb, args.split_json, args.canonical_mb_dir, args.aux_dir, args.aux_index, args.out_dir)
            for bb in BACKBONES
        ]
        print(f"[mb-full] 8 backbones on full 8-neighbour graph, workers={args.workers}", flush=True)

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(eval_backbone, t) for t in tasks]):
                r = fut.result()
                bb = r["backbone"]
                records[bb] = r
                print(
                    f"  {bb:30s} | R0={r['R0']['test']:.4f} R0+MF={r['R0']['mf_test']:.4f} "
                    f"R0+W={r['R0']['wconf_test']:.4f} | R3={r['R3']['test']:.4f} "
                    f"R3+MF={r['R3']['mf_test']:.4f} R3+W={r['R3']['wconf_test']:.4f} ({r['runtime_s']}s)",
                    flush=True,
                )

    # Paired hierarchical bootstrap (10,000 replicates, orthomosaic then tile)
    tiles = json.load(open(args.split_json))["tiles"]
    te = [t for t in tiles if t["split"] == "test"]
    y_te = np.array([t["label"] for t in te], dtype=np.int64)
    ortho_te = np.array([t["ortho"] for t in te])

    def load_p(bb: str, arm: str) -> np.ndarray:
        return np.load(_os.path.join(args.out_dir, bb, "preds", f"{arm}.npy"))

    csv_rows = []
    for bb in BACKBONES:
        r = records[bb]
        p_r0 = load_p(bb, "R0_base")
        p_r0mf = load_p(bb, "R0_MF")
        p_r0w = load_p(bb, "R0_MF_Wconf")
        p_r3 = load_p(bb, "R3_base")
        p_r3mf = load_p(bb, "R3_MF")
        p_r3w = load_p(bb, "R3_MF_Wconf")

        d_ctx, lo_ctx, hi_ctx, p_ctx = paired_boot(y_te, ortho_te, p_r3, p_r0, args.boot)
        d_m0, lo_m0, hi_m0, p_m0 = paired_boot(y_te, ortho_te, p_r0mf, p_r0, args.boot)
        d_w0, lo_w0, hi_w0, p_w0 = paired_boot(y_te, ortho_te, p_r0w, p_r0, args.boot)
        d_m3, lo_m3, hi_m3, p_m3 = paired_boot(y_te, ortho_te, p_r3mf, p_r3, args.boot)
        d_w3, lo_w3, hi_w3, p_w3 = paired_boot(y_te, ortho_te, p_r3w, p_r3, args.boot)
        d_r3w0, lo_r3w0, hi_r3w0, p_r3w0 = paired_boot(y_te, ortho_te, p_r3w, p_r0, args.boot)

        csv_rows.append({
            "backbone": bb,
            "R0": r["R0"]["test"],
            "R0_8cls": r["R0"]["test_8"],
            "R0_MF": r["R0"]["mf_test"],
            "R0_ICM": r["R0"]["icm_test"],
            "R0_Wconf": r["R0"]["wconf_test"],
            "R3": r["R3"]["test"],
            "R3_MF": r["R3"]["mf_test"],
            "R3_MF_8cls": r["R3"]["mf_test_8"],
            "R3_ICM": r["R3"]["icm_test"],
            "R3_Wconf": r["R3"]["wconf_test"],
            "R3_Wconf_8cls": r["R3"]["wconf_test_8"],
            "d_ctx": d_ctx, "lo_ctx": lo_ctx, "hi_ctx": hi_ctx, "p_ctx": p_ctx,
            "d_mf_R0": d_m0, "lo_mf_R0": lo_m0, "hi_mf_R0": hi_m0, "p_mf_R0": p_m0,
            "d_wconf_R0": d_w0, "lo_wconf_R0": lo_w0, "hi_wconf_R0": hi_w0, "p_wconf_R0": p_w0,
            "d_mf_R3": d_m3, "lo_mf_R3": lo_m3, "hi_mf_R3": hi_m3, "p_mf_R3": p_m3,
            "d_wconf_R3": d_w3, "lo_wconf_R3": lo_w3, "hi_wconf_R3": hi_w3, "p_wconf_R3": p_w3,
            "d_r3wconf_R0": d_r3w0, "lo_r3wconf_R0": lo_r3w0, "hi_r3wconf_R0": hi_r3w0, "p_r3wconf_R0": p_r3w0,
            "mf_beta_R0": r["R0"]["mf_beta"], "mf_iters_R0": r["R0"]["mf_iters"],
            "mf_beta_R3": r["R3"]["mf_beta"], "mf_iters_R3": r["R3"]["mf_iters"],
            "wconf_beta_R0": r["R0"]["wconf_beta"], "wconf_iters_R0": r["R0"]["wconf_iters"],
            "wconf_beta_R3": r["R3"]["wconf_beta"], "wconf_iters_R3": r["R3"]["wconf_iters"],
        })

    mb_csv = _os.path.join(args.out_dir, "multibackbone_smoothing_fullgraph.csv")
    with open(mb_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[mb-full] écrit {mb_csv}", flush=True)

    # Paires PEFT vs gelé
    peft_rows = []
    for frozen, peft, label in PEFT_PAIRS:
        pr0, fr0 = load_p(peft, "R0_base"), load_p(frozen, "R0_base")
        pm, fm = load_p(peft, "R3_MF"), load_p(frozen, "R3_MF")
        pw, fw = load_p(peft, "R0_MF_Wconf"), load_p(frozen, "R0_MF_Wconf")
        d0, l0, h0, p0 = paired_boot(y_te, ortho_te, pr0, fr0, args.boot)
        dm, lm, hm, pm_p = paired_boot(y_te, ortho_te, pm, fm, args.boot)
        dw, lw, hw, pw_p = paired_boot(y_te, ortho_te, pw, fw, args.boot)
        peft_rows.append({
            "pair": label, "frozen": frozen, "peft": peft,
            "frozen_R0": f1_11(y_te, fr0), "peft_R0": f1_11(y_te, pr0),
            "d_R0": d0, "lo_R0": l0, "hi_R0": h0, "p_R0": p0,
            "frozen_R3_MF": f1_11(y_te, fm), "peft_R3_MF": f1_11(y_te, pm),
            "d_R3_MF": dm, "lo_R3_MF": lm, "hi_R3_MF": hm, "p_R3_MF": pm_p,
            "frozen_R0_Wconf": f1_11(y_te, fw), "peft_R0_Wconf": f1_11(y_te, pw),
            "d_R0_Wconf": dw, "lo_R0_Wconf": lw, "hi_R0_Wconf": hw, "p_R0_Wconf": pw_p,
        })
    peft_csv = _os.path.join(args.out_dir, "peft_pairs_fullgraph.csv")
    with open(peft_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(peft_rows[0].keys()))
        w.writeheader()
        w.writerows(peft_rows)
    print(f"[mb-full] écrit {peft_csv}", flush=True)


if __name__ == "__main__":
    main()
