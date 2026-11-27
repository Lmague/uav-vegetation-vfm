#!/usr/bin/env python3
"""Évaluation 100 % DÉPLOYABLE sur le graphe COMPLET (8 vraies voisines = 80 088 annotées
+ 40 076 auxiliaires non annotées = 120 164 nœuds, zéro masque d'annotation).

Pour DINOv3-B et SimDINOv2-B aux tailles 224 (cls), 448 (center), 512 (center) + fusion [224;512_center] :
  1. Sondes canoniques (11 cls, BLAS mono-thread) :
       - R0            : tuile seule
       - R0 + MF_full  : lissage mean-field sur les 8 vraies voisines (tous les 120 164 nœuds)
       - R3_full       : moyenne 3×3 sur les 8 vraies voisines (annotées + auxiliaires)
       - R3_full + MF  : moyenne 3×3 complète + mean-field sur les 8 vraies voisines
  2. GNN (GATv2 2 et 3 couches + contrôle GNN2rw, 3 seeds) sur le graphe COMPLET de 120 164 nœuds.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analyze_native_followup import (
    BETA_GRID,
    C_GRID,
    ITER_GRID,
    N_CLS,
    STRIDE,
    build_adj,
    f1_11_8,
    load_canonical_matrix,
    mean_features,
    meanfield,
)
from src.utils import make_canonical_lr


def load_full_matrix(
    split_json: str,
    emb_root: str,
    base224_root: str,
    aux_dir: str,
    model: str,
    size: int,
    view: str,
):
    """Retourne (X_full [120164, D], y_lab [80088], s_full [120164], pos_full, n_lab)."""
    if view == "224+512_center":
        X224, y_lab, s_lab, pos_lab = load_canonical_matrix(split_json, emb_root, base224_root, model, 224, "cls")
        X512, _, _, _ = load_canonical_matrix(split_json, emb_root, base224_root, model, 512, "center")
        X_lab = np.concatenate([X224, X512], axis=1)
        A224 = np.load(_os.path.join(aux_dir, f"{model}_aux_224_cls.npy")).astype(np.float32)
        A512 = np.load(_os.path.join(aux_dir, f"{model}_aux_512_center.npy")).astype(np.float32)
        X_aux = np.concatenate([A224, A512], axis=1)
    else:
        X_lab, y_lab, s_lab, pos_lab = load_canonical_matrix(split_json, emb_root, base224_root, model, size, view)
        X_aux = np.load(_os.path.join(aux_dir, f"{model}_aux_{size}_{view}.npy")).astype(np.float32)

    aux_idx = json.load(open(_os.path.join(aux_dir, "aux_index.json")))
    assert len(aux_idx) == len(X_aux)
    pos_aux = [(a["ortho"], a["row"], a["col"]) for a in aux_idx]
    s_aux = np.array([a["split"] for a in aux_idx])

    X_full = np.concatenate([X_lab, X_aux], axis=0)
    s_full = np.concatenate([s_lab, s_aux], axis=0)
    pos_full = pos_lab + pos_aux
    return X_full, y_lab, s_full, pos_full, len(X_lab)


def eval_probe_fullgraph_one(task: tuple) -> dict:
    model, size, view, split_json, emb_root, base224_root, aux_dir = task
    t0 = time.time()
    X_full, y_lab, s_full, pos_full, n_lab = load_full_matrix(
        split_json, emb_root, base224_root, aux_dir, model, size, view
    )
    A = build_adj(pos_full)

    # Indices labellisés (< n_lab) et complets (lab + aux, par split d'ortho disjoint)
    s_lab = s_full[:n_lab]
    tr_lab = np.where(s_lab == "train")[0]
    va_lab = np.where(s_lab == "val")[0]
    te_lab = np.where(s_lab == "test")[0]

    va_all = np.where(s_full == "val")[0]
    te_all = np.where(s_full == "test")[0]
    # Position des tuiles labellisées dans les sous-graphes val_all / te_all
    va_mask_lab = va_all < n_lab
    te_mask_lab = te_all < n_lab
    A_va = A[va_all][:, va_all].tocsr()
    A_te = A[te_all][:, te_all].tocsr()

    # Vérification nombre moyen de voisins sur test labellisé
    deg_te_lab = float(np.asarray(A_te.sum(1)).ravel()[te_mask_lab].mean())

    def _fit_and_mf_full(Xmat: np.ndarray, tag: str) -> dict:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xmat[tr_lab])
        Xva_all = sc.transform(Xmat[va_all])
        Xte_all = sc.transform(Xmat[te_all])
        Xva_lab = Xva_all[va_mask_lab]
        Xte_lab = Xte_all[te_mask_lab]

        best_v, best_c, best_clf = -1.0, None, None
        for c in C_GRID:
            clf = make_canonical_lr(C=c, max_iter=2000, random_state=42)
            clf.fit(Xtr, y_lab[tr_lab])
            fv, _ = f1_11_8(y_lab[va_lab], clf.predict(Xva_lab))
            if fv > best_v:
                best_v, best_c, best_clf = fv, c, clf

        f11, f8 = f1_11_8(y_lab[te_lab], best_clf.predict(Xte_lab))

        # Mean-field sur TOUS les nœuds (lab + aux), évaluation sur labellisés uniquement
        logp_va = best_clf.predict_log_proba(Xva_all).astype(np.float32)
        logp_te = best_clf.predict_log_proba(Xte_all).astype(np.float32)
        best_mf = (-1.0, None, None)
        for beta in BETA_GRID:
            for it in ITER_GRID:
                pv_all = meanfield(logp_va, A_va, beta, it)
                fv_mf, _ = f1_11_8(y_lab[va_lab], pv_all[va_mask_lab])
                if fv_mf > best_mf[0]:
                    best_mf = (fv_mf, beta, it)
        pt_all = meanfield(logp_te, A_te, best_mf[1], best_mf[2])
        f11_mf, f8_mf = f1_11_8(y_lab[te_lab], pt_all[te_mask_lab])
        return {
            f"{tag}_C": best_c,
            f"{tag}_val_11": round(best_v, 4),
            f"{tag}_test_11": round(f11, 4),
            f"{tag}_test_8": round(f8, 4),
            f"{tag}_MF_beta": best_mf[1],
            f"{tag}_MF_iters": best_mf[2],
            f"{tag}_MF_val_11": round(best_mf[0], 4),
            f"{tag}_MF_test_11": round(f11_mf, 4),
            f"{tag}_MF_test_8": round(f8_mf, 4),
        }

    out = {
        "model": model,
        "size": size,
        "view": view,
        "dim": int(X_full.shape[1]),
        "mean_deg_test_full": round(deg_te_lab, 2),
    }
    out.update(_fit_and_mf_full(X_full, "R0"))
    if view != "224+512_center":
        Xm = mean_features(X_full, A)
        out.update(_fit_and_mf_full(Xm, "R3_full"))
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def run_gnn_fullgraph(split_json: str, emb_root: str, base224_root: str, aux_dir: str, threads: int = 8) -> list[dict]:
    import torch

    torch.set_num_threads(threads)
    gnn_dir = _os.path.join(REPO_ROOT, "research", "gnn_grid_2026-10-31", "code")
    if gnn_dir not in sys.path:
        sys.path.insert(0, gnn_dir)
    from run_gnn import build_graph, train_gnn

    configs = [
        ("simdinov2_vitb16", 224, "cls"),
        ("simdinov2_vitb16", 448, "center"),
        ("simdinov2_vitb16", 512, "center"),
        ("dinov3_vitb16_lvd", 224, "cls"),
        ("dinov3_vitb16_lvd", 448, "center"),
    ]
    results = []
    for model, size, view in configs:
        t0 = time.time()
        X_full, y_lab, s_full, pos_full, n_lab = load_full_matrix(
            split_json, emb_root, base224_root, aux_dir, model, size, view
        )
        s_lab = s_full[:n_lab]
        tr = np.where(s_lab == "train")[0]
        va = np.where(s_lab == "val")[0]
        te = np.where(s_lab == "test")[0]

        sc = StandardScaler().fit(X_full[tr])
        Xs = sc.transform(X_full).astype(np.float32)

        pos_lab = pos_full[:n_lab]
        pos_aux = pos_full[n_lab:]
        _, n_nodes, src_np, dst_np, attr_np = build_graph(None, pos_lab, pos_aux)
        src_t, dst_t, attr_t = torch.from_numpy(src_np), torch.from_numpy(dst_np), torch.from_numpy(attr_np)
        # y étendu à n_nodes (les auxiliaires ne sont jamais indexés dans la loss ni les métriques)
        y_pad = np.full(n_nodes, -1, dtype=np.int64)
        y_pad[:n_lab] = y_lab

        entry = {"model": model, "size": size, "view": view, "n_nodes": int(n_nodes), "arms": {}}
        for arm, layers, rewire in [("GNN2", 2, False), ("GNN3", 3, False), ("GNN2rw", 2, True)]:
            s_te, s_va = [], []
            for seed in (0, 1, 2):
                if rewire:
                    _, _, sr, dr, ar = build_graph(None, pos_lab, pos_aux, rewire_seed=100 + seed)
                    st, dt, at = torch.from_numpy(sr), torch.from_numpy(dr), torch.from_numpy(ar)
                else:
                    st, dt, at = src_t, dst_t, attr_t
                te_f1, va_f1, _ep, _pred, _np = train_gnn(
                    Xs, y_pad, tr, va, te, dt, st, at, n_nodes,
                    layers=layers, seed=seed, epochs=60, d=128, lr=1e-3, wd=0.05, p=0.1,
                )
                s_te.append(float(te_f1))
                s_va.append(float(va_f1))
            entry["arms"][arm] = {
                "test_mean": round(float(np.mean(s_te)), 4),
                "test_std": round(float(np.std(s_te, ddof=1)), 4),
                "test_seeds": [round(x, 4) for x in s_te],
                "val_mean": round(float(np.mean(s_va)), 4),
            }
            print(
                f"  [GNN-FULL] {model:18s} @{size:3d} {view:6s} {arm:6s} | "
                f"test={entry['arms'][arm]['test_mean']:.4f}±{entry['arms'][arm]['test_std']:.4f}",
                flush=True,
            )
        entry["runtime_s"] = round(time.time() - t0, 1)
        results.append(entry)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-json", default="research/gnn_grid_2026-10-31/out/canonical_split.json")
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--base224-root", required=True)
    ap.add_argument("--aux-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gnn-threads", type=int, default=8)
    args = ap.parse_args()

    _os.makedirs(args.out_dir, exist_ok=True)
    tasks = [
        ("dinov3_vitb16_lvd", 224, "cls", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
        ("dinov3_vitb16_lvd", 448, "center", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
        ("dinov3_vitb16_lvd", 512, "center", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
        ("dinov3_vitb16_lvd", 512, "224+512_center", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
        ("simdinov2_vitb16", 224, "cls", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
        ("simdinov2_vitb16", 448, "center", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
        ("simdinov2_vitb16", 512, "center", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
        ("simdinov2_vitb16", 512, "224+512_center", args.split_json, args.emb_root, args.base224_root, args.aux_dir),
    ]

    print("=== [1/2] SONDES + MEAN-FIELD SUR GRAPHE COMPLET (8 VRAIES VOISINES) ===", flush=True)
    probe_res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(eval_probe_fullgraph_one, t) for t in tasks]):
            r = fut.result()
            probe_res.append(r)
            r3s = (
                f"R3_full={r['R3_full_test_11']:.4f} R3_full+MF={r['R3_full_MF_test_11']:.4f}"
                if "R3_full_test_11" in r else "(fusion)"
            )
            print(
                f"  [FULL] {r['model']:18s} @{r['size']:3d} {r['view']:14s} (deg={r['mean_deg_test_full']:.2f}) | "
                f"R0={r['R0_test_11']:.4f} R0+MF={r['R0_MF_test_11']:.4f} | {r3s} ({r['runtime_s']}s)",
                flush=True,
            )
    probe_res.sort(key=lambda r: (r["model"], r["size"], r["view"]))
    with open(_os.path.join(args.out_dir, "fullgraph_probes_mf.json"), "w") as f:
        json.dump(probe_res, f, indent=2)

    print("\n=== [2/2] GNN (GATv2, 3 seeds) SUR GRAPHE COMPLET (120 164 NŒUDS) ===", flush=True)
    gnn_res = run_gnn_fullgraph(args.split_json, args.emb_root, args.base224_root, args.aux_dir, threads=args.gnn_threads)
    with open(_os.path.join(args.out_dir, "fullgraph_gnn.json"), "w") as f:
        json.dump(gnn_res, f, indent=2)

    print("\n[done] Évaluation graphe complet terminée.", flush=True)


if __name__ == "__main__":
    main()
