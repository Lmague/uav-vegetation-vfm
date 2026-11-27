#!/usr/bin/env python3
"""Suivi complet sur les embeddings natifs (384, 448, 512, 1024) + baselines 224 :
  1. Géométrie de l'espace latent (RankMe, anisotropie, cos voisins, CH, DB, silhouette, NC1, Fisher)
  2. Contexte 3x3 + lissage mean-field (R0, R0+MF, R3, R3+MF) + fusion [n_center ; 2n_cls]
  3. GNN (GATv2 2/3 couches + contrôle arêtes permutées, 3 seeds) sur 224 vs meilleur natif

Toutes les sondes sklearn tournent en BLAS mono-thread (AGENTS.md §4.8) ; parallélisation
inter-tâches par ProcessPoolExecutor. Le GNN PyTorch tourne ensuite dans la même job CPU.
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
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import make_canonical_lr

STRIDE = 112
N_CLS = 11
RHOL12 = 7
RARE_IDX = {1, 3, 7}
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
BETA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0]
ITER_GRID = [1, 2, 3, 5, 8, 12]
OFFS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def f1_11_8(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    per = f1_score(y, p, labels=list(range(N_CLS)), average=None, zero_division=0)
    f11 = float(np.mean(per))
    f8 = float(np.mean([per[i] for i in range(N_CLS) if i not in RARE_IDX]))
    return f11, f8


def load_canonical_matrix(
    split_json: str,
    emb_root: str,
    base224_root: str,
    model: str,
    size: int,
    view: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple]]:
    """Charge (X [80088, D], y [80088], split_arr [80088], pos) dans l'ordre exact de canonical_split.json."""
    tiles = json.load(open(split_json))["tiles"]
    y_ref = np.array([t["label"] for t in tiles], dtype=np.int64)
    s_ref = np.array([t["split"] for t in tiles])
    pos = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]

    mats = {}
    for sp_name in ("train", "val", "test"):
        if size == 224:
            X = np.load(_os.path.join(base224_root, f"{model}_{sp_name}.npy")).astype(np.float32)
            y = np.load(_os.path.join(base224_root, f"{model}_{sp_name}_labels.npy")).astype(np.int64)
            if int(y.max()) >= 11:
                keep = y != RHOL12
                X, y = X[keep], y[keep].copy()
                y[y > RHOL12] -= 1
        else:
            d = _os.path.join(emb_root, f"{model}_native_{size}")
            y = np.load(_os.path.join(d, f"{sp_name}_labels.npy")).astype(np.int64)
            if view == "cls":
                X = np.load(_os.path.join(d, f"{sp_name}_cls.npy")).astype(np.float32)
            elif view == "center":
                X = np.load(_os.path.join(d, f"{sp_name}_center.npy")).astype(np.float32)
            elif view == "center_cls":
                X = np.concatenate(
                    [
                        np.load(_os.path.join(d, f"{sp_name}_center.npy")).astype(np.float32),
                        np.load(_os.path.join(d, f"{sp_name}_cls.npy")).astype(np.float32),
                    ],
                    axis=1,
                )
            elif view.startswith("fuse2n_"):
                # fuse2n_cls or fuse2n_center : concat [size_center ; 1024_{cls|center}]
                sub = view.split("_", 1)[1]
                d2 = _os.path.join(emb_root, f"{model}_native_1024")
                X = np.concatenate(
                    [
                        np.load(_os.path.join(d, f"{sp_name}_center.npy")).astype(np.float32),
                        np.load(_os.path.join(d2, f"{sp_name}_{sub}.npy")).astype(np.float32),
                    ],
                    axis=1,
                )
            else:
                raise ValueError(view)
        m = s_ref == sp_name
        assert len(y) == int(m.sum()) and np.array_equal(y, y_ref[m]), f"Mismatch {model}@{size}/{view}/{sp_name}"
        mats[sp_name] = X

    X_all = np.empty((len(s_ref), mats["train"].shape[1]), dtype=np.float32)
    for sp_name in ("train", "val", "test"):
        X_all[s_ref == sp_name] = mats[sp_name]
    return X_all, y_ref, s_ref, pos


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


def mean_features(X: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    A1 = A.copy()
    A1.setdiag(1.0)
    cnt = np.asarray(A1.sum(1)).ravel()
    return (np.asarray(A1 @ X) / cnt[:, None]).astype(np.float32)


def meanfield(logp: np.ndarray, A: sp.csr_matrix, beta: float, iters: int) -> np.ndarray:
    init = np.argmax(logp, axis=1)
    Q = np.zeros((len(init), N_CLS), np.float32)
    Q[np.arange(len(init)), init] = 1.0
    for _ in range(iters):
        Q = A @ Q
        Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-9)
        Z = logp + beta * Q
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        Q = (E / E.sum(1, keepdims=True)).astype(np.float32)
    return Q.argmax(1).astype(np.int64)


# --------------------------------------------------------------------------- 1. GÉOMÉTRIE
def compute_geometry_one(task: tuple) -> dict:
    model, size, view, split_json, emb_root, base224_root = task
    t0 = time.time()
    X_all, y_all, s_all, pos = load_canonical_matrix(split_json, emb_root, base224_root, model, size, view)
    te = s_all == "test"
    X = X_all[te]
    y = y_all[te]
    D = X.shape[1]

    # Spectral : RankMe & stable rank sur X centré (test set 17 598 x 768)
    Xc = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    p = sv / (sv.sum() + 1e-12)
    p = p[p > 0]
    rankme = float(np.exp(-np.sum(p * np.log(p))))
    stable_rank = float((sv ** 2).sum() / (sv[0] ** 2 + 1e-12))

    # Anisotropie : cosinus moyen sur 10 000 paires aléatoires
    rng = np.random.RandomState(42)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    i1 = rng.randint(0, len(X), size=10000)
    i2 = rng.randint(0, len(X), size=10000)
    anisotropy = float((Xn[i1] * Xn[i2]).sum(axis=1).mean())

    # Redondance spatiale : cosinus moyen entre voisins ±112 px sur test
    pos_te = [pos[i] for i in np.where(te)[0]]
    A_te = build_adj(pos_te)
    rr, cc = A_te.nonzero()
    cos_nbr = float((Xn[rr] * Xn[cc]).sum(axis=1).mean())

    # Séparation de classes (sur test)
    sub_idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    sil = float(silhouette_score(Xn[sub_idx], y[sub_idx], metric="cosine"))
    ch = float(calinski_harabasz_score(X, y))
    db = float(davies_bouldin_score(X, y))

    # NC1 & Fisher ratio (sur les 11 classes présentes)
    mu_g = X.mean(axis=0)
    sw_trace, sb_trace = 0.0, 0.0
    for c in range(N_CLS):
        m = y == c
        if m.sum() == 0:
            continue
        mu_c = X[m].mean(axis=0)
        sw_trace += float(((X[m] - mu_c) ** 2).sum())
        sb_trace += float(m.sum() * ((mu_c - mu_g) ** 2).sum())
    fisher = float(sb_trace / (sw_trace + 1e-12))
    nc1_proxy = float(sw_trace / (sb_trace + 1e-12))

    return {
        "model": model,
        "size": size,
        "view": view,
        "dim": D,
        "rankme": round(rankme, 2),
        "rankme_ratio": round(rankme / D, 4),
        "stable_rank": round(stable_rank, 2),
        "anisotropy": round(anisotropy, 4),
        "cos_neighbors": round(cos_nbr, 4),
        "silhouette_cos": round(sil, 4),
        "calinski_harabasz": round(ch, 1),
        "davies_bouldin": round(db, 4),
        "fisher_ratio": round(fisher, 4),
        "nc1_trace_ratio": round(nc1_proxy, 4),
        "runtime_s": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- 2. CONTEXTE + MF
def run_context_mf_one(task: tuple) -> dict:
    model, size, view, split_json, emb_root, base224_root = task
    t0 = time.time()
    X_all, y, s, pos = load_canonical_matrix(split_json, emb_root, base224_root, model, size, view)
    tr, va, te = np.where(s == "train")[0], np.where(s == "val")[0], np.where(s == "test")[0]
    A = build_adj(pos)
    A_va = A[va][:, va].tocsr()
    A_te = A[te][:, te].tocsr()

    def _fit_and_mf(Xmat: np.ndarray, tag: str) -> dict:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xmat[tr])
        Xva = sc.transform(Xmat[va])
        Xte = sc.transform(Xmat[te])
        best_v, best_c, best_clf = -1.0, None, None
        for c in C_GRID:
            clf = make_canonical_lr(C=c, max_iter=2000, random_state=42)
            clf.fit(Xtr, y[tr])
            fv, _ = f1_11_8(y[va], clf.predict(Xva))
            if fv > best_v:
                best_v, best_c, best_clf = fv, c, clf
        p_te = best_clf.predict(Xte)
        f11, f8 = f1_11_8(y[te], p_te)
        logp_va = best_clf.predict_log_proba(Xva).astype(np.float32)
        logp_te = best_clf.predict_log_proba(Xte).astype(np.float32)

        best_mf = (-1.0, None, None)
        for beta in BETA_GRID:
            for it in ITER_GRID:
                pv_mf = meanfield(logp_va, A_va, beta, it)
                fv_mf, _ = f1_11_8(y[va], pv_mf)
                if fv_mf > best_mf[0]:
                    best_mf = (fv_mf, beta, it)
        pt_mf = meanfield(logp_te, A_te, best_mf[1], best_mf[2])
        f11_mf, f8_mf = f1_11_8(y[te], pt_mf)
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

    out = {"model": model, "size": size, "view": view, "dim": int(X_all.shape[1])}
    out.update(_fit_and_mf(X_all, "R0"))
    if not view.startswith("fuse2n_"):
        Xm = mean_features(X_all, A)
        out.update(_fit_and_mf(Xm, "R3"))
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- 3. GNN (GATv2)
def run_gnn_suite(split_json: str, emb_root: str, base224_root: str, threads: int = 12) -> list[dict]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(threads)
    gnn_dir = _os.path.join(REPO_ROOT, "research", "gnn_grid_2026-10-31", "code")
    if gnn_dir not in sys.path:
        sys.path.insert(0, gnn_dir)
    from run_gnn import build_graph, train_gnn

    configs = [
        ("simdinov2_vitb16", 224, "cls"),
        ("simdinov2_vitb16", 512, "center"),
        ("dinov3_vitb16_lvd", 224, "cls"),
        ("dinov3_vitb16_lvd", 448, "center"),
    ]
    results = []
    for model, size, view in configs:
        t0 = time.time()
        X_all, y, s, pos = load_canonical_matrix(split_json, emb_root, base224_root, model, size, view)
        tr, va, te = np.where(s == "train")[0], np.where(s == "val")[0], np.where(s == "test")[0]
        sc = StandardScaler().fit(X_all[tr])
        Xs = sc.transform(X_all).astype(np.float32)

        _p, n_nodes, src_np, dst_np, attr_np = build_graph(None, pos, [])
        src_t = torch.from_numpy(src_np)
        dst_t = torch.from_numpy(dst_np)
        attr_t = torch.from_numpy(attr_np)

        entry = {"model": model, "size": size, "view": view, "arms": {}}
        for arm, layers, rewire in [("GNN2", 2, False), ("GNN3", 3, False), ("GNN2rw", 2, True)]:
            scores_te, scores_va = [], []
            for seed in (0, 1, 2):
                if rewire:
                    _, _, s_rw, d_rw, a_rw = build_graph(None, pos, [], rewire_seed=100 + seed)
                    st, dt, at = torch.from_numpy(s_rw), torch.from_numpy(d_rw), torch.from_numpy(a_rw)
                else:
                    st, dt, at = src_t, dst_t, attr_t
                te_f1, va_f1, ep, _pred, n_params = train_gnn(
                    Xs, y, tr, va, te, dt, st, at, n_nodes,
                    layers=layers, seed=seed, epochs=60, d=128, lr=1e-3, wd=0.05, p=0.1,
                )
                scores_te.append(float(te_f1))
                scores_va.append(float(va_f1))
            entry["arms"][arm] = {
                "test_mean": round(float(np.mean(scores_te)), 4),
                "test_std": round(float(np.std(scores_te, ddof=1)), 4),
                "test_seeds": [round(x, 4) for x in scores_te],
                "val_mean": round(float(np.mean(scores_va)), 4),
            }
            print(
                f"  [GNN] {model:18s} @{size:3d} {view:6s} {arm:6s} | "
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gnn-threads", type=int, default=12)
    args = ap.parse_args()

    _os.makedirs(args.out_dir, exist_ok=True)
    models = ["dinov3_vitb16_lvd", "simdinov2_vitb16"]

    # --- 1. Géométrie latente (224 + 384/448/512/1024 x cls/center = 18 entrées)
    print("\n=== [1/3] GÉOMÉTRIE ESPACE LATENT ===", flush=True)
    geom_tasks = []
    for m in models:
        geom_tasks.append((m, 224, "cls", args.split_json, args.emb_root, args.base224_root))
        for sz in (384, 448, 512, 1024):
            for v in ("cls", "center"):
                geom_tasks.append((m, sz, v, args.split_json, args.emb_root, args.base224_root))
    geom_res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(compute_geometry_one, geom_tasks):
            geom_res.append(res)
            print(
                f"  [geom] {res['model']:18s} @{res['size']:4d} {res['view']:6s} | "
                f"RankMe={res['rankme']:6.1f} aniso={res['anisotropy']:.3f} "
                f"cosNbr={res['cos_neighbors']:.3f} sil={res['silhouette_cos']:.3f} "
                f"CH={res['calinski_harabasz']:6.1f} Fisher={res['fisher_ratio']:.3f}",
                flush=True,
            )
    with open(_os.path.join(args.out_dir, "geometry.json"), "w") as f:
        json.dump(geom_res, f, indent=2)

    # --- 2. Contexte 3x3 + Mean-Field + Fusion 2n
    print("\n=== [2/3] CONTEXTE 3x3 + MEAN-FIELD + FUSION 2n ===", flush=True)
    ctx_tasks = [
        ("dinov3_vitb16_lvd", 224, "cls", args.split_json, args.emb_root, args.base224_root),
        ("dinov3_vitb16_lvd", 448, "center", args.split_json, args.emb_root, args.base224_root),
        ("dinov3_vitb16_lvd", 448, "center_cls", args.split_json, args.emb_root, args.base224_root),
        ("dinov3_vitb16_lvd", 512, "center", args.split_json, args.emb_root, args.base224_root),
        ("dinov3_vitb16_lvd", 448, "fuse2n_cls", args.split_json, args.emb_root, args.base224_root),
        ("simdinov2_vitb16", 224, "cls", args.split_json, args.emb_root, args.base224_root),
        ("simdinov2_vitb16", 448, "center", args.split_json, args.emb_root, args.base224_root),
        ("simdinov2_vitb16", 512, "center", args.split_json, args.emb_root, args.base224_root),
        ("simdinov2_vitb16", 512, "fuse2n_cls", args.split_json, args.emb_root, args.base224_root),
        ("simdinov2_vitb16", 512, "fuse2n_center", args.split_json, args.emb_root, args.base224_root),
    ]
    ctx_res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_context_mf_one, t): t for t in ctx_tasks}
        for fut in as_completed(futs):
            res = fut.result()
            ctx_res.append(res)
            r3_str = (
                f"R3={res['R3_test_11']:.4f} R3+MF={res['R3_MF_test_11']:.4f}"
                if "R3_test_11" in res else "(fusion 2n)"
            )
            print(
                f"  [ctx] {res['model']:18s} @{res['size']:3d} {res['view']:13s} | "
                f"R0={res['R0_test_11']:.4f} R0+MF={res['R0_MF_test_11']:.4f} | {r3_str} ({res['runtime_s']}s)",
                flush=True,
            )
    ctx_res.sort(key=lambda r: (r["model"], r["size"], r["view"]))
    with open(_os.path.join(args.out_dir, "context_mf_2n.json"), "w") as f:
        json.dump(ctx_res, f, indent=2)

    # --- 3. GNN (GATv2 2/3 couches + rewired)
    print("\n=== [3/3] GNN (GATv2, 3 seeds) ===", flush=True)
    gnn_res = run_gnn_suite(args.split_json, args.emb_root, args.base224_root, threads=args.gnn_threads)
    with open(_os.path.join(args.out_dir, "gnn_native.json"), "w") as f:
        json.dump(gnn_res, f, indent=2)

    print("\n[done] Toutes les analyses terminées.", flush=True)


if __name__ == "__main__":
    main()
