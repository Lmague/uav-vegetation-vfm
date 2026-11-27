#!/usr/bin/env python3
"""Sondes canoniques (11 cls, BLAS mono-thread) + mean-field sur la concaténation
[tile_224 ; S_native_{cls,center}] pour S ∈ {384, 448, 512, 1024}, sur DINOv3-B et SimDINOv2-B.

Comparaison directe pommes-avec-pommes contre [tile_224 ; ctx512@224] (0.4953 / 0.5059).
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
    return float(np.mean(per)), float(np.mean([per[i] for i in range(N_CLS) if i not in RARE_IDX]))


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


def _probe_one(task: tuple) -> dict:
    model, size, kind, split_json, emb_root, base224_root = task
    t0 = time.time()
    tiles = json.load(open(split_json))["tiles"]
    y_ref = np.array([t["label"] for t in tiles], dtype=np.int64)
    s_ref = np.array([t["split"] for t in tiles])
    pos = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]

    mats = {}
    for sp_name in ("train", "val", "test"):
        X224 = np.load(_os.path.join(base224_root, f"{model}_{sp_name}.npy")).astype(np.float32)
        y224 = np.load(_os.path.join(base224_root, f"{model}_{sp_name}_labels.npy")).astype(np.int64)
        if int(y224.max()) >= 11:
            keep = y224 != RHOL12
            X224, y224 = X224[keep], y224[keep].copy()
            y224[y224 > RHOL12] -= 1

        d = _os.path.join(emb_root, f"{model}_native_{size}")
        Xn = np.load(_os.path.join(d, f"{sp_name}_{kind}.npy")).astype(np.float32)
        yn = np.load(_os.path.join(d, f"{sp_name}_labels.npy")).astype(np.int64)
        m = s_ref == sp_name
        assert np.array_equal(y224, yn) and np.array_equal(yn, y_ref[m])
        mats[sp_name] = np.concatenate([X224, Xn], axis=1)

    X_all = np.empty((len(s_ref), mats["train"].shape[1]), dtype=np.float32)
    for sp_name in ("train", "val", "test"):
        X_all[s_ref == sp_name] = mats[sp_name]

    tr, va, te = np.where(s_ref == "train")[0], np.where(s_ref == "val")[0], np.where(s_ref == "test")[0]
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_all[tr])
    Xva = sc.transform(X_all[va])
    Xte = sc.transform(X_all[te])

    best_v, best_c, best_clf = -1.0, None, None
    for c in C_GRID:
        clf = make_canonical_lr(C=c, max_iter=2000, random_state=42)
        clf.fit(Xtr, y_ref[tr])
        fv, _ = f1_11_8(y_ref[va], clf.predict(Xva))
        if fv > best_v:
            best_v, best_c, best_clf = fv, c, clf

    f11, f8 = f1_11_8(y_ref[te], best_clf.predict(Xte))

    A = build_adj(pos)
    A_va = A[va][:, va].tocsr()
    A_te = A[te][:, te].tocsr()
    logp_va = best_clf.predict_log_proba(Xva).astype(np.float32)
    logp_te = best_clf.predict_log_proba(Xte).astype(np.float32)
    best_mf = (-1.0, None, None)
    for beta in BETA_GRID:
        for it in ITER_GRID:
            fv_mf, _ = f1_11_8(y_ref[va], meanfield(logp_va, A_va, beta, it))
            if fv_mf > best_mf[0]:
                best_mf = (fv_mf, beta, it)
    f11_mf, f8_mf = f1_11_8(y_ref[te], meanfield(logp_te, A_te, best_mf[1], best_mf[2]))

    return {
        "model": model,
        "size": size,
        "view": f"224+{size}_{kind}",
        "dim": int(X_all.shape[1]),
        "best_C": best_c,
        "val_11": round(best_v, 4),
        "test_11": round(f11, 4),
        "test_8": round(f8, 4),
        "MF_beta": best_mf[1],
        "MF_iters": best_mf[2],
        "MF_test_11": round(f11_mf, 4),
        "MF_test_8": round(f8_mf, 4),
        "runtime_s": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-json", default="research/gnn_grid_2026-10-31/out/canonical_split.json")
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--base224-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    _os.makedirs(_os.path.dirname(args.out_json), exist_ok=True)
    tasks = [
        (m, sz, kind, args.split_json, args.emb_root, args.base224_root)
        for m in ("dinov3_vitb16_lvd", "simdinov2_vitb16")
        for sz in (384, 448, 512, 1024)
        for kind in ("cls", "center")
    ]
    print(f"[fuse224] {len(tasks)} sondes (workers={args.workers})", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_probe_one, t) for t in tasks]):
            r = fut.result()
            results.append(r)
            print(
                f"  {r['model']:18s} {r['view']:16s} | test_11={r['test_11']:.4f} "
                f"(+MF={r['MF_test_11']:.4f}) val={r['val_11']:.4f} C={r['best_C']} ({r['runtime_s']}s)",
                flush=True,
            )
    results.sort(key=lambda r: (r["model"], r["size"], r["view"]))
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[fuse224] sauvegardé : {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
