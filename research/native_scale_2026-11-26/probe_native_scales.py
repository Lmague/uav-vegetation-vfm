#!/usr/bin/env python3
"""Sondes linéaires canoniques (11 classes, mono-thread BLAS) sur les embeddings
natifs multi-tailles (384, 448, 512, 1024) pour DINOv3-B et SimDINOv2-B.

Pour chaque (modèle, taille), évalue 3 représentations :
  - cls          (768)  : token CLS global sur la fenêtre S×S
  - center       (768)  : moyenne des 14×14 patchs centraux (tuile 224×224 contextualisée)
  - center_cls   (1536) : concaténation [center ; cls]

Protocole strictement identique au benchmark canonique (AGENTS.md §4.3, §4.8) :
  - StandardScaler float32 fit sur train uniquement
  - make_canonical_lr (LogisticRegression lbfgs multinomial, max_iter=2000)
  - C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}, sélection val f1_macro (11 cls)
  - BLAS mono-thread ; parallélisation inter-tâches par ProcessPoolExecutor.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import make_canonical_lr

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
CLASS_NAMES = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF", "RUBC", "SEDG", "TUSS", "WILL"]
RARE_IDX = {1, 3, 7}  # ARCA, DRYI, RUBC


def _load_view(emb_dir: str, split: str, view: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.load(_os.path.join(emb_dir, f"{split}_labels.npy"))
    if view == "cls":
        X = np.load(_os.path.join(emb_dir, f"{split}_cls.npy")).astype(np.float32)
    elif view == "center":
        X = np.load(_os.path.join(emb_dir, f"{split}_center.npy")).astype(np.float32)
    elif view == "center_cls":
        Xc = np.load(_os.path.join(emb_dir, f"{split}_center.npy")).astype(np.float32)
        Xg = np.load(_os.path.join(emb_dir, f"{split}_cls.npy")).astype(np.float32)
        X = np.concatenate([Xc, Xg], axis=1)
    else:
        raise ValueError(view)
    return X, y


def _probe_one(task: tuple[str, int, str, str, str, int]) -> dict:
    model_name, size, view, emb_dir, out_dir, max_iter = task
    out_json = _os.path.join(out_dir, f"{model_name}_native_{size}_{view}.json")
    if _os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)

    t0 = time.time()
    X_tr, y_tr = _load_view(emb_dir, "train", view)
    X_va, y_va = _load_view(emb_dir, "val", view)
    X_te, y_te = _load_view(emb_dir, "test", view)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    val_scores = {}
    best_c, best_val = None, -1.0
    best_clf = None
    for c in C_GRID:
        clf = make_canonical_lr(C=c, max_iter=max_iter, random_state=42)
        clf.fit(X_tr, y_tr)
        pv = clf.predict(X_va)
        f1v = float(f1_score(y_va, pv, average="macro", zero_division=0))
        val_scores[str(c)] = f1v
        if f1v > best_val:
            best_val = f1v
            best_c = c
            best_clf = clf

    pt = best_clf.predict(X_te)
    per_cls = f1_score(y_te, pt, labels=list(range(11)), average=None, zero_division=0)
    f1_11 = float(np.mean(per_cls))
    f1_8 = float(np.mean([per_cls[i] for i in range(11) if i not in RARE_IDX]))
    acc = float(accuracy_score(y_te, pt))
    bacc = float(balanced_accuracy_score(y_te, pt))

    # Sauvegarder les prédictions test pour bootstrap apparié éventuel
    preds_dir = _os.path.join(out_dir, "preds")
    _os.makedirs(preds_dir, exist_ok=True)
    np.save(_os.path.join(preds_dir, f"{model_name}_native_{size}_{view}_test_preds.npy"), pt.astype(np.int16))

    res = {
        "model": model_name,
        "native_size": size,
        "view": view,
        "dim": int(X_tr.shape[1]),
        "best_C": best_c,
        "val_f1_11cls": best_val,
        "val_scores": val_scores,
        "test_f1_11cls": f1_11,
        "test_f1_8cls": f1_8,
        "test_accuracy": acc,
        "test_balanced_accuracy": bacc,
        "per_class_f1": {CLASS_NAMES[i]: float(per_cls[i]) for i in range(11)},
        "runtime_s": round(time.time() - t0, 1),
        "schema": "11cls_no_rhol",
        "split": "spatial_v3",
    }
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="dinov3_vitb16_lvd,simdinov2_vitb16")
    ap.add_argument("--sizes", default="384,448,512,1024")
    ap.add_argument("--views", default="cls,center,center_cls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    _os.makedirs(args.out_dir, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    tasks = []
    for m in models:
        for sz in sizes:
            emb_dir = _os.path.join(args.emb_root, f"{m}_native_{sz}")
            if not _os.path.isdir(emb_dir):
                print(f"[warn] manquant : {emb_dir}", flush=True)
                continue
            for v in views:
                tasks.append((m, sz, v, emb_dir, args.out_dir, args.max_iter))

    print(f"[probe] {len(tasks)} sondes à exécuter (workers={args.workers})", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_probe_one, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            res = fut.result()
            results.append(res)
            print(
                f"  {res['model']:18s} @{res['native_size']:4d} {res['view']:10s} | "
                f"val={res['val_f1_11cls']:.4f} test_11={res['test_f1_11cls']:.4f} "
                f"test_8={res['test_f1_8cls']:.4f} C={res['best_C']} ({res['runtime_s']}s)",
                flush=True,
            )

    results.sort(key=lambda r: (r["model"], r["native_size"], r["view"]))
    summary_path = _os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] Résumé sauvegardé : {summary_path}", flush=True)


if __name__ == "__main__":
    main()
