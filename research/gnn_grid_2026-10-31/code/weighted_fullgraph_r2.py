#!/usr/bin/env python3
"""Contexte 512 (R2) + lissage sur les 8 voisines réelles : ESTIMATION INTERMÉDIAIRE (proxy).

Les features de contexte 512 des 40 030 fenêtres non annotées n'existent pas (vérifié sur Narval le
2026-11-19). Proxy utilisé ici :
  * tuiles annotées : probabilités R2 (sonde sur [tuile 224 ; contexte 512], C=1e-4 du canonique) ;
  * fenêtres non annotées : probabilités R0 (tuile seule), faute de contexte 512.
Les votes des voisines non annotées sont donc plus faibles que dans un vrai déploiement : l'estimation
est PRUDENTE pour ces voisines (et exacte pour les voisines annotées). Un résultat nul ici ne prouve
pas que le contexte + lissage n'apporte rien ; un gain, lui, est un plancher plausible.

Réutilise les briques de weighted_fullgraph.py (même graphe, même champ moyen, même grille).
Mono-thread. Sorties dans out/decoding_full/ (preds_r2/, weighted_fullgraph_r2.json).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_smoothing import build_graph, f1_macro, onehot  # noqa: E402
from weighted_fullgraph import BETAS, ITERS, mf_all, row_normalise  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = HERE.parent
REPO = HERE.parents[2]
CTX = os.environ["CTX512_FEATURES"]  # features_ctx512.npy des tuiles annotées (copie de Narval)
N_CLS = 11


def main() -> None:
    sys.path.insert(0, str(REPO))
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    os.chdir(ROOT)
    outdir = ROOT / "out" / "decoding_full"
    (outdir / "preds_r2").mkdir(parents=True, exist_ok=True)

    tiles = json.load(open("out/canonical_split.json"))["tiles"]
    X = np.load("out/aux/features_tile.npy", mmap_mode="r")
    Xc = np.load(CTX, mmap_mode="r")
    Xa = np.load("out/aux/aux_features.npy", mmap_mode="r")
    aux = json.load(open("out/aux/aux_index.json"))
    n_lab, n_aux = len(tiles), len(aux)
    n = n_lab + n_aux
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    tr, va, te = (np.where(s == k)[0] for k in ("train", "val", "test"))
    pos_lab = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]
    pos_all = pos_lab + [(d["ortho"], d["row"], d["col"]) for d in aux]
    A = build_graph(pos_all)
    is_ann = np.arange(n) < n_lab
    assert X.shape[0] == Xc.shape[0] == n_lab

    # ---- sonde R2 sur [tuile ; contexte 512], C=1e-4 (sélection canonique sur validation)
    C2 = json.load(open("out/canon/summary_canonical.json"))["records"]["R2"]["selection"]["best_C"]
    Ztr = np.concatenate([X[tr], Xc[tr]], axis=1).astype(np.float32)
    sc2 = StandardScaler().fit(Ztr)
    Ztr = sc2.transform(Ztr).astype(np.float32)
    clf2 = make_canonical_lr(C=C2)
    clf2.fit(Ztr, y[tr])
    del Ztr
    P = np.empty((n, N_CLS), np.float32)
    for i in range(0, n_lab, 20000):
        j = min(i + 20000, n_lab)
        ch = sc2.transform(np.concatenate([X[i:j], Xc[i:j]], axis=1).astype(np.float32))
        P[i:j] = clf2.predict_proba(ch.astype(np.float32))
    # ---- sonde R0 (repli pour les fenêtres non annotées)
    C0 = json.load(open("out/decoding/scales.json"))["C"]
    sc0 = StandardScaler().fit(np.asarray(X[tr], np.float32))
    clf0 = make_canonical_lr(C=C0)
    clf0.fit(sc0.transform(np.asarray(X[tr], np.float32)).astype(np.float32), y[tr])
    for i in range(0, n_aux, 20000):
        j = min(i + 20000, n_aux)
        P[n_lab + i:n_lab + j] = clf0.predict_proba(
            sc0.transform(np.asarray(Xa[i:j], np.float32)).astype(np.float32))
    logp = np.log(np.maximum(P, 1e-12)).astype(np.float32)
    init = P.argmax(1)
    base_te = f1_macro(y[te], init[te])
    print(f"[r2] R2 base : C={C2} val={f1_macro(y[va], init[va]):.4f} test={base_te:.4f} "
          f"(canonique 0,5058)", flush=True)
    ref = np.load("out/canon/preds/R2.npy")
    same = float((init[te].astype(np.uint8) == ref).mean())
    print(f"[r2] GARDE-FOU : prédictions identiques à out/canon/preds/R2.npy : {same*100:.3f} %", flush=True)
    assert abs(base_te - 0.5057952450782065) < 1e-3 and same > 0.99

    ent = -(P * np.log(np.maximum(P, 1e-12))).sum(1) / np.log(N_CLS)
    coo = A.tocoo()
    ri, ci = coo.row, coo.col
    ones = np.ones(len(ri), np.float32)

    def W_from(w):
        k = w > 0
        return sp.csr_matrix((w[k].astype(np.float32), (ri[k], ci[k])), shape=(n, n))

    arms = [("R2h_UNI_FULL", "-", W_from(ones), ("row",)),
            ("R2h_ORACLE", "-", W_from(ones * is_ann[ci]), ("row",))]
    for g in (2, 4):
        arms.append(("R2h_entropie", f"gamma={g}", W_from(((1 - ent[ci]) ** g).astype(np.float32)),
                     ("row", "deg")))

    rows, best = [], {}
    q0 = onehot(init)
    for fam, par, W, norms in arms:
        for norm in norms:
            Wn = row_normalise(W, A, norm)
            for beta in BETAS:
                for T, L in mf_all(logp, Wn, beta, q0, ITERS).items():
                    r = {"fam": fam, "par": par, "norm": norm, "beta": beta, "T": T,
                         "val": f1_macro(y[va], L[va]), "test": f1_macro(y[te], L[te])}
                    rows.append(r)
                    if fam not in best or r["val"] > best[fam]["val"]:
                        best[fam] = r
        print(f"[r2] {fam:14s} {par:8s} meilleur val={best[fam]['val']:.4f} test={best[fam]['test']:.4f} "
              f"(beta={best[fam]['beta']}, T={best[fam]['T']}, {best[fam]['norm']})", flush=True)

    # prédictions retenues (choix sur validation) + base
    np.save(outdir / "preds_r2" / "R2_BASE.npy", init[te].astype(np.uint8))
    for fam in ("R2h_UNI_FULL", "R2h_ORACLE", "R2h_entropie"):
        r = best[fam]
        a = [x for x in arms if x[0] == fam and (fam != "R2h_entropie" or x[1] == r["par"])][0]
        L = mf_all(logp, row_normalise(a[2], A, r["norm"]), r["beta"], q0, (r["T"],))[r["T"]]
        np.save(outdir / "preds_r2" / f"{fam}.npy", L[te].astype(np.uint8))
    json.dump({"base_test": base_te, "val_base": f1_macro(y[va], init[va]), "selected": best, "grid": rows},
              open(outdir / "weighted_fullgraph_r2.json", "w"), indent=1)
    print("\n=== RÉSUMÉ (choix sur validation) ===")
    print(f"R2 base                  val={f1_macro(y[va], init[va]):.4f} test={base_te:.4f}")
    for fam, r in best.items():
        print(f"{fam:14s} {r['par']:8s} {r['norm']:3s} β={r['beta']:<4g} T={r['T']}  "
              f"val={r['val']:.4f} test={r['test']:.4f}")


if __name__ == "__main__":
    main()
