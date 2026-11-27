#!/usr/bin/env python3
"""Lissage des probabilités sur le graphe COMPLET (les 8 voisines réelles), avec poids par voisine.

Question : au déploiement on ne sait pas quelles tuiles sont « annotées », donc chaque tuile a ses
8 voisines réelles (annotées + fenêtres non annotées, `out/aux/`). Peut-on pondérer chaque voisine
pour retrouver le gain du graphe « annotées seules » (0,5297 pour R0+MF) ?

Règles de lecture (écrites AVANT de regarder le test) :
  * Seules les informations disponibles au déploiement servent à pondérer : la direction de
    l'arête, la probabilité prédite de la voisine (confiance, entropie), jamais son statut
    annoté. L'étiquette vraie n'entre ni dans les poids ni dans la propagation.
  * Les références : R0 base (0,4719), UNI_FULL (uniforme sur le graphe complet = déploiement
    sans pondération) et ORACLE (poids 1 si voisine annotée, 0 sinon : exige de connaître le
    statut annoté, donc NON déployable ; doit reproduire R0+MF = 0,5297 à β=8, T=5).
  * La famille de bras est déclarée dans ARMS ; tous sont rapportés. β, T, paramètre du bras et
    normalisation sont choisis sur la F1 de validation. Le test ne sert qu'à rapporter.
  * Le bras « domain » (P(annotée|x) apprise sur les orthos d'entraînement) est un DIAGNOSTIC :
    il reconstruit en partie le masque oracle, son gain mesure la proximité au chiffre
    « annotées seules », pas une précision sur toute la carte.

Threads BLAS par défaut, comme les scripts d'origine (voir plus bas). Aucun fichier existant
n'est modifié : sorties dans out/decoding_full/.
"""
from __future__ import annotations

import os

# Mono-thread par défaut (déterministe, canonique : AGENTS §4.8). Les scripts d'origine
# (decoding_scales.py, decoding_ablation.py) ne l'imposaient pas et les probabilités n'ont jamais été
# sauvegardées : la sonde est refaite ici, et R0 base sort à 0,4720 (1 thread) / 0,4718 (threads par
# défaut) contre 0,47194 stocké. Tous les bras de CE script partagent les mêmes numériques ; les
# garde-fous exigent <0,001 d'écart et >99 % de prédictions identiques aux fichiers stockés.
if os.environ.get("WFG_MONO", "1") == "1":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_smoothing import build_graph, f1_macro, onehot  # noqa: E402

ROOT = HERE.parent  # research/gnn_grid_2026-10-31
REPO = HERE.parents[2]  # racine du dépôt
STRIDE = 112
N_CLS = 11
BETAS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0)  # au-delà de 12 : les anciens choix touchaient le bord
ITERS = (1, 2, 3, 5, 8)


def row_normalise(W, A, norm):
    """row : somme des poids des voisines = 1 ; deg : divise par le NOMBRE de voisines présentes
    (une voisine peu fiable pèse alors moins au total, au lieu d'être renormalisée)."""
    if norm == "row":
        d = np.asarray(W.sum(1)).ravel()
    else:
        d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(1.0 / np.maximum(d, 1e-9)) @ W).tocsr()


def mf_all(logp, Wn, beta, q0, iters_wanted):
    """Champ moyen avec matrice de voisinage Wn déjà normalisée. Même mise à jour que
    label_smoothing.meanfield ; renvoie l'argmax à chaque nombre d'itérations demandé."""
    Q = q0
    out = {}
    for t in range(1, max(iters_wanted) + 1):
        M = Wn @ Q
        Z = logp + beta * M
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        Q = (E / E.sum(1, keepdims=True)).astype(np.float32)
        if t in iters_wanted:
            out[t] = Q.argmax(1).astype(np.int64)
    return out


def main() -> None:
    sys.path.insert(0, str(REPO))
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    os.chdir(ROOT)
    outdir = ROOT / "out" / "decoding_full"
    (outdir / "preds").mkdir(parents=True, exist_ok=True)

    tiles = json.load(open("out/canonical_split.json"))["tiles"]
    X = np.load("out/aux/features_tile.npy").astype(np.float32, copy=False)
    Xa = np.load("out/aux/aux_features.npy")  # float16, 40 030 x 768
    aux = json.load(open("out/aux/aux_index.json"))
    n_lab, n_aux = len(tiles), len(aux)
    assert X.shape[0] == n_lab and Xa.shape[0] == n_aux
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    tr, va, te = (np.where(s == k)[0] for k in ("train", "val", "test"))
    pos_lab = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]
    pos_aux = [(d["ortho"], d["row"], d["col"]) for d in aux]
    pos_all = pos_lab + pos_aux
    n = len(pos_all)
    print(f"[full] annotées={n_lab} aux={n_aux} train/val/test={len(tr)}/{len(va)}/{len(te)}", flush=True)

    # ---------- garde-fous de construction du graphe
    assert not (set(pos_lab) & set(pos_aux)), "positions annotées et auxiliaires qui se chevauchent"
    A = build_graph(pos_all)
    A_lab = build_graph(pos_lab)
    assert A[:n_lab, :n_lab].nnz == A_lab.nnz, (A[:n_lab, :n_lab].nnz, A_lab.nnz)
    print(f"[full] arêtes graphe complet={A.nnz}, annotées×annotées={A_lab.nnz} (identiques à build_graph)",
          flush=True)

    # ---------- sonde R0 : mêmes réglages que decoding_scales.py (C=0.001 de scales.json)
    C = json.load(open("out/decoding/scales.json"))["C"]
    sc = StandardScaler().fit(X[tr])
    clf = make_canonical_lr(C=C)
    clf.fit(sc.transform(X[tr]).astype(np.float32), y[tr])

    def proba(Xsrc):
        out = []
        for i in range(0, Xsrc.shape[0], 20000):
            ch = sc.transform(Xsrc[i:i + 20000].astype(np.float32)).astype(np.float32)
            out.append(clf.predict_proba(ch).astype(np.float32))
        return np.concatenate(out)

    P = np.concatenate([proba(X), proba(Xa)])
    logp = np.log(np.maximum(P, 1e-12)).astype(np.float32)
    init = P.argmax(1)
    base_te = f1_macro(y[te], init[te])
    print(f"[full] R0 base : val={f1_macro(y[va], init[va]):.4f} test={base_te:.4f} (C={C})", flush=True)
    same0 = float((init[te].astype(np.uint8) == np.load("out/decoding/preds/R0_base.npy")).mean())
    print(f"[full] GARDE-FOU R0 base : écart au 0,47194 stocké = {base_te - 0.4719370552553896:+.5f}, "
          f"prédictions identiques à R0_base.npy : {same0*100:.3f} %", flush=True)
    assert abs(base_te - 0.4719370552553896) < 1e-3 and same0 > 0.99, "R0 base ne reproduit pas 0,4719"

    # ---------- caractéristiques des arêtes (direction) et des voisines (confiance)
    coo = A.tocoo()
    ri, ci = coo.row, coo.col
    o_arr = np.array([p[0] for p in pos_all])
    r_arr = np.array([p[1] for p in pos_all])
    c_arr = np.array([p[2] for p in pos_all])
    assert (o_arr[ri] == o_arr[ci]).all()
    dr = (r_arr[ci] - r_arr[ri]) // STRIDE
    dc = (c_arr[ci] - c_arr[ri]) // STRIDE
    ortho_edge = (dr == 0) | (dc == 0)
    conf = P.max(1)
    ent = -(P * np.log(np.maximum(P, 1e-12))).sum(1) / np.log(N_CLS)
    is_ann = np.arange(n) < n_lab

    def W_from(edge_w):
        keep = edge_w > 0
        return sp.csr_matrix((edge_w[keep].astype(np.float32), (ri[keep], ci[keep])), shape=(n, n))

    # ---------- diagnostics : qui sont les voisines des tuiles de test ?
    te_mask = np.zeros(n, bool)
    te_mask[te] = True
    e_te = te_mask[ri]
    nb_ann = is_ann[ci][e_te]
    diag = {
        "voisines_par_tuile_test_graphe_complet": float(e_te.sum() / len(te)),
        "voisines_par_tuile_test_annotees_seules": float((e_te & is_ann[ci]).sum() / len(te)),
        "part_voisines_auxiliaires": float(1 - nb_ann.mean()),
        "conf_mediane_voisine_annotee": float(np.median(conf[ci[e_te]][nb_ann])),
        "conf_mediane_voisine_aux": float(np.median(conf[ci[e_te]][~nb_ann])),
    }
    # étiquette prédite de la voisine = étiquette vraie de la tuile centrale ? (annotée vs aux)
    ytile = np.full(n, -1)
    ytile[:n_lab] = y
    agree = (init[ci] == ytile[ri])[e_te]
    diag["accord_pred_voisine_vs_vraie_centre_annotee"] = float(agree[nb_ann].mean())
    diag["accord_pred_voisine_vs_vraie_centre_aux"] = float(agree[~nb_ann].mean())
    # une voisine confiante est-elle plus souvent juste ?
    for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01)):
        m = (conf[ci[e_te]] >= lo) & (conf[ci[e_te]] < hi)
        diag[f"accord_conf_{lo}-{hi}"] = [float(agree[m].mean()), int(m.sum())]
    print("[full] diagnostics :", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in diag.items()}, ensure_ascii=False), flush=True)

    # ---------- domaine : P(annotée | x), apprise sur les orthos d'entraînement SEULEMENT
    ortho_split = {t["ortho"]: t["split"] for t in tiles}
    sp_node = np.array([ortho_split[o] for o in o_arr])
    isdom = lambda k: np.where(sp_node == k)[0]  # noqa: E731
    Xall_row = lambda idx: np.concatenate([  # noqa: E731
        X[idx[idx < n_lab]], Xa[idx[idx >= n_lab] - n_lab].astype(np.float32)])
    best_dom = (-1, None)
    d_tr, d_va = isdom("train"), isdom("val")
    d_tr = np.concatenate([d_tr[d_tr < n_lab], d_tr[d_tr >= n_lab]])
    d_va = np.concatenate([d_va[d_va < n_lab], d_va[d_va >= n_lab]])
    sc_d = StandardScaler().fit(Xall_row(d_tr))
    Xd_tr = sc_d.transform(Xall_row(d_tr)).astype(np.float32)
    yd_tr = is_ann[d_tr].astype(int)
    Xd_va = sc_d.transform(Xall_row(d_va)).astype(np.float32)
    yd_va = is_ann[d_va].astype(int)
    for Cd in (1e-3, 1e-2, 1e-1):
        dom = LogisticRegression(C=Cd, max_iter=300)
        dom.fit(Xd_tr, yd_tr)
        auc = roc_auc_score(yd_va, dom.decision_function(Xd_va))
        if auc > best_dom[0]:
            best_dom = (auc, dom, Cd)
    auc_dom, dom, Cd = best_dom
    pdom = np.empty(n, np.float32)
    for i in range(0, n, 20000):
        idx = np.arange(i, min(i + 20000, n))
        ch = np.concatenate([X[idx[idx < n_lab]], Xa[idx[idx >= n_lab] - n_lab].astype(np.float32)])
        pdom[idx] = dom.predict_proba(sc_d.transform(ch).astype(np.float32))[:, 1]
    del Xd_tr, Xd_va
    diag["domain_auc_val"] = float(auc_dom)
    diag["domain_C"] = Cd
    diag["domain_pdom_moy_annotee"] = float(pdom[:n_lab].mean())
    diag["domain_pdom_moy_aux"] = float(pdom[n_lab:].mean())
    print(f"[full] domaine : AUC val={auc_dom:.3f} (C={Cd}) p moyen annotée={pdom[:n_lab].mean():.3f} "
          f"aux={pdom[n_lab:].mean():.3f}", flush=True)

    # ---------- utilité apprise de chaque voisine (ajouté APRÈS les premiers résultats : bras exploratoire)
    # cible : la prédiction de la voisine j est-elle égale à la VRAIE classe de la tuile centrale i ?
    # (seul le centre a besoin d'une étiquette, donc les voisines auxiliaires sont utilisables).
    # Variables : classe prédite de j, confiance et entropie de j, direction, cosinus(x_i, x_j).
    # PAS le statut annoté de j, PAS la prédiction de i (sinon on écrase la correction).
    # Entraînée sur les arêtes dont le centre est dans un ortho d'entraînement.
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xn16 = np.empty((n, X.shape[1]), np.float16)
    for i in range(0, n, 20000):
        idx = np.arange(i, min(i + 20000, n))
        ch = np.concatenate([X[idx[idx < n_lab]], Xa[idx[idx >= n_lab] - n_lab].astype(np.float32)])
        Xn16[idx] = (ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)).astype(np.float16)
    cosim = np.empty(len(ri), np.float32)
    for i in range(0, len(ri), 50000):
        a_, b_ = ri[i:i + 50000], ci[i:i + 50000]
        cosim[i:i + 50000] = (Xn16[a_].astype(np.float32) * Xn16[b_].astype(np.float32)).sum(1)
    del Xn16
    F_edge = np.column_stack([init[ci], conf[ci], ent[ci], ortho_edge.astype(np.float32), cosim]).astype(np.float32)
    centre_split = sp_node[ri]
    lab_edge = ri < n_lab
    m_tr = lab_edge & (centre_split == "train")
    m_va = lab_edge & (centre_split == "val")
    z = np.zeros(len(ri), np.int8)
    z[lab_edge] = (init[ci[lab_edge]] == y[ri[lab_edge]]).astype(np.int8)
    hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.1,
                                         categorical_features=[0], random_state=0)
    hgb.fit(F_edge[m_tr], z[m_tr])
    s_hat = hgb.predict_proba(F_edge)[:, 1].astype(np.float32)
    auc_use = roc_auc_score(z[m_va], s_hat[m_va])
    auc_conf = roc_auc_score(z[m_va], conf[ci][m_va])
    diag["utilite_auc_val"] = float(auc_use)
    diag["utilite_auc_val_confiance_seule"] = float(auc_conf)
    diag["utilite_s_moy_voisine_annotee"] = float(s_hat[is_ann[ci]].mean())
    diag["utilite_s_moy_voisine_aux"] = float(s_hat[~is_ann[ci]].mean())
    print(f"[full] utilité apprise : AUC val={auc_use:.3f} (confiance seule {auc_conf:.3f}) ; "
          f"ŝ moyen voisine annotée={s_hat[is_ann[ci]].mean():.3f}, aux={s_hat[~is_ann[ci]].mean():.3f}",
          flush=True)

    # ---------- les bras
    ones = np.ones(len(ri), np.float32)
    q0_hard = onehot(init)
    arms = []  # (famille, paramètre, W, normalisations, q0, déployable)

    def add(fam, par, edge_w, norms, q0=q0_hard, deployable=True):
        arms.append((fam, par, W_from(edge_w), norms, q0, deployable))

    add("UNI_FULL", "-", ones, ("row",))
    add("ORACLE", "-", ones * is_ann[ci], ("row",), deployable=False)
    for dl in (0.0, 0.25, 0.5, 0.75):
        add("direction", f"diag={dl}", np.where(ortho_edge, 1.0, dl).astype(np.float32), ("row", "deg"))
    for g in (1, 2, 4, 8):
        add("conf", f"gamma={g}", (conf[ci] ** g).astype(np.float32), ("row", "deg"))
    for g in (1, 2, 4):
        add("entropie", f"gamma={g}", ((1 - ent[ci]) ** g).astype(np.float32), ("row", "deg"))
    add("soft", "Q0=p", ones, ("row",), q0=P.copy())
    for g in (1, 2, 4):
        add("appris", f"gamma={g}", (s_hat ** g).astype(np.float32), ("row", "deg"))
    for g in (1, 2, 4):
        add("domain(DIAG)", f"gamma={g}", (pdom[ci] ** g).astype(np.float32), ("row", "deg"),
            deployable=False)

    rows = []
    best = {}
    for fam, par, W, norms, q0, dep in arms:
        for norm in norms:
            Wn = row_normalise(W, A, norm)
            for beta in BETAS:
                res = mf_all(logp, Wn, beta, q0, ITERS)
                for T, L in res.items():
                    r = {"fam": fam, "par": par, "norm": norm, "beta": beta, "T": T,
                         "val": f1_macro(y[va], L[va]), "test": f1_macro(y[te], L[te]),
                         "deployable": dep}
                    rows.append(r)
                    if fam not in best or r["val"] > best[fam]["val"]:
                        best[fam] = r
        print(f"[full] {fam:13s} {par:10s} -> meilleur val de la famille = "
              f"{best[fam]['val']:.4f} (test {best[fam]['test']:.4f})", flush=True)

    # ---------- garde-fou ORACLE : doit reproduire R0+MF de out/decoding/ablation.json
    ab = json.load(open("out/decoding/ablation.json"))["R0"]["mf"]
    Wo = [a for a in arms if a[0] == "ORACLE"][0][2]
    Lo = mf_all(logp, row_normalise(Wo, A, "row"), float(ab["beta"]), onehot(init), (int(ab["iters"]),))
    Lo = Lo[int(ab["iters"])]
    f_or = f1_macro(y[te], Lo[te])
    same = float((Lo[te].astype(np.uint8) == np.load("out/decoding/preds/R0_MF.npy")).mean())
    print(f"[full] GARDE-FOU oracle β={ab['beta']} T={ab['iters']} : test={f_or:.4f} "
          f"(attendu {ab['test']:.4f}), prédictions identiques à R0_MF.npy : {same*100:.3f} %", flush=True)
    assert abs(f_or - ab["test"]) < 1e-3 and same > 0.99, "l'oracle ne reproduit pas R0+MF"

    # ---------- prédictions des configurations retenues (choisies sur val)
    def preds_of(r):
        fam, par = r["fam"], r["par"]
        a = [x for x in arms if x[0] == fam and x[1] == par][0]
        Wn = row_normalise(a[2], A, r["norm"])
        return mf_all(logp, Wn, r["beta"], a[4], (r["T"],))[r["T"]]

    sel = {}
    for fam, r in best.items():
        L = preds_of(r)
        name = {"UNI_FULL": "UNI_FULL", "ORACLE": "ORACLE"}.get(fam, "W_" + fam.split("(")[0])
        np.save(outdir / "preds" / f"{name}.npy", L[te].astype(np.uint8))
        sel[name] = r
    np.save(outdir / "preds" / "R0_BASE.npy", init[te].astype(np.uint8))
    # meilleur déployable toutes familles confondues (sélection sur val)
    dep_rows = [r for r in rows if r["deployable"]]
    top = max(dep_rows, key=lambda r: r["val"])
    sel["TOP_DEPLOYABLE"] = top
    np.save(outdir / "preds" / "TOP_DEPLOYABLE.npy", preds_of(top)[te].astype(np.uint8))
    # uniforme au réglage de l'étude annotées seules (β=8, T=5), pour comparaison directe
    uni85 = [r for r in rows if r["fam"] == "UNI_FULL" and r["beta"] == 8.0 and r["T"] == 5][0]

    json.dump({"base_test": base_te, "C": C, "diag": diag, "selected": sel,
               "uniform_full_beta8_T5": uni85, "grid": rows},
              open(outdir / "weighted_fullgraph.json", "w"), indent=1, ensure_ascii=False)
    print("\n=== RÉSUMÉ (choix sur validation) ===")
    print(f"R0 base                      test={base_te:.4f}")
    for name, r in sel.items():
        print(f"{name:16s} {r['par']:10s} {r['norm']:3s} β={r['beta']:<5g} T={r['T']}  "
              f"val={r['val']:.4f} test={r['test']:.4f}  {'' if r['deployable'] else '(non déployable/diag)'}")
    print(f"UNI_FULL à β=8,T=5           val={uni85['val']:.4f} test={uni85['test']:.4f}")
    print("écrit", outdir / "weighted_fullgraph.json")


if __name__ == "__main__":
    main()
