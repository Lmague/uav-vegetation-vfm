#!/usr/bin/env python3
"""CLI analyse de l'espace latent : RankMe + anisotropie (+ couche-par-couche).

    python analyze.py --config configs/frozen_eval.yaml
    python analyze.py --config configs/frozen_eval.yaml --layerwise --layerwise-model dinov3_vitb16_lvd

Deux passes (with/without RHOL). Écrit ``results_dir/<pass>/latent_metrics.json`` et,
si ``--layerwise``, ``results_dir/layerwise/<model>_layerwise.json``.
"""
from __future__ import annotations

import argparse
import os

# Single-thread BLAS before importing numpy/scipy via ``src``.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

from src import utils
from src.config import load_config
from src.features import count_layerwise_layers, load_features, load_layerwise
from src.latent import drop_class, layerwise_curves, metrics_on


def _model_keys(cfg, only=None):
    if only:
        return list(only)
    keys = list(cfg.models)
    if cfg.probe.include_finetuned:
        keys += list(cfg.finetuned_models)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to specific model keys")
    ap.add_argument("--layerwise", action="store_true",
                    help="layerwise RankMe/anisotropy curves (requires layerwise features)")
    ap.add_argument("--layerwise-model", default=None, help="model for layerwise curves (default: first)")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="number of layerwise files (default: auto-detected)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.maybe_mount_drive(cfg.env)
    keys = _model_keys(cfg, args.only)
    rk_split, an_split = cfg.latent.rankme_split, cfg.latent.anisotropy_split

    # Modèles canoniques uniquement (12cls) → source_schema homogène "12cls".
    # ``drops`` est une liste décroissante de labels à retirer en cascade ([] = rien),
    # ou None si la passe n'est pas applicable (jamais le cas ici, source 12cls).
    for tag, drops, names in utils.rhol_passes():
        if drops is None:
            continue
        res = {}
        for key in keys:
            feats = load_features(cfg, key)
            for d in drops:  # cascade décroissante
                feats = {s: drop_class(*feats[s], d) for s in feats}
            res[key] = metrics_on(feats[rk_split][0], feats[an_split][0],
                                  n_pairs=cfg.latent.n_pairs,
                                  subsample_n=cfg.latent.rankme_subsample)
            m = res[key]
            print(f"[analyze:{tag}] {key:24s} dim={m['dim']:<5} "
                  f"RankMe={m['rankme']:>8.2f} Aniso={m['anisotropy']:+.4f}")
        out = os.path.join(cfg.paths.results_dir, tag, "latent_metrics.json")
        utils.save_json({"latent_metrics": res}, out)
        print(f"[analyze:{tag}] -> {out}")

    if args.layerwise:
        key = args.layerwise_model or keys[0]
        # n_layers = nombre RÉEL de fichiers layerwise extraits (pas une inférence d'archi).
        n_layers = (args.n_layers if args.n_layers is not None
                    else count_layerwise_layers(cfg, key, rk_split))
        feats = load_layerwise(cfg, key, rk_split, n_layers)
        curves = layerwise_curves(feats, n_pairs=cfg.latent.n_pairs,
                                  subsample_n=cfg.latent.rankme_subsample)
        out = os.path.join(cfg.paths.results_dir, "layerwise", f"{key}_layerwise.json")
        utils.save_json({"model": key, "split": rk_split, "curves": curves}, out)
        print(f"[analyze] couche-par-couche {key} ({len(curves)} couches) -> {out}")


if __name__ == "__main__":
    main()
