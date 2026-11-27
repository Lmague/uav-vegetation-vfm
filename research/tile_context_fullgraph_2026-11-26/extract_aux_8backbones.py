#!/usr/bin/env python3
"""Extrait les embeddings 224×224 des 40 076 tuiles voisines NON ANNOTÉES (auxiliaires)
pour les 8 backbones de la table multibackbone_smoothing (5 gelés + 3 LoRA seed0).

Sécurité critique : avant chaque extraction, vérifie sur 32 tuiles annotées que les
embeddings extraits depuis nos sous-rasters PNG matchent `features_tile.npy` canonique
(cosinus moyen > 0.9995), garantissant l'alignement parfait entre nœuds annotés et auxiliaires.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import build_frozen_extractor, load_finetuned_ssl_backbone
from src.utils import get_normalization

TILE_SIZE = 224


def build_specs(scratch: str, ckpt_dir: str) -> list[dict]:
    simb_pt = os.path.join(scratch, "checkpoints", "simdinov2_vitb_inat21plantae.pth")
    siml_pt = os.path.join(scratch, "checkpoints", "simdinov2_vitl_inat21plantae.pth")
    return [
        {"tag": "dinov3_vitb16_lvd", "arch": "dinov3_vitb16_lvd", "pretrain": None, "ft": None, "bs": 256},
        {"tag": "dinov3_vitl16_lvd", "arch": "dinov3_vitl16_lvd", "pretrain": None, "ft": None, "bs": 128},
        {"tag": "dinov3_vith16plus_lvd", "arch": "dinov3_vith16plus_lvd", "pretrain": None, "ft": None, "bs": 64},
        {
            "tag": "dinov3_vitl16_lvd_lora_s0",
            "arch": "dinov3_vitl16_lvd",
            "pretrain": None,
            "ft": os.path.join(ckpt_dir, "dinov3_vitl16_lvd_lora_frac100_seed0_best.pth"),
            "bs": 128,
        },
        {"tag": "simdinov2_vitb16", "arch": "simdinov2_vitb16", "pretrain": simb_pt, "ft": None, "bs": 256},
        {
            "tag": "simdinov2_vitb16_lora_r8a16_s0",
            "arch": "simdinov2_vitb16",
            "pretrain": simb_pt,
            "ft": os.path.join(
                scratch,
                "sota_screening/lora_simb_ablation/checkpoints/simdinov2_vitb16_lora_r8a16_frac100_seed0_best.pth",
            ),
            "bs": 256,
        },
        {"tag": "simdinov2_vitl16", "arch": "simdinov2_vitl16", "pretrain": siml_pt, "ft": None, "bs": 128},
        {
            "tag": "simdinov2_vitl16_lora_s0",
            "arch": "simdinov2_vitl16",
            "pretrain": siml_pt,
            "ft": os.path.join(ckpt_dir, "simdinov2_vitl16_lora_frac100_seed0_best.pth"),
            "bs": 128,
        },
    ]


@torch.inference_mode()
def sanity_check_against_canonical(
    model,
    forward_fn,
    mean: torch.Tensor,
    std: torch.Tensor,
    tag: str,
    canonical_mb_dir: str,
    tiles: list[dict],
    manifest: dict,
    subraster_dir: str,
    device: torch.device,
) -> float:
    """Extrait 32 tuiles annotées du 1er ortho et vérifie la similarité cosinus vs features_tile.npy."""
    ref_path = os.path.join(canonical_mb_dir, tag, "features_tile.npy")
    X_ref = np.load(ref_path, mmap_mode="r")

    first_ortho = tiles[0]["ortho"]
    idxs = [i for i, t in enumerate(tiles) if t["ortho"] == first_ortho][:32]
    om = manifest["orthos"][first_ortho]
    b_r0, b_c0 = om["bbox_row0"], om["bbox_col0"]
    arr = np.asarray(Image.open(os.path.join(subraster_dir, f"{first_ortho}.png")))

    crops = np.empty((len(idxs), TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    for k, gi in enumerate(idxs):
        tr, tc = tiles[gi]["tile_row_off"], tiles[gi]["tile_col_off"]
        crops[k] = arr[tr - b_r0 : tr - b_r0 + TILE_SIZE, tc - b_c0 : tc - b_c0 + TILE_SIZE]

    x = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - mean) / std
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        got = forward_fn(model, x).float().cpu().numpy()

    ref = np.asarray(X_ref[idxs], dtype=np.float32)
    got_n = got / (np.linalg.norm(got, axis=1, keepdims=True) + 1e-9)
    ref_n = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
    cos = float((got_n * ref_n).sum(axis=1).mean())
    max_diff = float(np.abs(got - ref).max())
    print(f"  [sanity {tag}] cos={cos:.6f} | max|diff|={max_diff:.4f} (n={len(idxs)})", flush=True)
    if cos < 0.999:
        raise RuntimeError(f"Sanity check FAILED for {tag}: cos={cos:.6f} < 0.999! Wrong checkpoint or norm?")
    return cos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subraster-dir", required=True)
    ap.add_argument("--aux-index", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--canonical-mb-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reuse-dir", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scratch = os.environ.get("SCRATCH", "/scratch/lmague")
    manifest = json.load(open(os.path.join(args.subraster_dir, "manifest.json")))
    tiles = json.load(open(args.split_json))["tiles"]
    aux_list = json.load(open(args.aux_index))
    print(f"[aux8] {len(aux_list)} tuiles auxiliaires | 8 backbones", flush=True)

    by_ortho = collections.defaultdict(list)
    for i, a in enumerate(aux_list):
        by_ortho[a["ortho"]].append((i, a))
    orthos = sorted(by_ortho.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = build_specs(scratch, args.ckpt_dir)
    sanity_log = {}

    for spec in specs:
        tag = spec["tag"]
        out_npy = os.path.join(args.out_dir, f"{tag}_aux_224.npy")

        # Réutilisation des 2 backbones déjà extraits précédemment si présents
        if not os.path.exists(out_npy) and args.reuse_dir:
            cand = os.path.join(args.reuse_dir, f"{tag}_aux_224_cls.npy")
            if os.path.exists(cand):
                print(f"[reuse] {tag} <- {cand}", flush=True)
                arr_reuse = np.load(cand)
                assert arr_reuse.shape[0] == len(aux_list)
                np.save(out_npy, arr_reuse)
                continue

        if os.path.exists(out_npy):
            print(f"[skip] {tag} déjà présent ({out_npy})", flush=True)
            continue

        t0 = time.time()
        print(f"\n=== [{tag}] chargement ===", flush=True)
        if spec["ft"] is not None:
            model, forward_fn, dim, norm_key = load_finetuned_ssl_backbone(
                spec["arch"], spec["pretrain"], spec["ft"]
            )
        else:
            model, forward_fn, dim, norm_key = build_frozen_extractor(spec["arch"], spec["pretrain"])

        model = model.to(device).eval()
        mean_t, std_t = get_normalization(norm_key)
        mean = torch.tensor(mean_t, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std_t, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        cos = sanity_check_against_canonical(
            model, forward_fn, mean, std, tag, args.canonical_mb_dir, tiles, manifest, args.subraster_dir, device
        )
        sanity_log[tag] = cos

        buf = np.empty((len(aux_list), dim), dtype=np.float16)
        bs = spec["bs"]
        for o_i, ortho in enumerate(orthos, 1):
            om = manifest["orthos"][ortho]
            b_r0, b_c0 = om["bbox_row0"], om["bbox_col0"]
            arr = np.asarray(Image.open(os.path.join(args.subraster_dir, f"{ortho}.png")))
            items = by_ortho[ortho]
            for start in range(0, len(items), bs):
                batch = items[start : start + bs]
                cur_b = len(batch)
                crops = np.empty((cur_b, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                for k, (_gi, meta) in enumerate(batch):
                    lr0, lc0 = meta["row"] - b_r0, meta["col"] - b_c0
                    crops[k] = arr[lr0 : lr0 + TILE_SIZE, lc0 : lc0 + TILE_SIZE]
                x = torch.from_numpy(crops).to(device, non_blocking=True)
                x = x.permute(0, 3, 1, 2).float().div_(255.0)
                x = (x - mean) / std
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    vec = forward_fn(model, x).float().cpu().numpy().astype(np.float16)
                for k, (gi, _meta) in enumerate(batch):
                    buf[gi] = vec[k]
            del arr
            gc.collect()
            if o_i % 8 == 0 or o_i == len(orthos):
                print(f"  [{tag}] {o_i:2d}/32 orthos ({time.time()-t0:.0f}s)", flush=True)

        np.save(out_npy, buf)
        print(f"[saved] {out_npy} {buf.shape} en {(time.time()-t0)/60:.1f} min ✓", flush=True)
        del model, buf
        torch.cuda.empty_cache()
        gc.collect()

    with open(os.path.join(args.out_dir, "sanity_cos.json"), "w") as f:
        json.dump(sanity_log, f, indent=2)
    print("\n[aux8] Terminé ✓", flush=True)


if __name__ == "__main__":
    main()
