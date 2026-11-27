#!/usr/bin/env python3
"""Extrait les embeddings (224 cls, 448 center, 512 center) pour les 40 076 tuiles
voisines NON ANNOTÉES (auxiliaires ±112 px) depuis les 32 sous-rasters PNG déjà sur Narval.

Zéro extraction locale ni transfert requis : 0/40 076 tuiles hors-limites pour 512×512.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from extract_native_scales import (
    CENTER_PATCHES,
    PATCH_SIZE,
    TILE_SIZE,
    clamp_window,
    forward_cls_and_patches,
    verify_sdpa_equivalence,
)
from src.models import build_frozen_extractor
from src.utils import get_normalization

BATCH_MAP = {224: 256, 448: 96, 512: 64}


def build_aux_list(split_json: str, manifest: dict) -> list[dict]:
    """Construit la liste déterministe des tuiles auxiliaires 1-hop (±112 px) non annotées."""
    tiles = json.load(open(split_json))["tiles"]
    ortho_split = {t["ortho"]: t["split"] for t in tiles}
    lab = {(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles}

    aux_set = set()
    for o, r, c in lab:
        om = manifest["orthos"][o]
        oh, ow = om["orig_h"], om["orig_w"]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + 112 * dr, c + 112 * dc
                if 0 <= nr <= oh - TILE_SIZE and 0 <= nc <= ow - TILE_SIZE:
                    if (o, nr, nc) not in lab:
                        aux_set.add((o, nr, nc))

    aux_list = [
        {"ortho": o, "row": int(r), "col": int(c), "split": ortho_split[o]}
        for (o, r, c) in sorted(aux_set)
    ]
    print(f"[aux] {len(aux_list)} tuiles auxiliaires uniques sur {len(ortho_split)} orthos", flush=True)
    return aux_list


def extract_model_aux(
    model_name: str,
    checkpoint: str | None,
    sizes: list[int],
    subraster_dir: str,
    manifest: dict,
    aux_list: list[dict],
    out_dir: str,
    device: torch.device,
) -> None:
    todo = []
    for sz in sizes:
        kind = "cls" if sz == 224 else "center"
        p = os.path.join(out_dir, f"{model_name}_aux_{sz}_{kind}.npy")
        if os.path.exists(p):
            print(f"[skip] {p} déjà présent", flush=True)
        else:
            todo.append(sz)
    if not todo:
        return

    print(f"\n=== Extraction AUX {model_name} (sizes={todo}) ===", flush=True)
    model, _fwd, dim, norm_key = build_frozen_extractor(model_name, checkpoint)
    model = model.to(device).eval()
    if model_name.startswith("simdinov2"):
        verify_sdpa_equivalence(model, device)

    mean_t, std_t = get_normalization(norm_key)
    mean = torch.tensor(mean_t, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_t, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    by_ortho = collections.defaultdict(list)
    for idx, item in enumerate(aux_list):
        by_ortho[item["ortho"]].append((idx, item))

    buffers = {sz: np.empty((len(aux_list), dim), dtype=np.float16) for sz in todo}
    t0_all = time.time()

    for o_i, ortho in enumerate(sorted(by_ortho.keys()), 1):
        t0 = time.time()
        om = manifest["orthos"][ortho]
        orig_h, orig_w = om["orig_h"], om["orig_w"]
        b_r0, b_c0 = om["bbox_row0"], om["bbox_col0"]
        arr = np.asarray(Image.open(os.path.join(subraster_dir, f"{ortho}.png")))
        items = by_ortho[ortho]

        for sz in todo:
            bs = BATCH_MAP.get(sz, 64)
            grid_hw = sz // PATCH_SIZE
            for start in range(0, len(items), bs):
                batch = items[start : start + bs]
                cur_b = len(batch)
                crops = np.empty((cur_b, sz, sz, 3), dtype=np.uint8)
                pr0s = np.empty(cur_b, dtype=np.int32)
                pc0s = np.empty(cur_b, dtype=np.int32)

                for i, (_idx, meta) in enumerate(batch):
                    tr, tc = meta["row"], meta["col"]
                    r0, _ = clamp_window(tr + TILE_SIZE // 2, sz, orig_h)
                    c0, _ = clamp_window(tc + TILE_SIZE // 2, sz, orig_w)
                    lr0, lc0 = r0 - b_r0, c0 - b_c0
                    crops[i] = arr[lr0 : lr0 + sz, lc0 : lc0 + sz]
                    pr0 = int(round((tr - r0) / PATCH_SIZE))
                    pc0 = int(round((tc - c0) / PATCH_SIZE))
                    pr0s[i] = max(0, min(grid_hw - CENTER_PATCHES, pr0))
                    pc0s[i] = max(0, min(grid_hw - CENTER_PATCHES, pc0))

                x = torch.from_numpy(crops).to(device, non_blocking=True)
                x = x.permute(0, 3, 1, 2).float().div_(255.0)
                x = (x - mean) / std

                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    cls_vec, patches_2d = forward_cls_and_patches(model, model_name, x, grid_hw)
                    if sz == 224:
                        vec = cls_vec
                    elif np.all(pr0s == pr0s[0]) and np.all(pc0s == pc0s[0]):
                        rs, cs = int(pr0s[0]), int(pc0s[0])
                        vec = patches_2d[:, rs : rs + CENTER_PATCHES, cs : cs + CENTER_PATCHES, :].mean(dim=(1, 2))
                    else:
                        vec = torch.stack(
                            [
                                patches_2d[i, pr0s[i] : pr0s[i] + CENTER_PATCHES, pc0s[i] : pc0s[i] + CENTER_PATCHES, :].mean(dim=(0, 1))
                                for i in range(cur_b)
                            ],
                            dim=0,
                        )
                v_np = vec.float().cpu().numpy().astype(np.float16)
                for i, (g_idx, _meta) in enumerate(batch):
                    buffers[sz][g_idx] = v_np[i]

        del arr
        gc.collect()
        print(f"  [aux {model_name}] {o_i:2d}/32 {ortho} ({len(items)} aux) : {time.time()-t0:.1f}s", flush=True)

    for sz in todo:
        kind = "cls" if sz == 224 else "center"
        out_p = os.path.join(out_dir, f"{model_name}_aux_{sz}_{kind}.npy")
        np.save(out_p, buffers[sz])
        print(f"[saved] {out_p} {buffers[sz].shape} ✓", flush=True)

    del model, buffers
    torch.cuda.empty_cache()
    gc.collect()
    print(f"=== {model_name} AUX terminé en {(time.time()-t0_all)/60:.1f} min ===", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subraster-dir", required=True)
    ap.add_argument("--split-json", default="research/gnn_grid_2026-10-31/out/canonical_split.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="dinov3_vitb16_lvd,simdinov2_vitb16")
    ap.add_argument("--sizes", default="224,448,512")
    ap.add_argument("--simdinov2-ckpt", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = json.load(open(os.path.join(args.subraster_dir, "manifest.json")))
    aux_list = build_aux_list(args.split_json, manifest)
    with open(os.path.join(args.out_dir, "aux_index.json"), "w") as f:
        json.dump(aux_list, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aux] device={device}", flush=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    for m in [x.strip() for x in args.models.split(",")]:
        ckpt = args.simdinov2_ckpt if m.startswith("simdinov2") else None
        extract_model_aux(m, ckpt, sizes, args.subraster_dir, manifest, aux_list, args.out_dir, device)


if __name__ == "__main__":
    main()
