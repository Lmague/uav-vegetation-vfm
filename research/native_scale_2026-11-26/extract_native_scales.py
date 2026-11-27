#!/usr/bin/env python3
"""Extraction frozen multi-tailles NATIVES (384, 448, 512, 1024) sur Narval GPU.

Pour chaque (modèle, taille) sur les 80 088 tuiles du split canonique 11 classes :
  - Découpe à la volée depuis les 32 sous-rasters PNG sans perte (1 ortho en RAM à la fois).
  - Passe la fenêtre à résolution NATIVE (aucun redimensionnement) dans le backbone gelé.
  - Extrait en un seul forward DEUX représentations (float16, dim=768) :
      * cls          : représentation globale (pooler_output DINOv3 / x_norm_clstoken SimDINOv2)
      * center_pool  : moyenne des 14×14 patchs correspondant exactement à la tuile 224×224
                       (après self-attention sur tout le contexte).
  - Pour SimDINOv2 : remplace `q @ k.T` par `F.scaled_dot_product_attention` (identique
    mathématiquement, vérifié <1e-3 en fp32 sur 224×224) pour tenir en VRAM à 1024×1024
    sans xformers.
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
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import build_frozen_extractor
from src.utils import get_normalization

TILE_SIZE = 224
PATCH_SIZE = 16
CENTER_PATCHES = TILE_SIZE // PATCH_SIZE  # 14

DEFAULT_BATCH = {
    224: 256,
    384: 128,
    448: 96,
    512: 64,
    1024: 16,
}


def clamp_window(center: int, size: int, dim: int) -> tuple[int, int]:
    if size >= dim:
        return 0, dim
    start = center - size // 2
    if start < 0:
        start = 0
    elif start + size > dim:
        start = dim - size
    return start, size


def patch_simdinov2_sdpa(model: torch.nn.Module) -> int:
    """Remplace le forward d'attention de SimDINOv2 par F.scaled_dot_product_attention."""
    n_patched = 0
    for blk in model.blocks:
        attn = blk.attn

        def _make_sdpa_forward(m):
            def _sdpa_forward(x, attn_bias=None):
                B, N, C = x.shape
                qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, C // m.num_heads)
                q, k, v = qkv.unbind(2)  # (B, N, H, D)
                q = q.transpose(1, 2)  # (B, H, N, D)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                x_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=0.0
                )
                x_out = x_out.transpose(1, 2).reshape(B, N, C)
                x_out = m.proj(x_out)
                x_out = m.proj_drop(x_out)
                return x_out
            return _sdpa_forward

        attn.forward = _make_sdpa_forward(attn)
        n_patched += 1
    return n_patched


@torch.inference_mode()
def forward_cls_and_patches(model, model_name: str, x: torch.Tensor, grid_hw: int):
    """Retourne (cls_vec [B, D], patches_2d [B, grid_hw, grid_hw, D])."""
    if model_name.startswith("dinov3"):
        out = model(pixel_values=x)
        cls_vec = out.pooler_output
        n_reg = int(getattr(model.config, "num_register_tokens", 0))
        patch_toks = out.last_hidden_state[:, 1 + n_reg :, :]
        B, N, D = patch_toks.shape
        assert N == grid_hw * grid_hw, f"DINOv3 patch count {N} != {grid_hw}^2"
        return cls_vec, patch_toks.reshape(B, grid_hw, grid_hw, D)
    elif model_name.startswith("simdinov2"):
        out = model.forward_features(x)
        cls_vec = out["x_norm_clstoken"]
        patch_toks = out["x_norm_patchtokens"]
        B, N, D = patch_toks.shape
        assert N == grid_hw * grid_hw, f"SimDINOv2 patch count {N} != {grid_hw}^2"
        return cls_vec, patch_toks.reshape(B, grid_hw, grid_hw, D)
    else:
        raise ValueError(f"Modèle non supporté : {model_name}")


def verify_sdpa_equivalence(model, device: torch.device) -> None:
    """Vérifie en fp32 sur un tenseur aléatoire 224×224 que le patch SDPA est exact."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.inference_mode():
        ref = model.forward_features(x)["x_norm_clstoken"].clone()
    n = patch_simdinov2_sdpa(model)
    with torch.inference_mode():
        got = model.forward_features(x)["x_norm_clstoken"]
    max_diff = (ref - got).abs().max().item()
    print(f"[sdpa-check] patched {n} blocs SimDINOv2 | max |diff| fp32 = {max_diff:.2e}", flush=True)
    if max_diff > 1e-3:
        raise RuntimeError(f"SDPA patch diverge trop : max_diff={max_diff}")


def load_canonical_index(split_json: str, splits_dir: str):
    """Construit l'ordre exact des tuiles aligné sur splits/{train,val,test}_11cls.csv."""
    with open(split_json) as f:
        d = json.load(f)
    by_key = {t["key"]: t for t in d["tiles"]}

    # Respecte exactement l'ordre de splits/*_11cls.csv pour compatibilité 1:1 avec le benchmark
    import csv
    split_tiles = {}
    for sp in ("train", "val", "test"):
        csv_path = os.path.join(splits_dir, f"{sp}_11cls.csv")
        rows = []
        with open(csv_path) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                key, lbl = row[0], int(row[1])
                meta = by_key[key]
                assert meta["split"] == sp and int(meta["label"]) == lbl
                rows.append((len(rows), meta))
        split_tiles[sp] = rows
        print(f"[index] {sp}: {len(rows)} tuiles alignées sur {sp}_11cls.csv", flush=True)
    return split_tiles


def extract_for_model(
    model_name: str,
    checkpoint: str | None,
    sizes: list[int],
    subraster_dir: str,
    split_tiles: dict,
    out_dir: str,
    device: torch.device,
) -> None:
    with open(os.path.join(subraster_dir, "manifest.json")) as f:
        manifest = json.load(f)

    # Quelles tailles restent à faire ?
    todo_sizes = []
    for sz in sizes:
        tag_dir = os.path.join(out_dir, f"{model_name}_native_{sz}")
        done = all(
            os.path.exists(os.path.join(tag_dir, f"{sp}_{kind}.npy"))
            for sp in ("train", "val", "test")
            for kind in ("cls", "center")
        )
        if done:
            print(f"[skip] {model_name} @ {sz} déjà complet dans {tag_dir}", flush=True)
        else:
            todo_sizes.append(sz)
    if not todo_sizes:
        return

    print(f"\n=== Chargement {model_name} (sizes={todo_sizes}) ===", flush=True)
    model, _fwd, dim, norm_key = build_frozen_extractor(model_name, checkpoint)
    model = model.to(device).eval()
    if model_name.startswith("simdinov2"):
        verify_sdpa_equivalence(model, device)

    mean_t, std_t = get_normalization(norm_key)
    mean = torch.tensor(mean_t, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_t, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    # Grouper toutes les tuiles par ortho pour ne charger chaque PNG qu'UNE SEULE FOIS !
    ortho_items: dict[str, list[tuple[str, int, dict]]] = collections.defaultdict(list)
    for sp in ("train", "val", "test"):
        for idx, meta in split_tiles[sp]:
            ortho_items[meta["ortho"]].append((sp, idx, meta))

    # Allouer les tableaux de sortie (float16) par taille et split
    buffers = {}
    for sz in todo_sizes:
        buffers[sz] = {}
        for sp in ("train", "val", "test"):
            n = len(split_tiles[sp])
            buffers[sz][sp] = {
                "cls": np.empty((n, dim), dtype=np.float16),
                "center": np.empty((n, dim), dtype=np.float16),
            }

    t_model = time.time()
    orthos = sorted(ortho_items.keys())
    for o_idx, ortho in enumerate(orthos, 1):
        t_o = time.time()
        om = manifest["orthos"][ortho]
        orig_h, orig_w = om["orig_h"], om["orig_w"]
        b_r0, b_c0 = om["bbox_row0"], om["bbox_col0"]

        png_path = os.path.join(subraster_dir, f"{ortho}.png")
        arr = np.asarray(Image.open(png_path))  # (H, W, 3) uint8
        items = ortho_items[ortho]

        for sz in todo_sizes:
            bs = DEFAULT_BATCH.get(sz, 32)
            grid_hw = sz // PATCH_SIZE
            n_items = len(items)

            for start in range(0, n_items, bs):
                batch_items = items[start : start + bs]
                cur_b = len(batch_items)
                crops = np.empty((cur_b, sz, sz, 3), dtype=np.uint8)
                pr0s = np.empty(cur_b, dtype=np.int32)
                pc0s = np.empty(cur_b, dtype=np.int32)

                for i, (_sp, _idx, meta) in enumerate(batch_items):
                    tr, tc = meta["tile_row_off"], meta["tile_col_off"]
                    r0, _ = clamp_window(tr + TILE_SIZE // 2, sz, orig_h)
                    c0, _ = clamp_window(tc + TILE_SIZE // 2, sz, orig_w)
                    lr0, lc0 = r0 - b_r0, c0 - b_c0
                    crops[i] = arr[lr0 : lr0 + sz, lc0 : lc0 + sz]
                    # Position exacte en patchs de la tuile centrale 224×224 dans le crop S×S
                    pr0 = int(round((tr - r0) / PATCH_SIZE))
                    pc0 = int(round((tc - c0) / PATCH_SIZE))
                    pr0s[i] = max(0, min(grid_hw - CENTER_PATCHES, pr0))
                    pc0s[i] = max(0, min(grid_hw - CENTER_PATCHES, pc0))

                x = torch.from_numpy(crops).to(device, non_blocking=True)
                x = x.permute(0, 3, 1, 2).float().div_(255.0)
                x = (x - mean) / std

                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    cls_vec, patches_2d = forward_cls_and_patches(model, model_name, x, grid_hw)
                    # Extraction du centre 14×14 : cas rapide si tout le batch a le même offset
                    if np.all(pr0s == pr0s[0]) and np.all(pc0s == pc0s[0]):
                        r_s, c_s = int(pr0s[0]), int(pc0s[0])
                        center_vec = patches_2d[:, r_s : r_s + CENTER_PATCHES, c_s : c_s + CENTER_PATCHES, :].mean(dim=(1, 2))
                    else:
                        center_list = [
                            patches_2d[i, pr0s[i] : pr0s[i] + CENTER_PATCHES, pc0s[i] : pc0s[i] + CENTER_PATCHES, :].mean(dim=(0, 1))
                            for i in range(cur_b)
                        ]
                        center_vec = torch.stack(center_list, dim=0)

                cls_np = cls_vec.float().cpu().numpy().astype(np.float16)
                ctr_np = center_vec.float().cpu().numpy().astype(np.float16)

                for i, (sp, idx, _meta) in enumerate(batch_items):
                    buffers[sz][sp]["cls"][idx] = cls_np[i]
                    buffers[sz][sp]["center"][idx] = ctr_np[i]

        del arr
        gc.collect()
        print(
            f"  [{model_name}] ortho {o_idx:2d}/{len(orthos)} {ortho} ({len(items)} tuiles, "
            f"{len(todo_sizes)} tailles) : {time.time()-t_o:.1f}s",
            flush=True,
        )

    # Sauvegarde disque
    for sz in todo_sizes:
        tag_dir = os.path.join(out_dir, f"{model_name}_native_{sz}")
        os.makedirs(tag_dir, exist_ok=True)
        for sp in ("train", "val", "test"):
            labels = np.array([m["label"] for _, m in split_tiles[sp]], dtype=np.int64)
            np.save(os.path.join(tag_dir, f"{sp}_cls.npy"), buffers[sz][sp]["cls"])
            np.save(os.path.join(tag_dir, f"{sp}_center.npy"), buffers[sz][sp]["center"])
            np.save(os.path.join(tag_dir, f"{sp}_labels.npy"), labels)
        meta_out = {
            "model": model_name,
            "native_size": sz,
            "grid_hw": sz // PATCH_SIZE,
            "center_patches": CENTER_PATCHES,
            "norm_key": norm_key,
            "dim": dim,
            "dtype": "float16",
            "schema": "11cls_no_rhol",
            "split": "spatial_v3",
        }
        with open(os.path.join(tag_dir, "meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2)
        print(f"[saved] {tag_dir} ✓", flush=True)

    del model, buffers
    torch.cuda.empty_cache()
    gc.collect()
    print(f"=== {model_name} terminé en {(time.time()-t_model)/60:.1f} min ===", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subraster-dir", required=True)
    ap.add_argument("--split-json", default="research/gnn_grid_2026-10-31/out/canonical_split.json")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="dinov3_vitb16_lvd,simdinov2_vitb16")
    ap.add_argument("--sizes", default="384,448,512,1024")
    ap.add_argument("--simdinov2-ckpt", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device} | models={args.models} | sizes={args.sizes}", flush=True)
    if device.type == "cuda":
        print(f"[extract] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    split_tiles = load_canonical_index(args.split_json, args.splits_dir)
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for m in models:
        ckpt = args.simdinov2_ckpt if m.startswith("simdinov2") else None
        extract_for_model(m, ckpt, sizes, args.subraster_dir, split_tiles, args.out_dir, device)


if __name__ == "__main__":
    main()
