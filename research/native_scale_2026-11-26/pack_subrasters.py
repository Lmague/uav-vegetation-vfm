#!/usr/bin/env python3
"""Découpe 32 sous-rasters PNG sans perte (1 par ortho du split canonique)
couvrant toutes les fenêtres jusqu'à MAX_SIZE=1024 px autour des 80 088 tuiles.

Réutilise exactement les coordonnées de `canonical_split.json` (aucun rejeu de
tilerization.py) et la même logique de clamp bord que `scripts/context_crop.py`.
Sauvegarde des PNG sans perte (lus nativement par PIL sur Narval sans rasterio).
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import time

import numpy as np
import rasterio
import rasterio.windows
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TILE_SIZE = 224
MAX_CONTEXT = 1024


def clamp_window(center: int, size: int, dim: int) -> tuple[int, int]:
    """Retourne (start, eff_size) pour une fenêtre `size` centrée sur `center` dans [0, dim)."""
    if size >= dim:
        return 0, dim
    start = center - size // 2
    if start < 0:
        start = 0
    elif start + size > dim:
        start = dim - size
    return start, size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-json",
        default="research/gnn_grid_2026-10-31/out/canonical_split.json",
    )
    ap.add_argument(
        "--cog-root",
        default=(
            "/media/erazal/PortableSSD/Mémoire/"
            "High-Resolution Arctic Vegetation Maps and Photogrammetry Data "
            "from Drone Surveys at Trail Valley Creek, Northwest Territories/photogrammetry"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default="/media/erazal/PortableSSD/memoire_work/native_subrasters",
    )
    ap.add_argument("--max-size", type=int, default=MAX_CONTEXT)
    ap.add_argument("--compress-level", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.split_json) as f:
        split_data = json.load(f)

    by_ortho: dict[str, list[dict]] = collections.defaultdict(list)
    for t in split_data["tiles"]:
        by_ortho[t["ortho"]].append(t)

    orthos = sorted(by_ortho.keys())
    print(f"[pack] {len(split_data['tiles'])} tuiles sur {len(orthos)} orthos | max_size={args.max_size}", flush=True)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest: dict = {"max_size": args.max_size, "tile_size": TILE_SIZE, "orthos": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            prev = json.load(f)
            manifest["orthos"].update(prev.get("orthos", {}))

    t_all = time.time()
    total_bytes = 0
    for idx, ortho in enumerate(orthos, 1):
        out_png = os.path.join(args.out_dir, f"{ortho}.png")
        cog_path = os.path.join(args.cog_root, ortho, f"{ortho}_rgb.cog.tif")
        if not os.path.exists(cog_path):
            raise FileNotFoundError(f"COG introuvable : {cog_path}")

        if os.path.exists(out_png) and ortho in manifest["orthos"]:
            sz = os.path.getsize(out_png)
            total_bytes += sz
            print(f"[{idx:2d}/{len(orthos)}] {ortho}: déjà présent ({sz/1e6:.1f} Mo), skip", flush=True)
            continue

        t0 = time.time()
        tiles = by_ortho[ortho]
        with rasterio.open(cog_path) as src:
            orig_h, orig_w = src.height, src.width
            n_bands = src.count
            bands = [1, 2, 3] if n_bands >= 3 else [1]

            r0_min, c0_min = orig_h, orig_w
            r1_max, c1_max = 0, 0
            for t in tiles:
                cr = t["tile_row_off"] + TILE_SIZE // 2
                cc = t["tile_col_off"] + TILE_SIZE // 2
                r0, eh = clamp_window(cr, args.max_size, orig_h)
                c0, ew = clamp_window(cc, args.max_size, orig_w)
                if r0 < r0_min:
                    r0_min = r0
                if c0 < c0_min:
                    c0_min = c0
                if r0 + eh > r1_max:
                    r1_max = r0 + eh
                if c0 + ew > c1_max:
                    c1_max = c0 + ew

            bbox_h = r1_max - r0_min
            bbox_w = c1_max - c0_min
            win = rasterio.windows.Window(c0_min, r0_min, bbox_w, bbox_h)
            arr = src.read(bands, window=win)  # (C, H, W)
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            # Vérification bit-à-bit sur la première tuile de l'ortho
            t_check = tiles[0]
            tr, tc = t_check["tile_row_off"], t_check["tile_col_off"]
            tile_direct = np.transpose(
                src.read(bands, window=rasterio.windows.Window(tc, tr, TILE_SIZE, TILE_SIZE)),
                (1, 2, 0),
            )
            if tile_direct.shape[2] == 1:
                tile_direct = np.repeat(tile_direct, 3, axis=2)
            tile_slice = arr[tr - r0_min : tr - r0_min + TILE_SIZE, tc - c0_min : tc - c0_min + TILE_SIZE]
            assert np.array_equal(tile_direct, tile_slice), f"Mismatch pixel sur {ortho}!"

        img = Image.fromarray(arr, mode="RGB")
        tmp_png = out_png + ".tmp.png"
        img.save(tmp_png, format="PNG", compress_level=args.compress_level)
        os.replace(tmp_png, out_png)
        sz = os.path.getsize(out_png)
        total_bytes += sz

        manifest["orthos"][ortho] = {
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
            "bbox_row0": int(r0_min),
            "bbox_col0": int(c0_min),
            "bbox_h": int(bbox_h),
            "bbox_w": int(bbox_w),
            "n_tiles": len(tiles),
            "png_bytes": int(sz),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        del img, arr
        gc.collect()
        print(
            f"[{idx:2d}/{len(orthos)}] {ortho}: {bbox_h}x{bbox_w} ({len(tiles)} tuiles) "
            f"-> {sz/1e6:.1f} Mo ({time.time()-t0:.1f}s) ✓",
            flush=True,
        )

    print(
        f"[pack] Terminé : {len(orthos)} sous-rasters, {total_bytes/1e9:.2f} Go total "
        f"en {time.time()-t_all:.0f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
