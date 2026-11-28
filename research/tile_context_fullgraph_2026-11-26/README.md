# Deployable 8-neighbour spatial context and smoothing

Full replication of the 8-backbone smoothing table (`out/multibackbone_smoothing_fullgraph.csv`) and the 3 PEFT-vs-frozen pairs (`out/peft_pairs_fullgraph.csv`) on the complete 8-neighbour graph (80,088 annotated tiles plus 40,076 off-transect auxiliary tiles = 120,164 nodes, no annotation mask at inference).

## Scripts

- `extract_aux_8backbones.py`: extracts frozen and seed-0 LoRA embeddings for the 40,076 off-transect auxiliary windows across all 8 backbones (alignment verified against canonical tile embeddings at cosine >= 0.99989).
- `run_multibackbone_fullgraph.py`: fits canonical linear probes (single-thread BLAS), runs uniform mean-field, ICM, and confidence-weighted smoothing (`gamma = 4`), and computes paired hierarchical bootstrap 95% CIs (10,000 replicates over orthomosaics then tiles).

## Outputs

- `out/multibackbone_smoothing_fullgraph.csv`: 8 backbones (`R0`, `R0_MF`, `R0_Wconf`, `R3`, `R3_MF`, `R3_Wconf`, 8-class scores, and paired 95% CIs).
- `out/peft_pairs_fullgraph.csv`: 3 PEFT-vs-frozen pairs (`DINOv3-L`, `SimDINOv2-B`, `SimDINOv2-L`, seed 0).

## Summary

1. The best deployable score reaches 0.5089 (11 classes) and 0.6998 (8 classes) with frozen DINOv3-B (`3x3 average + confidence-weighted smoothing`), followed by frozen SimDINOv2-L (0.5076), DINOv3-L (0.5065), and DINOv3-H+ single tile + confidence smoothing (0.5053), matching the frozen 512 px zoom-out fusion baseline (0.5058).
2. Smoothing improves macro F1 across all 8 backbones (all 95% CIs exclude zero): +0.013 to +0.023 for uniform mean-field (p <= 0.015), and +0.012 to +0.031 for confidence-weighted smoothing (p <= 0.010).
3. The small single-tile PEFT edge disappears and turns slightly negative across all 3 pairs under confidence-weighted smoothing (-0.0034 to -0.0086).
4. Restricting smoothing to annotated transect neighbours only (6.53 neighbours/tile) inflates F1 to 0.5306-0.5394 (+0.025 to +0.035), and neither a domain classifier (val AUC 0.642, test F1 0.4967) nor a supervised edge-utility classifier (val AUC 0.792, test F1 0.4975) recovers that gap over simple confidence weighting (0.4978; see `research/gnn_grid_2026-10-31/out/decoding_full/weighted_fullgraph.json`).
