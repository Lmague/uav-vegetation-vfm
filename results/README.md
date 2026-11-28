# results/

Scalar outputs and statistical tables supporting the figures and tables in the root `README.md`.

## Main directories

- `baseline_clean_2026-10-08/`: authoritative frozen benchmark baseline (230 linear probes, 11-class schema, single-thread BLAS, 0 failures). Contains normative protocols (`BASELINE_PROTOCOL.md`, `QUALITY_CONTROL.md`, `statistics/STATISTICAL_PROTOCOL.md`), per-seed and per-class tables (`summary.csv`, `per_seed.csv`, `per_class_f1.csv`), paired hierarchical bootstrap results (`statistics/pairwise_effects.csv`, `bh_summary.csv`, `bootstrap_summary.csv`), and saved predictions (`preds/*.npz`).
- `tile_context_2026-11-01/`: spatial context, smoothing, learned aggregation (3x3 attention, 3x3 CNN, GATv2), 8-class arm comparisons (`arms_8cls.csv`), smoothing strata (`strata.json`), and neighbourhood scale (`neighbourhood_scale.json`).
- `rapport_data/`: embedding geometry and transferability analyses (`loo_triage.csv`, `geometry_models.csv`, `dim_probe_summary.csv`, `head_sweep_summary.csv`, and exploratory SMARTIES-L tables).
- `datacurve_lora/`: random-subset learning curves for DINOv3-B LoRA and frozen linear probes (`results_agg.csv`, `frozen_probe_results_agg.csv`).
- `texture/`: classical colour and texture descriptor ablation and redundancy JSONs.
- `leo_explora_ssl/`: self-supervised continued pretraining probe trajectories (Léo aerial tiles probed on Arctic-TVC).

Related deployable full-graph (8-neighbour) and native-resolution control outputs live under `research/tile_context_fullgraph_2026-11-26/out/`, `research/gnn_grid_2026-10-31/out/decoding_full/`, and `research/native_scale_2026-11-26/`.

## Rebuilding README figures

```bash
pip install -r requirements-figures.txt
python scripts/make_readme_plots.py
```
