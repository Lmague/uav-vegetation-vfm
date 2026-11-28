# Makefile for Arctic-TVC public benchmark repository
#
# Targets:
#   figures  - rebuild all 10 README figures from committed scalar tables (CPU, no torch)
#   test     - run all repository guards and unit tests
#   compile  - syntax-check all tracked Python modules and scripts
#   status   - check presence of key data files and README artifacts
#
# Canonical linear-probe numbers require single-thread BLAS to avoid L-BFGS
# floating-point reduction drift (~0.0015 F1). All 5 thread variables are exported here.

export OMP_NUM_THREADS        := 1
export OPENBLAS_NUM_THREADS   := 1
export MKL_NUM_THREADS        := 1
export NUMEXPR_NUM_THREADS    := 1
export VECLIB_MAXIMUM_THREADS := 1

PY ?= python3

.PHONY: help figures test compile status

help:
	@echo "Targets: figures | test | compile | status"

figures:
	$(PY) scripts/make_readme_plots.py

compile:
	$(PY) -m compileall -q src scripts research results/baseline_clean_2026-10-08/run_baseline_probe.py \
	  probe.py extract.py analyze.py train.py make_figures.py

test: compile
	$(PY) tests/test_generate_splits_11cls.py
	$(PY) tests/test_texture_features.py
	$(PY) tests/test_core.py
	$(PY) tests/test_repo_guards.py

status:
	@echo "=== Key artifacts referenced by README.md ==="
	@for f in results/baseline_clean_2026-10-08/per_seed.csv \
	          results/baseline_clean_2026-10-08/per_class_f1.csv \
	          results/baseline_clean_2026-10-08/summary.csv \
	          results/baseline_clean_2026-10-08/statistics/pairwise_effects.csv \
	          results/tile_context_2026-11-01/canonical_probe_arms.json \
	          results/tile_context_2026-11-01/gnn_canonical.json \
	          results/tile_context_2026-11-01/arms_8cls.csv \
	          research/tile_context_fullgraph_2026-11-26/out/multibackbone_smoothing_fullgraph.csv \
	          research/tile_context_fullgraph_2026-11-26/out/peft_pairs_fullgraph.csv \
	          research/gnn_grid_2026-10-31/out/decoding_full/weighted_fullgraph.json \
	          research/gnn_grid_2026-10-31/out/decoding_full/weighted_fullgraph_r2.json \
	          results/rapport_data/loo_triage.csv \
	          results/rapport_data/geometry_models.csv \
	          results/rapport_data/dim_probe_summary.csv \
	          results/datacurve_lora/results_agg.csv \
	          lora_spatial_v2/results_spatial_summary.csv \
	          spatial_datacurve/class_composition.csv \
	          splits/train_11cls.csv \
	          figures/readme/01_leaderboard.png ; do \
	    [ -f "$$f" ] && printf "  OK       %s\n" "$$f" || printf "  MISSING  %s\n" "$$f" ; \
	done
