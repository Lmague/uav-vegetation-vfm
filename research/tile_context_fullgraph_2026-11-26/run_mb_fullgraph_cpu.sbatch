#!/bin/bash
#SBATCH --job-name=mb_fullgraph
#SBATCH --account=def-bouguess
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/mb_fullgraph_%j.out
#SBATCH --error=logs/mb_fullgraph_%j.err
set -euo pipefail

module load python/3.11
source "$HOME/ENV/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CODE_DIR="$HOME/benchmark-memoire"
cd "$CODE_DIR"
mkdir -p logs

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date)"
python research/tile_context_fullgraph_2026-11-26/run_multibackbone_fullgraph.py \
  --split-json "$CODE_DIR/research/gnn_grid_2026-10-31/out/canonical_split.json" \
  --canonical-mb-dir "$SCRATCH/context_multibackbone/out" \
  --aux-dir "$SCRATCH/tile_context_fullgraph_2026-11-26/aux_embeddings" \
  --aux-index "$SCRATCH/native_scale_2026-11-26/aux_embeddings/aux_index.json" \
  --out-dir "$SCRATCH/tile_context_fullgraph_2026-11-26/out" \
  --workers 8 \
  --boot 10000

echo "[slurm] multibackbone fullgraph terminé : $(date)"
