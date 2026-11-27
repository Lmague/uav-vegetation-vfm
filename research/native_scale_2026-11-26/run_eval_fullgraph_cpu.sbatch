#!/bin/bash
#SBATCH --job-name=fullgraph_eval
#SBATCH --account=def-bouguess
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fullgraph_eval_%j.out
#SBATCH --error=logs/fullgraph_eval_%j.err
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
python research/native_scale_2026-11-26/eval_fullgraph_all.py \
  --split-json "$CODE_DIR/research/gnn_grid_2026-10-31/out/canonical_split.json" \
  --emb-root "$SCRATCH/native_scale_2026-11-26/embeddings" \
  --base224-root "$SCRATCH/embeddings" \
  --aux-dir "$SCRATCH/native_scale_2026-11-26/aux_embeddings" \
  --out-dir "$SCRATCH/native_scale_2026-11-26/fullgraph" \
  --workers 8 \
  --gnn-threads 8

echo "[slurm] évaluation graphe complet terminée : $(date)"
