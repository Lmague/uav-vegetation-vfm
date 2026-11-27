#!/bin/bash
#SBATCH --job-name=native_followup
#SBATCH --account=def-bouguess
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=logs/native_followup_%j.out
#SBATCH --error=logs/native_followup_%j.err
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
python research/native_scale_2026-11-26/analyze_native_followup.py \
  --split-json "$CODE_DIR/research/gnn_grid_2026-10-31/out/canonical_split.json" \
  --emb-root "$SCRATCH/native_scale_2026-11-26/embeddings" \
  --base224-root "$SCRATCH/embeddings" \
  --out-dir "$SCRATCH/native_scale_2026-11-26/followup" \
  --workers 8 \
  --gnn-threads 8

echo "[slurm] followup terminé : $(date)"
