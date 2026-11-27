#!/bin/bash
#SBATCH --job-name=fuse224_native
#SBATCH --account=def-bouguess
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fuse224_native_%j.out
#SBATCH --error=logs/fuse224_native_%j.err
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
python research/native_scale_2026-11-26/probe_fuse224_native.py \
  --split-json "$CODE_DIR/research/gnn_grid_2026-10-31/out/canonical_split.json" \
  --emb-root "$SCRATCH/native_scale_2026-11-26/embeddings" \
  --base224-root "$SCRATCH/embeddings" \
  --out-json "$SCRATCH/native_scale_2026-11-26/followup/fuse224_native.json" \
  --workers 8

echo "[slurm] fuse224 terminé : $(date)"
