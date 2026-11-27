#!/bin/bash
#SBATCH --job-name=native_scale_probe
#SBATCH --account=def-bouguess
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/native_scale_probe_%j.out
#SBATCH --error=logs/native_scale_probe_%j.err
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

EMB_ROOT="$SCRATCH/native_scale_2026-11-26/embeddings"
OUT_DIR="$SCRATCH/native_scale_2026-11-26/probes"

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date)"
python research/native_scale_2026-11-26/probe_native_scales.py \
  --emb-root "$EMB_ROOT" \
  --out-dir "$OUT_DIR" \
  --models "dinov3_vitb16_lvd,simdinov2_vitb16" \
  --sizes "384,448,512,1024" \
  --views "cls,center,center_cls" \
  --workers 8

echo "[slurm] sondes terminées : $(date)"
