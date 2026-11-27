#!/bin/bash
#SBATCH --job-name=native_scale_ext
#SBATCH --account=def-bouguess_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/native_scale_ext_%j.out
#SBATCH --error=logs/native_scale_ext_%j.err
set -euo pipefail

module load python/3.11 cuda/12.2 cudnn/8.9
source "$HOME/ENV/bin/activate"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export TOKENIZERS_PARALLELISM=false

CODE_DIR="$HOME/benchmark-memoire"
cd "$CODE_DIR"
mkdir -p logs

SUBRASTERS="$SCRATCH/native_scale_2026-11-26/subrasters"
OUT_DIR="$SCRATCH/native_scale_2026-11-26/embeddings"

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python research/native_scale_2026-11-26/extract_native_scales.py \
  --subraster-dir "$SUBRASTERS" \
  --split-json "$CODE_DIR/research/gnn_grid_2026-10-31/out/canonical_split.json" \
  --splits-dir "$CODE_DIR/splits" \
  --out-dir "$OUT_DIR" \
  --models "dinov3_vitb16_lvd,simdinov2_vitb16" \
  --sizes "384,448,512,1024" \
  --simdinov2-ckpt "$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth"

echo "[slurm] extraction terminée : $(date)"
