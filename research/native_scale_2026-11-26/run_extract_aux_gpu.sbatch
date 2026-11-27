#!/bin/bash
#SBATCH --job-name=aux_native_ext
#SBATCH --account=def-bouguess_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/aux_native_ext_%j.out
#SBATCH --error=logs/aux_native_ext_%j.err
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

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python research/native_scale_2026-11-26/extract_aux_neighbors.py \
  --subraster-dir "$SCRATCH/native_scale_2026-11-26/subrasters" \
  --split-json "$CODE_DIR/research/gnn_grid_2026-10-31/out/canonical_split.json" \
  --out-dir "$SCRATCH/native_scale_2026-11-26/aux_embeddings" \
  --models "dinov3_vitb16_lvd,simdinov2_vitb16" \
  --sizes "224,448,512" \
  --simdinov2-ckpt "$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth"

echo "[slurm] extraction AUX terminée : $(date)"
