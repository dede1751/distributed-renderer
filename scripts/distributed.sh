#!/bin/bash
#SBATCH --job-name=Google
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --array=0-99
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_logs/google/google_%A_%a.out
#SBATCH --error=./slurm_logs/google/google_%A_%a.err

TOTAL_JOBS=100
echo "Running job ${SLURM_ARRAY_TASK_ID} of ${TOTAL_JOBS}."

# Change these!
JSON_PATH=/cluster/work/riner/users/asgobbi/datasets/objaverse/google_dedup.json
RENDER_DIR=/cluster/work/riner/users/asgobbi/datasets/renders/google

blenderproc run render.py \
    --json_path ${JSON_PATH} \
    --shard_idx ${SLURM_ARRAY_TASK_ID} \
    --num_workers ${TOTAL_JOBS} \
    --output_dir ${RENDER_DIR}
#    --config config.config.json \
#    --max_objects 10 \
#    --log_mem \
#    --seed 42 \
