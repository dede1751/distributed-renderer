#!/bin/bash
#SBATCH --job-name=Google
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --array=0-99
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_logs/google/google_%A_%a.out
#SBATCH --error=./slurm_logs/google/google_%A_%a.err

################### Change these! ################### 
BASE_DIR="/cluster/work/riner/users/asgobbi/datasets"
JSON_FILE="$BASE_DIR/lists/google_dedup.json"
OUTPUT_DIR="$BASE_DIR/renders/google"
TOTAL_JOBS=100
#####################################################

echo "Running job $SLURM_ARRAY_TASK_ID of $TOTAL_JOBS."
blenderproc run render.py \
    --json_file $JSON_FILE \
    --shard_idx $SLURM_ARRAY_TASK_ID \
    --num_workers $TOTAL_JOBS \
    --log_resources \
    --output_dir $OUTPUT_DIR
#    --config config/config.json \
#    --max_objects 10 \
#    --seed 42 \
