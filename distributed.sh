#!/bin/bash
#SBATCH --job-name=ShapeNet
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --array=0-49
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_logs/shapenet/shapenet_%A_%a.out
#SBATCH --error=./slurm_logs/shapenet/shapenet_%A_%a.err

# Deploy 50 renderers using 4CPU and 6G per cpu each.

TOTAL_JOBS=50
echo "Running job ${SLURM_ARRAY_TASK_ID} of ${TOTAL_JOBS}."

# Change these!
JSON_PATH=/cluster/work/riner/users/asgobbi/datasets/shapenetcore-glb/shapenet.json
RENDER_DIR=/cluster/work/riner/users/asgobbi/datasets/renders/shapenet

blenderproc run render.py \
    --json_path ${JSON_PATH} \
    --shard_idx ${SLURM_ARRAY_TASK_ID} \
    --num_workers ${TOTAL_JOBS} \
    --output_dir ${RENDER_DIR}
#    --config config.config.json \
#    --max_objects 10 \
#    --log_mem \
#    --seed 42 \
