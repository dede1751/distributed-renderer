#!/bin/bash
#SBATCH --job-name=ObjaverseRender
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --array=0-49
#SBATCH --time=36:00:00
# Deploy 50 renderers using 8CPU and 8G per cpu.

TOTAL_JOBS=50
echo "Running job ${SLURM_ARRAY_TASK_ID} of ${TOTAL_JOBS}."

blenderproc run render.py \
    --data_path /cluster/work/riner/users/asgobbi/datasets/objaverse \
    --shard_idx ${SLURM_ARRAY_TASK_ID} \
    --num_workers ${TOTAL_JOBS} \
    --output_dir /cluster/work/riner/users/asgobbi/datasets/objaverse/renders \
    --seed 42
