#!/bin/bash

blenderproc run render.py \
    --json_path /cluster/work/riner/users/asgobbi/datasets/objaverse/test.json \
    --shard_idx 0 \
    --num_workers 1 \
    --max_objects 3 \
    --output_dir ./outputs \
    --log_mem \
    --seed 42