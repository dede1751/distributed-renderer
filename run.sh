#!/bin/bash

blenderproc run render.py \
    --data_path /cluster/work/riner/users/asgobbi/datasets/objaverse \
    --shard_idx 0 \
    --num_workers 1 \
    --max_objects 1 \
    --output_dir ./outputs \
    --seed 42