#!/bin/bash

python runner.py \
    --json_file /cluster/work/riner/users/asgobbi/datasets/lists/test.json \
    --shard_idx 0 \
    --shard_offset 2 \
    --num_workers 1 \
    --max_objects 3 \
    --obj_per_rerun 2 \
    --output_dir ./outputs \
    --log_resources \
    --seed 42 \
    --config config/dbg.json 
