#!/bin/bash
set -x
export GPU=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc-per-node=$GPU --master-port 46662 --master-addr 0.0.0.0 run_gen.py ${@:1}
