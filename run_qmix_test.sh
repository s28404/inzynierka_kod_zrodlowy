#!/bin/bash
#
# Run QMIX algorithm on the easiest SMAC v2 map (5v5)
#

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$(dirname "$0")/.venv/bin/python" fine_tuned/smacv2/smacv2_run.py \
algorithm=qmix \
task=smacv2/protoss_10_vs_10 \
experiment.off_policy_n_envs_per_worker=2 \
experiment.off_policy_collected_frames_per_batch=1000 \
experiment.parallel_collection=false \
experiment.render=false \
experiment.buffer_device=cpu \
experiment.off_policy_memory_size=50000 \
experiment.max_n_frames=100000
