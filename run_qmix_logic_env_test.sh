#!/bin/bash
#
# Copyright (c) 2026 Kajetan Frąckowiak, s28404
#
# Projekt: Algorytm DEMIR dla SMACv2 i Custom Logic Environment
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Praca Inżynierska (2026)
#
# Opis: Skrypt treningowy dla QMIX algorithm na Logic Environment (SynchronizedFactory).
#

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$(dirname "$0")/.venv/bin/python" fine_tuned/logic_env/logic_env_run.py \
algorithm=demir \
task=logic_env/synchronized \
experiment.off_policy_n_envs_per_worker=4 \
experiment.off_policy_collected_frames_per_batch=1000 \
experiment.parallel_collection=false \
experiment.render=false \
experiment.buffer_device=cpu \
experiment.off_policy_memory_size=200000 \
experiment.max_n_frames=10000000 \
seed=1

# Run QMIX algorithm
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$(dirname "$0")/.venv/bin/python" fine_tuned/logic_env/logic_env_run.py \
algorithm=qmix \
task=logic_env/synchronized \
experiment.off_policy_n_envs_per_worker=4 \
experiment.off_policy_collected_frames_per_batch=1000 \
experiment.parallel_collection=false \
experiment.render=false \
experiment.buffer_device=cpu \
experiment.off_policy_memory_size=200000 \
experiment.max_n_frames=10000000 \
seed=1

# Run NGU algorithm
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$(dirname "$0")/.venv/bin/python" fine_tuned/logic_env/logic_env_run.py \
algorithm=ngu \
task=logic_env/synchronized \
experiment.off_policy_n_envs_per_worker=4 \
experiment.off_policy_collected_frames_per_batch=1000 \
experiment.parallel_collection=false \
experiment.render=false \
experiment.buffer_device=cpu \
experiment.off_policy_memory_size=200000 \
experiment.max_n_frames=10000000 \
seed=1

# Run RND algorithm
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$(dirname "$0")/.venv/bin/python" fine_tuned/logic_env/logic_env_run.py \
algorithm=rnd \
task=logic_env/synchronized \
experiment.off_policy_n_envs_per_worker=4 \
experiment.off_policy_collected_frames_per_batch=1000 \
experiment.parallel_collection=false \
experiment.render=false \
experiment.buffer_device=cpu \
experiment.off_policy_memory_size=200000 \
experiment.max_n_frames=10000000 \
seed=1


