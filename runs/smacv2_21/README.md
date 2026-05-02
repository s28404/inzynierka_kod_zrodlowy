# SMACv2 21-Computer Training Setup

## Overview

This directory contains scripts for distributed training of 21 independent experiments across 21 computers. Each computer runs exactly **one** experiment independently, with automatic resource management (RAM/GPU detection) to avoid OOM errors.

**Total experiments to run:** 21 (distributed 1 per computer)

| Map | Algorithm | Seeds | Count |
|-----|-----------|-------|-------|
| protoss_5_vs_5 | QMIX, DEMIR, NGU, RND | 100, 101, 102 | 12 |
| protoss_10_vs_11 | QMIX, DEMIR, RND | 100, 101, 102 | 9 |

## Quick Start

### On Each Computer (Machine 1-21)

Pick one of the numbered scripts and run it. **Run ONLY ONE per computer**:

```bash
cd /home/s28404/projects/inzynierka_kod_zrodlowy
./runs/smacv2_21/01_protoss_5_vs_5_qmix_seed100.sh     # Computer 1
./runs/smacv2_21/02_protoss_5_vs_5_qmix_seed101.sh     # Computer 2
./runs/smacv2_21/03_protoss_5_vs_5_qmix_seed102.sh     # Computer 3
# ... and so on up to 21
```

Each script will:
1. ✓ Detect GPU/RAM availability
2. ✓ Auto-tune batch sizes based on hardware
3. ✓ Fall back to CPU if GPU unavailable (but will warn)
4. ✓ Use hyperparameters from `fine_tuned/smacv2/conf/config.yaml`
5. ✓ Log to W&B, CSV, and JSON
6. ✓ Save checkpoints periodically
7. ✓ Create experiment output in `outputs/YYYY-MM-DD/HH-MM-SS/`

### Hyperparameters

**Default DEMIR:** `beta1=0.7, beta2=0.3, encoder=IDM`

All training hyperparameters (LR, max_frames, batch sizes, etc.) are inherited from:
- `fine_tuned/smacv2/conf/config.yaml` (main config)
- `benchmarl/conf/algorithm/demir.yaml` (DEMIR-specific)
- `benchmarl/conf/algorithm/qmix.yaml` (QMIX-specific)
- `benchmarl/conf/algorithm/ngu.yaml` (NGU-specific)
- `benchmarl/conf/algorithm/rnd.yaml` (RND-specific)

### Environment Variables (Optional Tuning)

You can override defaults before running:

```bash
# Auto-tune batch sizes based on RAM/GPU (default: enabled)
AUTO_TUNE=1 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh

# Disable auto-tuning (use config defaults only)
AUTO_TUNE=0 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh

# Override specific batch size (without re-tuning everything)
OFF_POLICY_TRAIN_BATCH_SIZE=96 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh

# Require GPU (fail if unavailable)
REQUIRE_GPU=1 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh

# Allow CPU fallback if GPU insufficient
REQUIRE_GPU=0 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh

# Set memory thresholds (MB)
MIN_RAM_MB=12000 MIN_GPU_FREE_MB=6000 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh
```

## Collecting Results

### After Training Completes

After each training finishes, collect all artifacts (checkpoints, logs, CSV, JSON) into a ZIP file for easy download:

```bash
# On the same computer where training finished:
./runs/smacv2_21/collect_results.sh "protoss_5_vs_5_qmix_seed100"

# Output: protoss_5_vs_5_qmix_seed100.zip (in current directory)
```

Or collect all 21 at once (if running on a central machine):

```bash
./runs/smacv2_21/collect_all_21_results.sh /tmp/smacv2_results
# Creates 21 ZIP files in /tmp/smacv2_results/
```

### ZIP File Contents

Each ZIP contains:
```
EXPERIMENT_NAME.zip
├── outputs/
│   ├── checkpoints/           ← Model snapshots (checkpoint_1000.pt, etc.)
│   ├── config.pkl             ← Full Hydra + algorithm config
│   ├── EXPERIMENT_NAME.json   ← Experiment metadata + final metrics
│   └── wandb/                 ← W&B logs (if enabled)
├── logs_thesis/
│   ├── *.csv                  ← Training metrics per frame/step
│   └── *.jsonl                ← Line-delimited JSON logs
├── smacv2_run.log             ← Stdout/stderr log
└── README.txt                 ← Quick reference guide
```

### Downloading ZIP Files from Cluster

From your local machine:

```bash
scp s28404@host1:/path/to/protoss_5_vs_5_qmix_seed100.zip ./results/
scp s28404@host2:/path/to/protoss_5_vs_5_qmix_seed101.zip ./results/
# ... repeat for all 21 hosts

# Or use a script:
for i in {1..21}; do
  host="host$i"
  scp s28404@$host:~/projects/inzynierka_kod_zrodlowy/protoss_*.zip ./results/
done
```

## Analyzing Results

After downloading all 21 ZIPs:

```bash
# 1. Extract all ZIPs
unzip -q '*.zip'

# 2. Aggregate metrics into single CSV (for plotting/comparison)
python3 runs/smacv2_21/aggregate_thesis_results.py . thesis_results.csv

# 3. Open thesis_results.csv in Excel, Python, or R for:
#    - Plotting learning curves (frames vs return)
#    - Comparing algorithms (QMIX vs DEMIR vs NGU vs RND)
#    - Statistical analysis
```

### CSV Columns for Analysis

| Column | Meaning | Notes |
|--------|---------|-------|
| `task` | SMACv2 map | `protoss_5_vs_5`, `protoss_10_vs_11` |
| `algorithm` | Algorithm name | `qmix`, `demir`, `ngu`, `rnd` |
| `seed` | Random seed | 100, 101, 102 |
| `variant` | Variant/ablation | `baseline`, `no_barlow_twins`, `mlp_encoder`, `beta1_0`, `beta2_0` |
| `final_return_mean` | Final evaluation return | Most recent eval |
| `final_return_std` | Std dev of final return | Across eval episodes |
| `final_win_rate` | Final win rate on SMACv2 | For comparison with paper |
| `max_return` | Best return seen | Peak performance |
| `total_frames` | Training frames at end | Indicates training duration |
| `checkpoints` | # of saved checkpoints | Useful for model selection |

### Example Analysis in Python

```python
import pandas as pd

df = pd.read_csv('thesis_results.csv')

# Compare algorithms on protoss_5_vs_5
results = df[df['task'] == 'protoss_5_vs_5'].groupby('algorithm').agg({
    'final_return_mean': ['mean', 'std'],
    'final_win_rate': ['mean', 'std']
}).round(3)
print(results)

# DEMIR vs baseline
demir = df[df['algorithm'] == 'demir']['final_return_mean'].mean()
qmix = df[df['algorithm'] == 'qmix']['final_return_mean'].mean()
improvement = (demir - qmix) / qmix * 100
print(f"DEMIR improvement over QMIX: {improvement:.1f}%")
```

## Troubleshooting

### OOM Error
- ✓ Script will detect and fail gracefully with error code 3
- Check available RAM: `free -h`
- Check GPU memory: `nvidia-smi`
- Increase `MIN_RAM_MB` threshold or `REQUIRE_GPU=0` for CPU fallback

### GPU Out of Memory
- Script will fall back to CPU or fail with error code 4
- Check GPU: `nvidia-smi`
- Try: `OFF_POLICY_TRAIN_BATCH_SIZE=64` (smaller batch)
- Or: `OFF_POLICY_MEMORY_SIZE=40000` (smaller replay buffer)

### Experiment Not Starting
- Check experiment directory name: `ls outputs/`
- Ensure `fine_tuned/smacv2/conf/config.yaml` exists and is readable
- Check Python version: `python3 --version` (need 3.8+)

### W&B Sync Issues
- Set `experiment.create_json=true` to ensure JSON logs anyway
- CSV metrics will still be generated regardless of W&B

## Script Descriptions

| Script | Purpose |
|--------|---------|
| `run_one.sh` | Base runner for single experiment (handles GPU/RAM/batch-tuning) |
| `01_*.sh` - `21_*.sh` | Task-specific wrappers (calls `run_one.sh` with parameters) |
| `collect_results.sh` | Bundles single experiment outputs into ZIP for download |
| `collect_all_21_results.sh` | Collects all 21 experiments into separate ZIPs |
| `aggregate_thesis_results.py` | Combines CSV/JSON from multiple experiments into analysis file |

## Important Notes

1. **Each script is independent** — they don't depend on each other
2. **Auto-tuning is safe** — thresholds are conservative (50% GPU, 70% RAM)
3. **Checkpoints are always saved** — `checkpoint_interval=1000000` frames
4. **W&B is optional** — JSON and CSV logging always work
5. **Results are portable** — ZIP files contain everything needed for reproduction/analysis

## Contact & Questions

- For experiment setup issues: check `fine_tuned/smacv2/conf/config.yaml`
- For metrics definition: see `benchmarl/experiment/thesis_logger.py`
- For reproducibility: save the `config.pkl` from outputs
- For model inference: use `checkpoints/checkpoint_*.pt` files

---

**Last updated:** 2026-05-02  
**Author:** Kajetan Frąckowiak  
**Project:** Engineering Thesis — PJATK
