# Setup Complete: 21-Experiment Training Infrastructure

## What Has Been Created

### 1. **Training Scripts** (`runs/smacv2_21/`)

```
✓ run_one.sh                      Base runner with auto-tuning & resource management
✓ 01_protoss_5_vs_5_qmix_seed100.sh
✓ 02_protoss_5_vs_5_qmix_seed101.sh
✓ ... (21 total)
✓ 21_protoss_10_vs_11_rnd_seed102.sh
```

**Each script:**
- Runs independently (1 per computer)
- Auto-detects GPU/RAM availability
- Falls back to CPU if GPU insufficient (or fails with `REQUIRE_GPU=1`)
- Auto-tunes batch sizes based on hardware (20-CPU → bigger batches; fewer CPUs → smaller)
- Inherits all LR/hyperparams from `fine_tuned/smacv2/conf/config.yaml` (unchanged)
- Logs to W&B + CSV + JSON
- Saves checkpoints to `outputs/YYYY-MM-DD/HH-MM-SS/`

### 2. **Results Collection** (`runs/smacv2_21/`)

```
✓ collect_results.sh              Bundle 1 experiment into ZIP
✓ collect_all_21_results.sh       Bundle all 21 experiments into separate ZIPs
✓ aggregate_thesis_results.py     Merge all CSV/JSON into single analysis file
```

**Workflow:**
1. After training: `./collect_results.sh "experiment_name"`
2. Download ZIP files from clusters
3. Extract all ZIPs
4. Run aggregator: `python3 aggregate_thesis_results.py . results.csv`
5. Analyze in Excel/Python/R

### 3. **Documentation**

```
✓ README.md                       Complete guide (quick start, troubleshooting, analysis)
```

## Experiment Breakdown

### Map: `protoss_5_vs_5` (12 experiments)

| # | Script | Algorithm | Seed | Purpose |
|---|--------|-----------|------|---------|
| 1 | `01_protoss_5_vs_5_qmix_seed100.sh` | QMIX | 100 | Baseline |
| 2 | `02_protoss_5_vs_5_qmix_seed101.sh` | QMIX | 101 | Baseline |
| 3 | `03_protoss_5_vs_5_qmix_seed102.sh` | QMIX | 102 | Baseline |
| 4 | `04_protoss_5_vs_5_demir_seed100.sh` | DEMIR | 100 | β₁=0.7, β₂=0.3, IDM |
| 5 | `05_protoss_5_vs_5_demir_seed101.sh` | DEMIR | 101 | β₁=0.7, β₂=0.3, IDM |
| 6 | `06_protoss_5_vs_5_demir_seed102.sh` | DEMIR | 102 | β₁=0.7, β₂=0.3, IDM |
| 7 | `07_protoss_5_vs_5_ngu_seed100.sh` | NGU | 100 | Never Give Up baseline |
| 8 | `08_protoss_5_vs_5_ngu_seed101.sh` | NGU | 101 | Never Give Up baseline |
| 9 | `09_protoss_5_vs_5_ngu_seed102.sh` | NGU | 102 | Never Give Up baseline |
| 10 | `10_protoss_5_vs_5_rnd_seed100.sh` | RND | 100 | Random Network Distillation |
| 11 | `11_protoss_5_vs_5_rnd_seed101.sh` | RND | 101 | Random Network Distillation |
| 12 | `12_protoss_5_vs_5_rnd_seed102.sh` | RND | 102 | Random Network Distillation |

### Map: `protoss_10_vs_11` (9 experiments)

| # | Script | Algorithm | Seed | Purpose |
|---|--------|-----------|------|---------|
| 13 | `13_protoss_10_vs_11_qmix_seed100.sh` | QMIX | 100 | Baseline |
| 14 | `14_protoss_10_vs_11_qmix_seed101.sh` | QMIX | 101 | Baseline |
| 15 | `15_protoss_10_vs_11_qmix_seed102.sh` | QMIX | 102 | Baseline |
| 16 | `16_protoss_10_vs_11_demir_seed100.sh` | DEMIR | 100 | β₁=0.7, β₂=0.3, IDM |
| 17 | `17_protoss_10_vs_11_demir_seed101.sh` | DEMIR | 101 | β₁=0.7, β₂=0.3, IDM |
| 18 | `18_protoss_10_vs_11_demir_seed102.sh` | DEMIR | 102 | β₁=0.7, β₂=0.3, IDM |
| 19 | `19_protoss_10_vs_11_rnd_seed100.sh` | RND | 100 | Random Network Distillation |
| 20 | `20_protoss_10_vs_11_rnd_seed101.sh` | RND | 101 | Random Network Distillation |
| 21 | `21_protoss_10_vs_11_rnd_seed102.sh` | RND | 102 | Random Network Distillation |

## Key Configuration Details

### DEMIR Hyperparameters (Current)
- `beta1: 0.7` — Weight of quality signal (interaction importance)
- `beta2: 0.3` — Weight of novelty signal (state exploration)
- `encoder_type: idm` — Inverse Dynamics Model for representation learning
- `demir_scale: 0.05` — Scale factor for intrinsic reward

Set in: `benchmarl/conf/algorithm/demir.yaml`

### Training Hyperparameters (Inherited from Config)
- `learning_rate: 0.00005` — From `fine_tuned/smacv2/conf/config.yaml`
- `max_n_frames: 10,000,000` — Total training frames
- `off_policy_memory_size: 150,000` (auto-tuned, can be 40K-150K)
- `off_policy_train_batch_size: 64-128` (auto-tuned based on GPU)
- `evaluation_interval: 250,000` frames
- `evaluation_episodes: 32` — Per evaluation

### Auto-Tuning Strategy

**For `off_policy_n_envs_per_worker`** (based on CPU cores):
- ≥16 cores → 8 envs
- 8-15 cores → 6 envs
- <8 cores → 4 envs

**For `off_policy_collected_frames_per_batch`** (scales with n_envs):
- n_envs=8 → 1000 frames
- n_envs=6 → 750 frames
- n_envs=4 → 500 frames

**For `off_policy_train_batch_size`** (based on GPU memory):
- GPU ≥7GB → 128 (batch)
- GPU 5.5-7GB → 96 (batch)
- GPU <5.5GB → 64 (batch)
- CPU-only → 64 (batch)

**For `off_policy_memory_size`** (replay buffer size):
- GPU ≥7GB → 80K transitions
- GPU <7GB → 40K transitions
- RAM ≥14GB (CPU buf) → 150K transitions
- RAM 11-14GB (CPU buf) → 120K transitions
- RAM 9-11GB (CPU buf) → 80K transitions
- RAM <9GB (CPU buf) → 40K transitions

## Running on 21 Computers

### Recommended Setup

**Option 1: SSH to each machine and run**
```bash
ssh user@host-01
cd ~/projects/inzynierka_kod_zrodlowy
./runs/smacv2_21/01_protoss_5_vs_5_qmix_seed100.sh &
exit
```

**Option 2: Parallel SSH dispatch**
```bash
# From a head node, dispatch all 21 jobs
for i in {1..21}; do
  host="user@host-$i"
  script="./runs/smacv2_21/$(printf "%02d" $i)_*.sh"
  ssh $host "cd ~/projects && $script" &
done
wait
```

**Option 3: Use existing dispatcher (if available)**
```bash
CLUSTER_HOSTS_FILE=runs/full/hosts.school.txt python3 runs/full/dispatch_smacv2_jobs.py
```

## Output Structure

After training, each machine creates:

```
outputs/
└── 2026-05-02/
    └── 12-34-56/
        ├── qmix_smacv2_protoss_5_vs_5_seed100_2026-05-02-12_34_56/
        │   ├── checkpoints/
        │   │   ├── checkpoint_1000000.pt
        │   │   ├── checkpoint_2000000.pt
        │   │   └── ...
        │   ├── config.pkl                ← Full config
        │   ├── qmix_smacv2_...json       ← Metrics JSON
        │   └── wandb/                    ← W&B logs
        └── smacv2_run.log                ← Training log
```

CSV metrics also written to `logs_thesis/` (if configured).

## Collection & Analysis

### Step 1: Collect All Results
```bash
# On each completed machine:
cd ~/projects/inzynierka_kod_zrodlowy
./runs/smacv2_21/collect_results.sh "protoss_5_vs_5_qmix_seed100" /tmp
# Creates: /tmp/protoss_5_vs_5_qmix_seed100.zip

# Or batch collect on a central machine:
./runs/smacv2_21/collect_all_21_results.sh /tmp/results
# Creates: /tmp/results/*.zip (all 21)
```

### Step 2: Download ZIPs Locally
```bash
# From your laptop
mkdir ~/thesis_results
for i in {1..21}; do
  host="user@host-$i"
  scp $host:~/tmp/*.zip ~/thesis_results/
done
```

### Step 3: Aggregate & Analyze
```bash
cd ~/thesis_results
unzip -q '*.zip'
python3 runs/smacv2_21/aggregate_thesis_results.py . summary.csv

# Open summary.csv in Excel or Python
# Columns:
# - task, algorithm, seed, variant
# - final_return_mean, final_return_std, final_win_rate
# - max_return, total_frames, checkpoints
```

### Step 4: Create Plots/Tables for Thesis
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('summary.csv')

# Plot 1: Mean return by algorithm
df.groupby('algorithm')['final_return_mean'].mean().plot(kind='bar')
plt.ylabel('Final Return')
plt.title('Algorithm Comparison on protoss_5_vs_5')
plt.savefig('comparison.pdf')

# Plot 2: Win rate comparison
for algo in df['algorithm'].unique():
    subset = df[df['algorithm'] == algo]
    win_rate = subset['final_win_rate'].mean()
    print(f"{algo}: {win_rate:.1%}")
```

## Troubleshooting

### GPU OOM
```bash
OFF_POLICY_TRAIN_BATCH_SIZE=48 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh
```

### CPU-only fallback
```bash
REQUIRE_GPU=0 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh
```

### Strict GPU requirement
```bash
REQUIRE_GPU=1 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh
# Will fail if GPU unavailable or insufficient
```

### Check what auto-tune will use
```bash
AUTO_TUNE=1 ./runs/smacv2_21/06_protoss_5_vs_5_demir_seed102.sh 2>&1 | head -5
# Shows: "Auto-tune: n_envs=X, frames_per_batch=Y, train_batch=Z, memory_size=W"
```

## Summary

✅ **21 independent experiments ready to run**  
✅ **Each can run on any hardware (GPU/CPU auto-detection)**  
✅ **Batch sizes auto-tuned per machine**  
✅ **All metrics logged to CSV/JSON/W&B**  
✅ **Checkpoints saved for model inspection**  
✅ **Easy collection into ZIP files**  
✅ **Aggregation script for thesis analysis**  

**Total setup time:** ~10 min per machine  
**Total training time:** ~2-3 weeks per experiment (GPU)  
**Data collection time:** ~1 hour to collect all 21 ZIPs  
**Analysis time:** ~30 min to aggregate and plot  

---

**Project:** Engineering Thesis — DEMIR vs SOTA  
**Author:** Kajetan Frąckowiak  
**Institute:** PJATK  
**Date:** 2026-05-02
