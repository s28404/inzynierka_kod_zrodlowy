# Test Checklist — After Running test_quick_train.sh

## What to Verify

### 1. Training Output ✓
- [ ] Script detects CUDA device
- [ ] Auto-tuning output shows: `Auto-tune: n_envs=X, frames_per_batch=Y, train_batch=Z, memory_size=W`
- [ ] Config loads successfully
- [ ] Training loop starts without errors
- [ ] Takes ~1-2 minutes for 10K frames

### 2. Experiment Directory ✓
After training, check:
```bash
ls outputs/2026-05-02/*/qmix_smacv2_protoss_5_vs_5_seed999*/
```
Should contain:
- [ ] `config.pkl` — Full configuration
- [ ] `*.json` — Experiment metrics
- [ ] `wandb/` — W&B logs (if enabled)
- [ ] `checkpoints/` — Model snapshots (may be empty if `checkpoint_interval=0`)

### 3. Collection Script ✓
After running `collect_results.sh`:
```bash
ls -lh /tmp/qmix_smacv2_protoss_5_vs_5_seed999*.zip
```
Should show:
- [ ] ZIP file created
- [ ] Size > 1MB (should be 5-50MB with logs)

### 4. ZIP Contents ✓
Extract and verify:
```bash
unzip -l /tmp/qmix_smacv2_protoss_5_vs_5_seed999*.zip | head -30
```
Should contain:
- [ ] `outputs/` directory
- [ ] `outputs/*/config.pkl`
- [ ] `outputs/*/*.json` (metrics)
- [ ] `outputs/*/wandb/` directory
- [ ] `logs_thesis/` directory (if CSV logging enabled)
- [ ] `README.txt`
- [ ] `smacv2_run.log`

### 5. CSV Metrics ✓
Check for training data:
```bash
unzip -p /tmp/qmix_smacv2_protoss_5_vs_5_seed999*.zip | grep -i "\.csv"
```
Or extract and check:
```bash
ls logs_thesis/*.csv
cat logs_thesis/qmix_smacv2_protoss_5_vs_5_seed999*.csv | head -5
```
Should show:
- [ ] CSV file exists with headers: `frame,step,train_return_mean,eval_return_mean,...`
- [ ] At least 1-2 rows of data (from evaluations)

### 6. W&B Logs ✓
Inside the ZIP, check:
```bash
unzip -l .zip | grep wandb | head -10
```
Should show:
- [ ] `wandb/` directory present
- [ ] `run-*` subdirectories
- [ ] JSON metadata files
- [ ] Debug logs

### 7. Checkpoints ✓ (Optional)
```bash
unzip -l .zip | grep checkpoint
```
Should show:
- [ ] `.pt` files if `checkpoint_interval < max_n_frames`
- [ ] Empty if `checkpoint_interval=0` (default for quick test)

---

## Quick Visual Check

After extracting the ZIP locally, you should see:
```
qmix_smacv2_protoss_5_vs_5_seed999/
├── README.txt                       ← Guide
├── outputs/
│   ├── config.pkl                   ← Full config
│   ├── qmix_smacv2_protoss_5_vs_5_seed999.json    ← Metrics
│   ├── checkpoints/                 ← Models (may be empty)
│   └── wandb/                       ← W&B logs
└── logs_thesis/
    ├── *.csv                        ← Training metrics
    └── *.jsonl                      ← Line-delimited logs
```

---

## Expected Metrics in CSV

If CSV logging works, columns should include:
- `frame` — Total frames processed
- `step` — Training iteration
- `train_return_mean` — Mean training return
- `eval_return_mean` — Mean evaluation return
- `eval_return_std` — Std dev of eval return
- `eval_win_rate` — Win rate on SMAC (should be 0-100%)
- `total_frames` — Total frames at that point

Example row:
```
frame,step,train_return_mean,eval_return_mean,eval_return_std,eval_win_rate
5000,0,NaN,12.5,3.2,0.0
10000,1,15.0,18.3,2.8,20.0
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Script times out | Increase timeout or reduce `MAX_FRAMES` further |
| No experiment found | Check `outputs/` directory exists; verify seed is `999` |
| ZIP empty or corrupted | Check collect_results.sh has proper paths |
| No CSV files | CSV logging may not be enabled in config; check W&B logs instead |
| No W&B data | Offline mode or disabled; check config `loggers` field |
| Exit code != 0 | Check GPU memory, RAM, or dependencies |

---

## Next Steps

If all checks ✓:
1. You're ready to deploy to 21 machines
2. Run ONE script per computer: `./01_protoss_5_vs_5_qmix_seed100.sh`
3. After training, collect results: `./collect_results.sh <exp_name>`
4. Download all 21 ZIPs
5. Aggregate: `python3 aggregate_thesis_results.py . results.csv`
6. Analyze in Excel/Python/R

If any check ✗:
- Check logs in `/tmp/test_training.log`
- Review error messages from training
- Verify GPU: `nvidia-smi`
- Check RAM: `free -h`
