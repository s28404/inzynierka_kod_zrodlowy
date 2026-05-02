# Quick Test Guide — How to Run

## Two Options

### Option 1: Simple One-Liner (Recommended for Fast Testing)
```bash
cd ~/projects/inzynierka_kod_zrodlowy
bash runs/smacv2_21/test_oneliner.sh
```

**Time:** ~2-3 minutes  
**Output:** Prints training output, collects ZIP, shows contents  
**Best for:** Quick verification of end-to-end pipeline

---

### Option 2: Full Test Script (Recommended for Detailed Verification)
```bash
cd ~/projects/inzynierka_kod_zrodlowy
./runs/smacv2_21/test_quick_train.sh
```

**Time:** ~3-5 minutes  
**Output:** Detailed step-by-step output + verification checks  
**Best for:** Thorough validation before deploying to 21 machines

---

## What These Tests Do

### test_quick_train.sh
1. ✅ Runs 10K frame training on `protoss_5_vs_5` with QMIX
2. ✅ Uses `seed=999` to mark as test run
3. ✅ Collects results using `collect_results.sh`
4. ✅ Creates ZIP file in `/tmp/`
5. ✅ Verifies ZIP contents
6. ✅ Shows what files were created

### test_oneliner.sh
- Same as above but in one bash command
- More compact output
- Easier to run in one line

---

## Expected Output

### GPU Detection
```
RUN: algo=qmix task=smacv2/protoss_5_vs_5 seed=999
Devices: sampling=cuda, train=cuda, buffer=cpu
Auto-tune: n_envs=8, frames_per_batch=1000, train_batch=128, memory_size=150000
```

### Training Progress
```
Algorithm: qmix, Task: smacv2/protoss_5_vs_5
Loaded config:
  max_n_frames: 10000
  off_policy_n_envs_per_worker: 8
  off_policy_train_batch_size: 128
  ...
```

### Collection Result
```
✓ Experiment directory: qmix_smacv2_protoss_5_vs_5_seed999_...
✓ Outputs directory: EXISTS
✓ Logs (CSV/JSONL): EXISTS (or NOT FOUND if logging disabled)
✓ README: EXISTS

ZIP file ready for download:
  /tmp/qmix_smacv2_protoss_5_vs_5_seed999_....zip
```

---

## After Running Tests

### 1. Download the ZIP
```bash
# From your machine
scp user@host:/tmp/qmix_smacv2_protoss_5_vs_5_seed999*.zip ~/Downloads/
```

### 2. Extract and Explore
```bash
unzip ~/Downloads/qmix_smacv2_protoss_5_vs_5_seed999*.zip
cd qmix_smacv2_protoss_5_vs_5_seed999/

# Look at structure
ls -lh outputs/
ls -lh logs_thesis/
cat README.txt
```

### 3. Verify Files Exist
Using the [TEST_CHECKLIST.md](TEST_CHECKLIST.md):
- [ ] `config.pkl` — training configuration
- [ ] `*.json` — experiment metrics
- [ ] `wandb/` — W&B logs
- [ ] `logs_thesis/*.csv` — training metrics (if enabled)
- [ ] `README.txt` — quick reference

### 4. Check Metrics
```python
# Python
import json
with open('outputs/qmix_smacv2_protoss_5_vs_5_seed999.json') as f:
    data = json.load(f)
    print(f"Metadata: {data.get('metadata')}")
    print(f"Metrics rows: {len(data.get('metrics', []))}")
    if data['metrics']:
        print(f"Last metric: {data['metrics'][-1]}")
```

---

## Troubleshooting

### Script Hangs
- May take a few seconds for environment to load
- Use Ctrl+C to stop and check errors
- Check GPU: `nvidia-smi`

### "No experiment found"
- Training may have failed
- Check `/tmp/test_training.log` if using Option 2
- Review error messages in script output

### ZIP File Missing
- Collection script may have failed
- Try running `collect_results.sh` manually:
  ```bash
  ./runs/smacv2_21/collect_results.sh "qmix_smacv2_protoss_5_vs_5_seed999_*"
  ```

### No CSV Files in ZIP
- CSV logging may not be configured
- Check W&B logs (`wandb/`) instead
- JSON metrics (`*.json`) will always be present

---

## Quick Validation

After extraction, verify everything works with:
```bash
# Count files in ZIP
unzip -l /tmp/qmix_smacv2_protoss_5_vs_5_seed999*.zip | wc -l

# Check JSON is valid
python3 -m json.tool outputs/qmix_smacv2_protoss_5_vs_5_seed999.json > /dev/null && echo "✓ JSON valid"

# Check CSV headers (if exists)
head -1 logs_thesis/*.csv 2>/dev/null || echo "⚠ No CSV files"

# List checkpoint files
unzip -l /tmp/qmix_smacv2_protoss_5_vs_5_seed999*.zip | grep "\.pt$"
```

---

## Once Tests Pass ✅

You're ready to:
1. Distribute one script per computer (01_*.sh through 21_*.sh)
2. Run: `./01_protoss_5_vs_5_qmix_seed100.sh` on each machine
3. After training: `./collect_results.sh "exp_name"` on each machine
4. Download all 21 ZIPs
5. Aggregate: `python3 aggregate_thesis_results.py . results.csv`
6. Analyze results!

---

## See Also

- [README.md](README.md) — Full user guide
- [SETUP_SUMMARY.md](SETUP_SUMMARY.md) — Detailed technical reference
- [TEST_CHECKLIST.md](TEST_CHECKLIST.md) — Detailed checklist of what to verify
