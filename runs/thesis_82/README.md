# Thesis Experiment Scripts (82 total)

This folder contains per-experiment bash launchers for distributed execution.

Counts generated from the provided matrix:
- SMACv2 main: 27
- SMACv2 ablations: 12
- SynchronizedFactory main: 4
- MiniGrid main: 27
- MiniGrid ablations: 12
- Total: 82

Note: the request text mentioned "~85", but the explicit combinations sum to 82.

## Usage
Run one script per machine, e.g.:

```bash
bash 001_smac_p5v5_qmix_s100.sh
```

Each script is independent.

## Core defaults
- DEMIR classic: beta1=0.7, beta2=0.3, encoder=idm
- SMAC/Logic no-barlow ablation: `algorithm.barlow_scale=0.0`
- MiniGrid no-barlow ablation: `--demir-barlow-scale 0.0`
