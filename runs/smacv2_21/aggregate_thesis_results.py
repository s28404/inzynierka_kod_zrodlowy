#!/usr/bin/env python3
"""
Aggregates metrics from all 21 SMACv2 experiments into a single CSV for analysis.

Usage:
    python3 aggregate_thesis_results.py <results_dir> [output_csv]

Example:
    python3 aggregate_thesis_results.py /tmp/smacv2_results results_summary.csv

This script:
1. Finds all ZIP files or extracted directories
2. Extracts experiment metadata from directory names
3. Collects final metrics (if available)
4. Creates a summary CSV for easy plotting and comparison
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import csv


@dataclass
class ExperimentMetadata:
    """Metadata extracted from experiment directory name"""
    task: str
    algorithm: str
    seed: int
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    variant: str = "baseline"  # or ablation type
    
    @classmethod
    def from_name(cls, exp_name: str) -> Optional["ExperimentMetadata"]:
        """
        Parse experiment name to extract metadata.
        Expected format: qmix_smacv2_protoss_5_vs_5_qmix_seed100_timestamp
        or: qmix_smacv2_protoss_5_vs_5_demir_scale0p05_encoder_idm_beta1_0p7_beta2_0p3_seed2_timestamp
        """
        # Extract task
        match = re.search(r'(protoss_\d+_vs_\d+|terran_\d+_vs_\d+|zerg_\d+_vs_\d+)', exp_name)
        task = match.group(1) if match else "unknown"
        
        # Extract algorithm
        algo_match = re.search(r'(qmix|demir|ngu|rnd)', exp_name.lower())
        algo = algo_match.group(1) if algo_match else "unknown"
        
        # Extract seed (look for seed<N> or _seed<N>)
        seed_match = re.search(r'seed(\d+)', exp_name)
        seed = int(seed_match.group(1)) if seed_match else -1
        
        # Extract DEMIR hyperparams if present
        beta1, beta2 = None, None
        if 'demir' in algo:
            b1_match = re.search(r'beta1_0p(\d+)', exp_name)
            b2_match = re.search(r'beta2_0p(\d+)', exp_name)
            if b1_match:
                beta1 = float('0.' + b1_match.group(1))
            if b2_match:
                beta2 = float('0.' + b2_match.group(1))
        
        # Detect ablation type
        variant = "baseline"
        if "ablation" in exp_name.lower() or "nobarlow" in exp_name.lower():
            variant = "no_barlow_twins"
        elif "mlp" in exp_name.lower():
            variant = "mlp_encoder"
        elif beta1 == 0.0:
            variant = "beta1_0"
        elif beta2 == 0.0:
            variant = "beta2_0"
        
        return cls(
            task=task,
            algorithm=algo,
            seed=seed,
            beta1=beta1,
            beta2=beta2,
            variant=variant
        )


@dataclass
class ExperimentResults:
    """Aggregated results from one experiment"""
    metadata: ExperimentMetadata
    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    checkpoint_count: int = 0
    
    # Final metrics (from JSON if available)
    final_return_mean: Optional[float] = None
    final_return_std: Optional[float] = None
    final_win_rate: Optional[float] = None
    max_return: Optional[float] = None
    total_frames: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export"""
        return {
            "task": self.metadata.task,
            "algorithm": self.metadata.algorithm,
            "seed": self.metadata.seed,
            "variant": self.metadata.variant,
            "beta1": self.metadata.beta1,
            "beta2": self.metadata.beta2,
            "checkpoints": self.checkpoint_count,
            "final_return_mean": self.final_return_mean,
            "final_return_std": self.final_return_std,
            "final_win_rate": self.final_win_rate,
            "max_return": self.max_return,
            "total_frames": self.total_frames,
        }


def find_experiments(results_dir: Path) -> List[Path]:
    """Find all experiment directories"""
    experiments = []
    
    # Look for extracted directories or ZIP contents
    for item in results_dir.rglob("*.json"):
        # Typical path: results_dir/EXPERIMENT_NAME/outputs/EXPERIMENT_NAME.json
        if "outputs" in str(item.parent):
            experiments.append(item.parent.parent)
    
    # Also look for direct directories
    for item in results_dir.iterdir():
        if item.is_dir() and "qmix_" in item.name or "demir_" in item.name:
            experiments.append(item)
    
    return list(set(experiments))


def extract_metrics_from_json(json_path: Path) -> Dict:
    """Extract final metrics from JSON file"""
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        metrics = {}
        
        # Try to extract final evaluation metrics
        if "metrics" in data and len(data["metrics"]) > 0:
            last_metric = data["metrics"][-1]
            metrics["final_return_mean"] = last_metric.get("eval_return_mean")
            metrics["final_return_std"] = last_metric.get("eval_return_std")
            metrics["final_win_rate"] = last_metric.get("eval_win_rate")
            metrics["total_frames"] = last_metric.get("frame")
            
            # Find max return
            returns = [m.get("eval_return_mean") for m in data["metrics"] 
                      if m.get("eval_return_mean") is not None]
            if returns:
                metrics["max_return"] = max(returns)
        
        return metrics
    except Exception as e:
        print(f"Warning: Could not extract metrics from {json_path}: {e}")
        return {}


def process_experiment(exp_dir: Path) -> Optional[ExperimentResults]:
    """Process one experiment directory and extract metadata/metrics"""
    exp_name = exp_dir.name
    
    # Parse metadata from name
    meta = ExperimentMetadata.from_name(exp_name)
    if meta is None:
        return None
    
    results = ExperimentResults(metadata=meta)
    
    # Find JSON file
    json_file = exp_dir / "outputs" / f"{exp_name}.json"
    if json_file.exists():
        results.json_path = json_file
        metrics = extract_metrics_from_json(json_file)
        results.final_return_mean = metrics.get("final_return_mean")
        results.final_return_std = metrics.get("final_return_std")
        results.final_win_rate = metrics.get("final_win_rate")
        results.max_return = metrics.get("max_return")
        results.total_frames = metrics.get("total_frames")
    
    # Count checkpoints
    checkpoint_dir = exp_dir / "outputs" / "checkpoints"
    if checkpoint_dir.exists():
        results.checkpoint_count = len(list(checkpoint_dir.glob("*.pt")))
    
    # Find CSV if present
    csv_dir = exp_dir / "logs_thesis"
    if csv_dir.exists():
        csvs = list(csv_dir.glob("*.csv"))
        if csvs:
            results.csv_path = csvs[0]
    
    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("thesis_results_summary.csv")
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(2)
    
    print(f"Scanning for experiments in: {results_dir}")
    exp_dirs = find_experiments(results_dir)
    
    if not exp_dirs:
        print("No experiments found. Looking for directories manually...")
        exp_dirs = [d for d in results_dir.rglob("*") if d.is_dir() and 
                   ("qmix_" in d.name or "demir_" in d.name) and "outputs" in str(d)]
    
    print(f"Found {len(exp_dirs)} experiments")
    
    results_list = []
    for exp_dir in exp_dirs:
        result = process_experiment(exp_dir)
        if result:
            results_list.append(result)
            print(f"  ✓ {exp_dir.name}")
        else:
            print(f"  ✗ Could not process {exp_dir.name}")
    
    if not results_list:
        print("ERROR: No experiments processed successfully")
        sys.exit(3)
    
    # Sort by task, algorithm, seed for better readability
    results_list.sort(
        key=lambda r: (r.metadata.task, r.metadata.algorithm, r.metadata.seed)
    )
    
    # Write CSV
    fieldnames = [
        "task", "algorithm", "seed", "variant", "beta1", "beta2",
        "checkpoints", "final_return_mean", "final_return_std",
        "final_win_rate", "max_return", "total_frames"
    ]
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_list:
            writer.writerow(result.to_dict())
    
    print(f"\n✓ Summary CSV written to: {output_csv}")
    print(f"  Rows: {len(results_list)}")
    print(f"  Size: {output_csv.stat().st_size / 1024:.1f} KB")
    print("\nSummary by algorithm:")
    
    # Print summary by algorithm
    by_algo = {}
    for result in results_list:
        algo = result.metadata.algorithm
        if algo not in by_algo:
            by_algo[algo] = []
        by_algo[algo].append(result)
    
    for algo, results in sorted(by_algo.items()):
        returns = [r.final_return_mean for r in results if r.final_return_mean is not None]
        if returns:
            avg_return = sum(returns) / len(returns)
            print(f"  {algo}: {len(results)} exps, avg return = {avg_return:.2f}")


if __name__ == "__main__":
    main()
