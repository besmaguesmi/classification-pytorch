# src/run_experiments.py

# This script AUTOMATES running multiple experiments
# Instead of manually running 8 commands, run 1!
import os
import subprocess
import yaml
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def run_experiment(experiment_name, params_overrides=None):
    """
    Run a DVC experiment with specific parameters

    Args:
        experiment_name: Name of the experiment
        params_overrides: Dictionary of parameter overrides
    """
    # Create experiments directory
    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)

    logging.info(f"Running experiment: {experiment_name}")

    # Build command with parameter overrides
    cmd = ["dvc", "exp", "run", "--name", experiment_name]

    if params_overrides:
        for key, value in params_overrides.items():
            cmd.extend(["-S", f"{key}={value}"])

    logging.info(f"Command: {' '.join(cmd)}")

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logging.info(f"Experiment {experiment_name} completed successfully!")

        # Get experiment results
        subprocess.run(["dvc", "exp", "show", "--no-pager"])

        # Save experiment results
        exp_results = {
            "experiment_name": experiment_name,
            "params_overrides": params_overrides,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        with open(f"{exp_dir}/results.json", 'w') as f:
            json.dump(exp_results, f, indent=2)

        return True
    else:
        logging.error(f"Experiment {experiment_name} failed:")
        logging.error(result.stderr)

        # Save error results
        exp_results = {
            "experiment_name": experiment_name,
            "params_overrides": params_overrides,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": result.stderr
        }

        with open(f"{exp_dir}/results.json", 'w') as f:
            json.dump(exp_results, f, indent=2)

        return False

def compare_experiments(experiment_names):
    """Compare multiple experiments"""
    if len(experiment_names) < 2:
        logging.error("Need at least 2 experiments to compare")
        return

    cmd = ["dvc", "exp", "diff"] + experiment_names
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logging.info(f"Comparison of {experiment_names}:")
        print(result.stdout)
    else:
        logging.error(f"Failed to compare experiments: {result.stderr}")

if __name__ == "__main__":
    # Define experiments
    experiments = {
        "baseline": {},
        "resnet50": {"model.backbone": "resnet50"},
        "resnet101": {"model.backbone": "resnet101"},
        "unfrozen": {"train.freeze_backbone": False},
        "larger_batch": {"train.batch_size": 32},
        "more_epochs": {"train.max_epochs": 10},
        "higher_lr": {"train.learning_rate": 0.0001},
        "custom_cnn": {"model.backbone": "none"},
    }

    # Run all experiments
    successful_experiments = []
    for name, overrides in experiments.items():
        if run_experiment(name, overrides):
            successful_experiments.append(name)

    logging.info("\n" + "=" * 60)
    logging.info("EXPERIMENTS SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total experiments: {len(experiments)}")
    logging.info(f"Successful: {len(successful_experiments)}")
    logging.info(f"Failed: {len(experiments) - len(successful_experiments)}")

    if successful_experiments:
        logging.info("\nTo compare experiments:")
        logging.info("  dvc exp show")
        logging.info("\nTo apply the best experiment:")
        logging.info("  dvc exp apply <experiment-name>")

        # Compare successful experiments
        if len(successful_experiments) >= 2:
            compare_experiments(successful_experiments[:2])
