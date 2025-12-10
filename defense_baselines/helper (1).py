import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Process input parameters for agent training")

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster",              
        action="store_true", 
        help="Run under SLURM (submit via sbatch)."
    )
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna tuning and use fixed hyperparams")

    # Optional fixed hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (e.g., 5e-5)"
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=None,
        help="PPO clipping ratio (e.g., 0.2)"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Total training batch size (e.g., 150000)"
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Minibatch size (e.g., 4096)"
    )

    return parser.parse_args()
