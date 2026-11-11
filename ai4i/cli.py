"""One command runs data checks, EDA, E00-E03 and verified outputs."""
import argparse
from datetime import datetime
from pathlib import Path
from .experiments import run_experiments
from .storage import LOCAL_TIME

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="AI4I: EDA, baseline and three experiment rounds.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "ai4i2020.csv")
    parser.add_argument("--output", type=Path, default=None, help="A new or empty result directory.")
    args = parser.parse_args(argv)
    output = args.output or PROJECT_ROOT / "results" / datetime.now(LOCAL_TIME).strftime("%Y-%m-%d_%H%M%S_%f")
    run_experiments(args.data, output)
