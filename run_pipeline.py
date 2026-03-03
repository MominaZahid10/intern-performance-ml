"""
Intern Performance Pipeline Runner
====================================
Runs the full pipeline: data generation → model training → dashboard data export.
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(description: str, script: str):
    """Run a Python script and stream output."""
    print(f"\n>>> {description}")
    print(f"{'-' * 60}")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script)],
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print(f"\n✗ FAILED: {script} exited with code {result.returncode}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  Intern Performance Prediction - Full Pipeline")
    print("=" * 60)

    run_step("Step 1/2 - Generating synthetic intern data", "generate_data.py")
    run_step("Step 2/2 - Training models & generating outputs", "train_model.py")

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"  Open dashboard.html in a browser to view results")
    print(f"  Outputs saved in: {os.path.join(BASE_DIR, 'outputs')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
