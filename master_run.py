# master_run.py
"""
Master execution script for macroeconomic forecasting project.

This script assumes that:
- The fully cleaned and model-ready dataset is already present.
- Data validation outputs have already been generated and are included in the repository.
- Replication begins at PCA factor construction.

Execution order:
04a_pca_factors.py
04b_baselines_var_linear.py
04c_xgb.py
04d_rnn.py
04e_merge_and_report.py
04f_plot_forecasts.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "04a_pca_factors.py",
    "04b_baselines_var_linear.py",
    "04c_xgb.py",
    "04d_rnn.py",
    "04e_merge_and_report.py",
    "04f_plot_forecasts.py",
]

def run_script(script_name):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"\nERROR in {script_name}")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)

def main():
    root = Path.cwd()

    for script in SCRIPTS:
        script_path = root / script
        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script}")

        run_script(script)

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
