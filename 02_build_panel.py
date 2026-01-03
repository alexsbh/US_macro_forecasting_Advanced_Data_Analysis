"""02_build_panel.py

Purpose
- Turn the collection of raw, per-series CSV files into a single, clean monthly panel
  that can be used directly for feature engineering and forecasting.

What the script does
1) Loads the raw monthly series produced by the scraping step.
2) Applies a common sample window so all series align on the same monthly timeline.
3) Handles frequency mismatches:
   - some series are effectively quarterly (values only observed in certain months);
     these are converted to monthly using time-based linear interpolation and
     bounded linear extrapolation at the sample edges.
4) Produces a single wide panel (Date as the index; one column per variable) and
   writes it to the processed-data directory.

Inputs
- Raw series files created by the data collection script.
- A list of series that require quarterly-to-monthly conversion (defined below).

Outputs
- A processed monthly panel saved under the project’s processed-data folder.
- (Optional) small audit information for the interpolation/extrapolation step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# =========================================================
# Project paths helper (portable, no hardcoded Desktop paths)
# =========================================================
from pathlib import Path

def get_project_root(start: Path | None = None) -> Path:
    """Find the project (repo) root regardless of the current working directory.

    Strategy: walk upward from `start` (defaults to this script) until we find a
    directory containing a `Data/` folder, or a repo marker like `.git`.
    """
    if start is None:
        start = Path(__file__).resolve()
    start = start.resolve()
    if start.is_file():
        start = start.parent

    markers = {".git", "pyproject.toml", "requirements.txt", "environment.yml"}
    for p in [start] + list(start.parents):
        if (p / "Data").is_dir():
            return p
        if any((p / m).exists() for m in markers):
            return p
    return start

def ensure_data_tree(root: Path) -> dict[str, Path]:
    """Create the standard Data/ subfolders (idempotent)."""
    data = root / "Data"
    raw = data / "Raw data"
    processed = data / "Processed data"
    validation = data / "Data validation"
    oos = data / "Pseudo OOS Forecasting Results"
    for p in (data, raw, processed, validation, oos):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "data": data, "raw": raw, "processed": processed, "validation": validation, "oos": oos}

PROJECT_ROOT = get_project_root()
PATHS = ensure_data_tree(PROJECT_ROOT)
DATA_DIR = PATHS["data"]

QUARTERLY_VARS: List[str] = [
    "gdp_deflator",
    "gdp_nominal",
    "gdp_real",
    "exports_real",
    "imports_real",
    "fhfa_hpi_all_transactions",
    "sloos_tightening_ci_large_mid",
    "delinq_consumer_loans",
    "delinq_credit_cards",
    "chargeoff_credit_cards",
    "m2_velocity",
    "m1_velocity",
    "commodity_index_proxy",
]

START_MONTH = "1960-01-01"
END_MONTH = "2025-08-01"


def _ensure_month_start(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    # Convert any date stamp to a standardized month-start timestamp (YYYY-MM-01).
    return d.dt.to_period("M").dt.to_timestamp(how="start")


def _linear_extrapolate_ends(s: pd.Series) -> pd.Series:
    """Linearly extrapolate missing values at the start/end of a series.

How it works
- If a series has leading NaNs, estimate them by extending the line implied by the
  first two non-missing observations.
- If a series has trailing NaNs, estimate them by extending the line implied by the
  last two non-missing observations.

Assumptions
- The series is indexed by an increasing DateTimeIndex.
- Only the edges are extrapolated; interior missing values are not handled here.
"""
    out = s.copy()
    out = out.astype(float)

    valid = out.dropna()
    if valid.shape[0] < 2:
        return out  # not enough points

    # Extrapolate missing values at the start of the series using the first two valid points.
    first_idx = valid.index[0]
    second_idx = valid.index[1]
    first_val = valid.iloc[0]
    second_val = valid.iloc[1]

    # Compute a time-aware slope (change per day) so extrapolation respects irregular spacing.
    dt_days = (second_idx - first_idx).days
    if dt_days != 0:
        slope_per_day = (second_val - first_val) / dt_days
        left_mask = out.index < first_idx
        if left_mask.any():
            days_from_first = (out.index[left_mask] - first_idx).days.astype(float)
            out.loc[left_mask] = first_val + slope_per_day * days_from_first

    # Extrapolate missing values at the end of the series using the last two valid points.
    last_idx = valid.index[-1]
    prev_idx = valid.index[-2]
    last_val = valid.iloc[-1]
    prev_val = valid.iloc[-2]

    dt_days = (last_idx - prev_idx).days
    if dt_days != 0:
        slope_per_day = (last_val - prev_val) / dt_days
        right_mask = out.index > last_idx
        if right_mask.any():
            days_from_last = (out.index[right_mask] - last_idx).days.astype(float)
            out.loc[right_mask] = last_val + slope_per_day * days_from_last

    return out


def quarterly_to_monthly_linear(df: pd.DataFrame, var: str) -> Tuple[pd.Series, Dict[str, Optional[float]]]:
    """Convert a sparsely observed (e.g., quarterly) series into a fully monthly series.

Steps
1) Use time-based linear interpolation to fill missing months between observed points.
2) Apply bounded linear extrapolation to fill missing months before the first observed
   point and after the last observed point (using the helper above).

Returns
- The monthly series aligned to the full monthly index.
- An audit dictionary that records how many values were interpolated/extrapolated.
"""
    if var not in df.columns:
        return pd.Series(index=df.index, dtype="float64", name=var), {
            "var": var, "status": "MISSING_COLUMN", "n_valid_before": None, "n_valid_after": None
        }

    s = pd.to_numeric(df[var], errors="coerce")
    s.name = var

    n_before = int(s.notna().sum())

    # Fill interior missing months via time-based linear interpolation.
    s_interp = s.copy()
    # The 'time' interpolation method requires a DateTimeIndex ordered in time.
    s_interp = s_interp.interpolate(method="time")

    # After interpolation, fill any remaining edge NaNs via linear extrapolation.
    s_filled = _linear_extrapolate_ends(s_interp)

    n_after = int(s_filled.notna().sum())

    return s_filled, {
        "var": var,
        "status": "OK",
        "n_valid_before": n_before,
        "n_valid_after": n_after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default=str(DATA_DIR / "Raw data" / "raw_complete_wide.csv"))
    ap.add_argument("--outdir", type=str, default=str(DATA_DIR / "Processed data"))
    ap.add_argument("--start", type=str, default=START_MONTH)
    ap.add_argument("--end", type=str, default=END_MONTH)
    args = ap.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise SystemExit(f"ERROR: input file not found: {infile.resolve()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load the raw panel created from individual series files.
    df = pd.read_csv(infile)
    if "date" not in df.columns:
        raise SystemExit("ERROR: input CSV must contain a 'date' column")

    # Standardize the date index (month-start) and set it as the DataFrame index.
    df["date"] = _ensure_month_start(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date")
    df.index = pd.DatetimeIndex(df.index)

    # Restrict the dataset to the project’s analysis sample window.
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()

    # Create a complete monthly index so frequency conversions are unambiguous.
    full_index = pd.date_range(start=start, end=end, freq="MS")  # month-start
    df = df.reindex(full_index)

    # Convert selected low-frequency (quarterly) series into monthly series.
    audit_rows = []
    for var in QUARTERLY_VARS:
        s_new, audit = quarterly_to_monthly_linear(df, var)
        audit_rows.append(audit)
        if audit["status"] == "OK":
            df[var] = s_new

    # Save the cleaned, aligned monthly panel to the processed-data folder.
    out_csv = outdir / "processed_panel.csv"
    df_out = df.reset_index().rename(columns={"index": "date"})
    df_out.to_csv(out_csv, index=False)

    audit_csv = outdir / "processing_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)

    print(f"Wrote: {out_csv.resolve()}")
    print(f"Wrote: {audit_csv.resolve()}")


if __name__ == "__main__":
    main()
