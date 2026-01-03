"""04f_plot_forecasts.py

Purpose
- Produce publication-ready plots that compare realized values to model forecasts,
  and create “all-model” comparison plots for each target and horizon.

What the script does
1) Loads the merged forecast and metric files produced by the reporting step.
2) Determines a “best model” (winner) for each (target, horizon) using a clear,
   quantitative criterion (typically lowest RMSE, subject to basic validity checks).
3) Creates two families of figures:
   - Winner vs realized: a clean plot showing the selected best forecast against
     the realized series.
   - All models vs realized: an overlay plot that shows the realized series and
     every model’s forecast to facilitate qualitative comparison.
4) Saves figures to the project’s figures folder with consistent filenames so the
   LaTeX report can include them programmatically.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Plot styling / helpers
# -----------------------------
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

def apply_plot_style():
    # Set a consistent plotting theme: clean background, readable fonts, and unobtrusive gridlines.
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
    })


def style_ax(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", labelsize=10)


def set_main_and_subtitle(fig: plt.Figure, ax: plt.Axes, main_title: str, subtitle: str) -> None:
    """Apply the paper's title hierarchy.

    - Main title: bold, smaller than before, and close to the subtitle.
    - Subtitle: fixed text ("Pseudo out-of-sample forecast") with compact spacing.
    """
    fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.965)
    ax.set_title(subtitle, fontsize=12, pad=6)


def clip_ylim(y: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float]:
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if lo == hi:
        return (lo - 1.0, hi + 1.0)
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


# -----------------------------
# Constants / labels
# -----------------------------
MODEL_ORDER = [
    "VAR",
    "Ridge",
    "ElasticNet",
    "XGBoost",
    "LSTM_SGD",
    "GRU_SGD",
    "LSTM_Adam",
    "GRU_Adam",
]

MODEL_PRETTY = {
    "VAR": "VAR",
    "Ridge": "Ridge",
    "ElasticNet": "ElasticNet",
    "XGBoost": "XGBoost",
    "LSTM_SGD": "LSTM_SGD",
    "GRU_SGD": "GRU_SGD",
    "LSTM_Adam": "LSTM_Adam",
    "GRU_Adam": "GRU_Adam",
}

TARGET_LABELS = {
    "cpi_headline_yoy": "CPI inflation (headline, YoY)",
    "cpi_core_yoy": "CPI inflation (core, YoY)",
    "indpro_total_yoy": "Industrial production (YoY)",
    "unrate_u3": "Unemployment rate (U3)",
    "t10y2y": "Yield curve slope (10Y–2Y)",
}

# Edit these if you want slightly different wording.
TARGET_YLABELS = {
    "cpi_headline_yoy": "Inflation rate (%)",
    "cpi_core_yoy": "Inflation rate (%)",
    "indpro_total_yoy": "Industrial production growth (%)",
    "unrate_u3": "Unemployment rate (%)",
    "t10y2y": "Yield curve slope (pp)",
}

FOOTNOTE = "Source: FRED, LSEG Refinitiv"


# -----------------------------
# IO helpers
# -----------------------------
def project_root_from_script() -> Path:
    here = Path(__file__).resolve().parent
    for c in [here] + list(here.parents):
        if (c / "Data").exists():
            return c
    return Path.cwd()


def out_dir(root: Path) -> Path:
    return root / "Data" / "Processed" / "Final"


def available_horizons(outp: Path) -> list[int]:
    hs = []
    for h in [1, 12]:
        if (outp / f"final_forecasts_h{h}.csv").exists() and (outp / f"final_metrics_h{h}.csv").exists():
            hs.append(h)
    return hs


# -----------------------------
# Winner selection
# -----------------------------
def choose_winners(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the best-performing model for a given (target, horizon).

    Selection rule
    - Primary criterion: minimize RMSE.
    - Validity constraints: exclude models with missing / non-finite RMSE and exclude
      models evaluated on substantially fewer observations than the maximum available
      for that target/horizon (to avoid “winning” on an easier subset).

    Returns
    - DataFrame with columns: target, horizon_months, winner_model
    """
    m = metrics.copy()

    for c in ["rmse", "mae", "n_obs", "horizon_months"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "horizon_months", "model", "rmse", "n_obs"])
    if m.empty:
        return pd.DataFrame(columns=["target", "horizon_months", "winner_model"])

    # Keep only models with near-maximum number of scored observations (within 10%)
    g = m.groupby(["target", "horizon_months"])["n_obs"].transform("max")
    m = m[m["n_obs"] >= 0.90 * g]

    # Winner = lowest RMSE
    m = m.sort_values(["target", "horizon_months", "rmse"], ascending=[True, True, True])
    winners = (
        m.groupby(["target", "horizon_months"], as_index=False)
         .first()[["target", "horizon_months", "model"]]
         .rename(columns={"model": "winner_model"})
    )
    return winners


# -----------------------------
# Plotting
# -----------------------------
def plot_target_horizon(
    outp: Path,
    fc: pd.DataFrame,
    winners: pd.DataFrame,
    target: str,
    h: int,
) -> None:
    sub = fc[(fc["target"] == target) & (fc["horizon_months"] == h)].copy()
    if sub.empty:
        return

    sub["origin_date"] = pd.to_datetime(sub["origin_date"])
    sub = sub.sort_values("origin_date")

    id_cols = {"origin_date", "target", "horizon_months", "y_true"}
    model_cols = [m for m in MODEL_ORDER if m in sub.columns and m not in id_cols]
    if not model_cols:
        return

    pdir = outp / "plots" / f"h{h}"
    pdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # All models vs realized
    # -------------------------
    fig, ax = plt.subplots(figsize=(16, 6))
    style_ax(ax)

    ax.plot(sub["origin_date"], sub["y_true"], color="black", linewidth=3.0, label="Realized")
    for m in model_cols:
        ax.plot(sub["origin_date"], sub[m], linewidth=1.2, alpha=0.9, label=MODEL_PRETTY.get(m, m))

    main_title = f"{TARGET_LABELS.get(target, target)} — {h} months ahead"
    set_main_and_subtitle(fig, ax, main_title, "Pseudo out-of-sample forecast")

    ax.set_xlabel("")  # remove x-axis label
    ax.set_ylabel(TARGET_YLABELS.get(target, "Value"))

    y_all = np.concatenate([sub["y_true"].to_numpy(float)] + [sub[m].to_numpy(float) for m in model_cols])
    ax.set_ylim(*clip_ylim(y_all))

    # Lift legend upward so it doesn't sit on the plot area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False)

    fig.text(0.01, 0.01, FOOTNOTE, ha="left", va="bottom", fontsize=9, alpha=0.85)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(pdir / f"all_models_{target}.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Winner vs realized
    # -------------------------
    wrow = winners[(winners["target"] == target) & (winners["horizon_months"] == h)]
    if wrow.empty:
        return
    winner = wrow["winner_model"].iloc[0]
    if winner not in sub.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    style_ax(ax)

    ax.plot(sub["origin_date"], sub["y_true"], color="black", linewidth=3.0, label="Realized")
    ax.plot(sub["origin_date"], sub[winner], color="#4F81BD", linewidth=2.2, label=MODEL_PRETTY.get(winner, winner))

    main_title = f"{TARGET_LABELS.get(target, target)} — {h} months ahead"
    set_main_and_subtitle(fig, ax, main_title, "Pseudo out-of-sample forecast")

    ax.set_xlabel("")  # remove x-axis label
    ax.set_ylabel(TARGET_YLABELS.get(target, "Value"))

    y_w = np.concatenate([sub["y_true"].to_numpy(float), sub[winner].to_numpy(float)])
    ax.set_ylim(*clip_ylim(y_w))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=10)

    fig.text(0.01, 0.01, FOOTNOTE, ha="left", va="bottom", fontsize=9, alpha=0.85)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(pdir / f"winner_{target}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_plot_style()
    root = project_root_from_script()
    outp = out_dir(root)

    hs = available_horizons(outp)
    if not hs:
        raise FileNotFoundError(f"No final forecast/metric files found in {outp}")

    for h in hs:
        fc_path = outp / f"final_forecasts_h{h}.csv"
        met_path = outp / f"final_metrics_h{h}.csv"

        fc = pd.read_csv(fc_path)
        met = pd.read_csv(met_path)

        winners = choose_winners(met)
        targets = sorted(set(fc["target"].astype(str)))

        for tgt in targets:
            plot_target_horizon(outp, fc, winners, tgt, h)

        print(f"Saved plots for h={h} to {outp / 'plots' / f'h{h}'}")


if __name__ == "__main__":
    main()
