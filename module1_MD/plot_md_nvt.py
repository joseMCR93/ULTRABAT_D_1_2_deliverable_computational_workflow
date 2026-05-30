"""
plot_md_nvt.py
--------------
Reads all summary_*.txt files from an NVT molecular dynamics run and
generates two PNG plots:
  1. Temperature (K) vs Step
  2. Potential Energy (eV) vs Step

Usage:
    python plot_md_nvt.py                  # searches in the current directory
    python plot_md_nvt.py /path/to/data    # searches in the given path
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
DPI      = 150
FIG_SIZE = (12, 5)

TEMP_COLOR = "#E05C5C"   # red for temperature
EPOT_COLOR = "#4C8BDB"   # blue for potential energy
MEAN_COLOR = "#2CA02C"   # green for rolling mean

WINDOW = 200   # rolling mean window in steps; increase for smoother curves
# ──────────────────────────────────────────────────────────────────────────────


def extract_start_step(filename: str) -> int:
    """
    Extracts the starting step from the filename to sort files correctly.
    Examples:
        summary_NVT_mace_PB_300K_0.txt              -> 0
        summary_NVT_mace_PB_300K_1-100000.txt       -> 1
        summary_NVT_mace_PB_300K_900001-1000000.txt -> 900001
    """
    base = os.path.basename(filename)
    match = re.search(r"_(\d+)(?:-\d+)?\.txt$", base)
    return int(match.group(1)) if match else 0


def read_md_files(folder: str) -> pd.DataFrame:
    """Reads and concatenates all summary*.txt files sorted by starting step."""
    pattern = os.path.join(folder, "summary*.txt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No 'summary*.txt' files found in: {folder!r}"
        )

    # Sort by starting step, not alphabetically
    files = sorted(files, key=extract_start_step)

    print(f"  -> {len(files)} file(s) found (sorted by starting step):")
    frames = []
    for f in files:
        print(f"     {os.path.basename(f)}")
        try:
            df = pd.read_csv(f, sep=r"\s+", comment="#")
            df.columns = [c.strip() for c in df.columns]
            frames.append(df)
        except Exception as e:
            print(f"     WARNING: could not read {os.path.basename(f)}: {e}")

    if not frames:
        raise ValueError("No files could be read successfully.")

    combined = pd.concat(frames, ignore_index=True)

    # Flexible column detection
    col_map = {}
    for col in combined.columns:
        cl = col.lower()
        if "step" in cl and "step" not in col_map:
            col_map["step"] = col
        if "potential" in cl and "energy" in cl:
            col_map["epot"] = col
        if "temperature" in cl:
            col_map["temp"] = col

    missing = [k for k in ("step", "epot", "temp") if k not in col_map]
    if missing:
        raise KeyError(
            f"Could not find columns: {missing}\n"
            f"Available columns: {list(combined.columns)}"
        )

    combined = combined.rename(columns={
        col_map["step"]: "Step",
        col_map["epot"]: "Epot",
        col_map["temp"]: "Temp",
    })

    combined = (combined[["Step", "Epot", "Temp"]]
                .dropna()
                .sort_values("Step")
                .reset_index(drop=True))
    return combined


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _add_grid(ax):
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", ls="--", alpha=0.4)
    ax.grid(True, which="minor", ls=":",  alpha=0.2)


def plot_temperature(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(df["Step"], df["Temp"],
            color=TEMP_COLOR, lw=0.6, alpha=0.45, label="Instantaneous temperature")

    rm = rolling_mean(df["Temp"], WINDOW)
    ax.plot(df["Step"], rm,
            color=MEAN_COLOR, lw=2.0, label=f"Rolling mean (window = {WINDOW} steps)")

    mean_val = df["Temp"].mean()
    ax.axhline(mean_val, color="gray", lw=1.2, ls="--",
               label=f"Global mean = {mean_val:.1f} K")

    ax.set_xlabel("MD Step", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.set_title("NVT -- Temperature vs Step", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    _add_grid(ax)

    stats = (f"min = {df['Temp'].min():.1f} K\n"
             f"max = {df['Temp'].max():.1f} K\n"
             f"std = {df['Temp'].std():.1f} K")
    ax.text(0.99, 0.97, stats, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7))

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_potential_energy(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(df["Step"], df["Epot"],
            color=EPOT_COLOR, lw=0.6, alpha=0.45, label="Potential energy")

    rm = rolling_mean(df["Epot"], WINDOW)
    ax.plot(df["Step"], rm,
            color=MEAN_COLOR, lw=2.0, label=f"Rolling mean (window = {WINDOW} steps)")

    mean_val = df["Epot"].mean()
    ax.axhline(mean_val, color="gray", lw=1.2, ls="--",
               label=f"Global mean = {mean_val:.4f} eV")

    ax.set_xlabel("MD Step", fontsize=12)
    ax.set_ylabel("Potential Energy (eV)", fontsize=12)
    ax.set_title("NVT -- Potential Energy vs Step", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    _add_grid(ax)

    stats = (f"min = {df['Epot'].min():.4f} eV\n"
             f"max = {df['Epot'].max():.4f} eV\n"
             f"std = {df['Epot'].std():.4f} eV")
    ax.text(0.99, 0.97, stats, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7))

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    folder = os.path.abspath(folder)
    print(f"\nSearching for files in: {folder}")

    df = read_md_files(folder)

    print(f"\nData loaded: {len(df):,} total steps")
    print(f"   Step : {int(df['Step'].min()):,} -> {int(df['Step'].max()):,}")
    print(f"   Temp : {df['Temp'].mean():.2f} +/- {df['Temp'].std():.2f} K")
    print(f"   Epot : {df['Epot'].mean():.4f} +/- {df['Epot'].std():.4f} eV")

    out_temp = os.path.join(folder, "nvt_temperature.png")
    out_epot = os.path.join(folder, "nvt_potential_energy.png")

    print("\nGenerating plots...")
    plot_temperature(df, out_temp)
    plot_potential_energy(df, out_epot)

    print(f"\nDone. PNG files saved to: {folder}\n")


if __name__ == "__main__":
    main()
