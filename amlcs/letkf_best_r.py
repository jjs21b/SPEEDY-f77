#!/usr/bin/env python3
"""
Report the best tuning configuration from an rmse_summary.csv.

Works for both tuning workflows:
  - LETKF localization sweeps  (letkf_r_tuning.py collect)  -> varies r
  - ReverseSDE inflation sweeps (reversesde_tuning.py collect) -> varies infla
  - 2D LETKF sweeps                                            -> varies r AND infla

The script auto-detects which parameter(s) actually vary in the summary and
ranks over those "cells". The best cell is the one that achieves the minimum
RMSE for the greatest number of variables (a scale-invariant "wins" count),
NOT the lowest overall average, because the variables live on very different
scales.

Prints:
  - the best cell by number of per-variable wins (ties broken by mean rank)
  - a per-variable table of which cell wins each variable
  - per-cell win counts and overall_avg (shown for reference only)

Usage
-----
    # LETKF r sweep
    python letkf_best_r.py letkf_tuning_runs/all_arctan/rmse_summary.csv

    # ReverseSDE inflation sweep
    python letkf_best_r.py reversesde_tuning_runs/arctan_w_inflation/rmse_summary.csv

    # Force which parameter(s) to group by (default: auto)
    python letkf_best_r.py <summary.csv> --by infla
    python letkf_best_r.py <summary.csv> --by both --metric min
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Candidate tuning-parameter columns, in display order.
PARAM_COLS = ["r", "infla"]


def _detect_vars(df):
    """Derive variable names from `<var>_avg` columns present in the summary."""
    vars_found = []
    for col in df.columns:
        if col.endswith("_avg") and col != "overall_avg":
            vars_found.append(col[: -len("_avg")])
    return vars_found


def _detect_keys(df, by):
    """
    Decide which parameter column(s) identify a tuning cell.

    by='auto': use whichever present parameters actually vary (>1 unique value);
               if none vary, fall back to all present parameters (single cell).
    by='r'/'infla': force that single column (must be present).
    by='both': use all present parameter columns.
    """
    present = [c for c in PARAM_COLS if c in df.columns]
    if not present:
        sys.exit("Summary has no tuning-parameter columns (expected 'r' and/or 'infla').")

    if by == "auto":
        varying = [c for c in present if df[c].nunique(dropna=False) > 1]
        return varying if varying else present
    if by == "both":
        return present
    # single forced column
    if by not in present:
        sys.exit(f"--by {by} requested but column '{by}' is not in the summary "
                 f"(present: {', '.join(present)}).")
    return [by]


def _fmt(x):
    return "n/a" if pd.isna(x) else f"{x:.6g}"


def _fmt_param(col, v):
    if pd.isna(v):
        return "n/a"
    if col == "r":
        try:
            return str(int(v))
        except (ValueError, TypeError):
            return str(v)
    return f"{float(v):.6g}"


def _cell_label(keys, cell):
    return " ".join(f"{k}={_fmt_param(k, v)}" for k, v in zip(keys, cell))


def _unique_cells(df, keys):
    """Unique cell keys (tuples), preserving order of appearance."""
    cells = []
    for _, row in df.iterrows():
        key = tuple(row[k] for k in keys)
        if key not in cells:
            cells.append(key)
    return cells


def rank_and_score(df, vars_found, metric, keys):
    """
    For each variable, find the winning cell (minimum RMSE) and per-cell ranks.

    Returns:
        wins:       dict cell -> number of variables won
        winner_of:  dict var  -> winning cell (or None)
        mean_rank:  dict cell -> mean rank across variables (lower is better)
        cells:      list of cells in display order
    """
    cells = _unique_cells(df, keys)
    wins = {c: 0 for c in cells}
    rank_sums = {c: 0.0 for c in cells}
    rank_counts = {c: 0 for c in cells}
    winner_of = {}

    for var in vars_found:
        col = f"{var}_{metric}"
        if col not in df.columns:
            winner_of[var] = None
            continue
        sub = df[list(keys) + [col]].dropna(subset=[col])
        if sub.empty:
            winner_of[var] = None
            continue

        # Rank ascending: lowest RMSE -> rank 1
        ranks = sub[col].rank(method="min", ascending=True)
        for (_, srow), rk in zip(sub.iterrows(), ranks):
            key = tuple(srow[k] for k in keys)
            rank_sums[key] += float(rk)
            rank_counts[key] += 1

        best_idx = sub[col].idxmin()
        win_cell = tuple(sub.loc[best_idx, k] for k in keys)
        winner_of[var] = win_cell
        wins[win_cell] += 1

    mean_rank = {c: (rank_sums[c] / rank_counts[c]) if rank_counts[c] else np.inf
                 for c in cells}
    return wins, winner_of, mean_rank, cells


def pick_best(wins, mean_rank):
    """Best cell: most wins; ties broken by lower mean rank, then cell values."""
    if not wins:
        return None
    return sorted(wins.keys(),
                  key=lambda c: (-wins[c], mean_rank.get(c, np.inf), c))[0]


def row_for_cell(df, keys, cell):
    """Return the row (as a dict) for a given cell, or empty dict if absent."""
    mask = pd.Series(True, index=df.index)
    for k, v in zip(keys, cell):
        mask &= (df[k] == v)
    sub = df[mask]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", help="Path to rmse_summary.csv from a tuning collect step.")
    parser.add_argument("--metric", choices=["avg", "min"], default="avg",
                        help="Per-variable metric used to decide each variable's winner: "
                             "'avg' (mean over cycles) or 'min' (lowest over cycles).")
    parser.add_argument("--by", choices=["auto", "r", "infla", "both"], default="auto",
                        help="Which tuning parameter(s) to group by "
                             "(default: auto-detect the varying parameter).")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        sys.exit(f"Summary CSV not found: {summary_path}")

    df = pd.read_csv(summary_path)
    keys = _detect_keys(df, args.by)

    df = df.sort_values(keys).reset_index(drop=True)
    vars_found = _detect_vars(df)
    if not vars_found:
        sys.exit("No per-variable columns (<var>_avg) found in the summary.")

    wins, winner_of, mean_rank, cells = rank_and_score(df, vars_found, args.metric, keys)
    best = pick_best(wins, mean_rank)
    if best is None:
        sys.exit("No rankable data (all per-variable values are NaN).")

    n_scored = sum(1 for v in winner_of.values() if v is not None)
    param_desc = " x ".join(keys)
    cell_desc = ", ".join(_cell_label(keys, c) for c in cells)

    print(f"Summary: {summary_path}")
    print(f"Variables: {', '.join(vars_found)}")
    print(f"Swept parameter(s): {param_desc}")
    print(f"Cells:  {cell_desc}")
    print(f"Ranking metric: per-variable '{args.metric}', best = most variable wins")
    print()
    print(f"==> BEST {_cell_label(keys, best)}  (wins {wins[best]}/{n_scored} variables, "
          f"mean rank {mean_rank[best]:.3g})")
    print()

    # Per-cell win counts
    print("Per-cell win counts (variables for which this cell has the lowest RMSE):")
    for cell in cells:
        marker = "  <-- best" if cell == best else ""
        overall = row_for_cell(df, keys, cell).get("overall_avg", np.nan)
        label = _cell_label(keys, cell)
        print(f"  {label:<20}: wins={wins[cell]:>2}/{n_scored}   "
              f"mean_rank={mean_rank[cell]:.3g}   overall_avg={_fmt(overall)} (ref){marker}")
    print()

    # Per-variable winners
    print(f"Per-variable winning cell (by '{args.metric}'):")
    suffix = f"_{args.metric}"
    for var in vars_found:
        win_cell = winner_of.get(var)
        if win_cell is None:
            print(f"  {var:<6}: n/a")
            continue
        col = f"{var}{suffix}"
        win_val = row_for_cell(df, keys, win_cell).get(col, np.nan)
        at_best = row_for_cell(df, keys, best).get(col, np.nan)
        flag = "  *" if win_cell == best else ""
        print(f"  {var:<6}: best {_cell_label(keys, win_cell)} ({col}={_fmt(win_val)})"
              f"  |  at best ({_cell_label(keys, best)}): {_fmt(at_best)}{flag}")
    print()

    # Full per-variable breakdown at the chosen best cell
    print(f"Per-variable RMSE at best {_cell_label(keys, best)}:")
    best_row = row_for_cell(df, keys, best)
    for var in vars_found:
        avg = best_row.get(f"{var}_avg", np.nan)
        low = best_row.get(f"{var}_min", np.nan)
        print(f"  {var:<6}: avg = {_fmt(avg)}   min = {_fmt(low)}")


if __name__ == "__main__":
    main()
