#!/usr/bin/env python3
"""Generate paper-quality EnSF reverse-SDE trajectory figures.

This script replaces the exploratory output from spaghetti_plots_v2.py with
one focused figure for the flagship variable(s) in a selected experiment.  For
each flagship variable, the figure contains exactly two panels:

1. the spatial-mean SDE state; and
2. the tracked grid point with the largest analysis increment.

The plotted trajectories use physical model units.  The internal normalized
state is useful for debugging the numerical solver, but physical units make the
figure interpretable as an atmospheric data-assimilation result.  The reverse
SDE is shown from pseudo-time tau=1 (Gaussian starting state) to approximately
tau=0 (the last state saved immediately before the final Euler step).

Flagship variables:

* all-linear and all-arctangent: specific humidity;
* WDG/WSG/TPH: zonal and meridional wind; and
* pressure-only: surface pressure.

The existing NetCDF variable and run-directory names may still contain
"ReverseSDE".  "EnSF" is used as the scientific method name in the figure.

Edit only the USER SETTINGS section when switching experiments.
"""

from __future__ import annotations

###############################################################################
# USER SETTINGS
###############################################################################

# Choose one of: "all_linear", "all_arctangent", "wdg_wsg_tph",
# or "pressure_only".
EXPERIMENT = "wdg_wsg_tph"

# Path to the sde_tracking.nc file for the selected experiment.
NC_PATH = (
    "/gpfs/home/jjs21b/AMLCS/runs/"
    "t21_80_0.05_30_ReverseSDE_1_1_100/wdg_wsg/data/sde_tracking.nc"
)

OUTPUT_DIR = "/gpfs/home/jjs21b/AMLCS/paper_figures/wdg_wsg_tph"

# Archived cycle 0 is the first assimilation cycle.
CYCLE_INDEX = 0

# The tracking order must match track_gridpoint_locs in sequential_methods.py.
# "largest_analysis_increment" is preferred because it displays the location
# where the EnSF update acted most strongly.  The background-error location is
# retained as an explicit alternative, but the script never plots both.
REPRESENTATIVE_POINT = "largest_analysis_increment"
TRACKED_POINTS = {
    "largest_background_error": {
        "index": 0,
        "latitude_index": 8,
        "longitude_index": 31,
        "label": "Largest Initial Background Error",
    },
    "largest_analysis_increment": {
        "index": 1,
        "latitude_index": 24,
        "longitude_index": 36,
        "label": "Largest Analysis Increment",
    },
}

PNG_DPI = 400

###############################################################################
# END USER SETTINGS
###############################################################################

from pathlib import Path
import string
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from netCDF4 import Dataset, chartostring
except ImportError:
    Dataset = None
    chartostring = None


VARIABLES = {
    "UG1": {
        "name": "Zonal Wind",
        "symbol": r"$u$",
        "units": r"$\mathrm{m\,s^{-1}}$",
    },
    "VG1": {
        "name": "Meridional Wind",
        "symbol": r"$v$",
        "units": r"$\mathrm{m\,s^{-1}}$",
    },
    "TG1": {
        "name": "Temperature",
        "symbol": r"$T$",
        "units": r"$\mathrm{K}$",
    },
    "TRG1": {
        "name": "Specific Humidity",
        "symbol": r"$q$",
        "units": r"$\mathrm{g\,kg^{-1}}$",
    },
    "PSG1": {
        "name": "Surface Pressure",
        "symbol": r"$p_s$",
        "units": r"$\log(p_s/P_0)$",
    },
}

EXPERIMENTS = {
    "all_linear": {
        "name": "All-Linear Observations",
        "flagship": ("TRG1",),
    },
    "all_arctangent": {
        "name": "All-Arctangent Observations",
        "flagship": ("TRG1",),
    },
    "wdg_wsg_tph": {
        "name": "Wind-Direction/Wind-Speed Observations",
        "flagship": ("UG1", "VG1"),
    },
    "pressure_only": {
        "name": "Pressure-Only Observations",
        "flagship": ("PSG1",),
    },
}

MEMBER_COLOR = "tab:blue"
MEAN_COLOR = "#08306b"
START_COLOR = "tab:orange"
ANALYSIS_COLOR = "tab:green"


def _decode_var_names(raw_values) -> list[str]:
    """Decode NetCDF string, byte-string, or character-array names."""
    raw = np.asarray(raw_values)
    if chartostring is not None and raw.dtype.kind in {"S", "U"} and raw.ndim > 1:
        try:
            raw = np.asarray(chartostring(raw))
        except (TypeError, ValueError):
            pass

    decoded: list[str] = []
    for value in np.atleast_1d(raw):
        if isinstance(value, str):
            decoded.append(value.strip())
        elif isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(bytes(value).decode("utf-8").strip())
        elif np.asarray(value).ndim > 0:
            chars = np.asarray(value).ravel()
            pieces = [
                bytes(item).decode("utf-8")
                if isinstance(item, (bytes, bytearray, np.bytes_))
                else str(item)
                for item in chars
            ]
            decoded.append("".join(pieces).strip())
        else:
            decoded.append(str(value).strip())
    return decoded


def _to_float(values, fill_value=None) -> np.ndarray:
    """Convert a masked NetCDF slice to float and replace fill values."""
    array = np.asarray(np.ma.filled(values, np.nan), dtype=float)
    if fill_value is not None:
        array[array == float(fill_value)] = np.nan
    array[np.abs(array) > 1.0e30] = np.nan
    return array


def _average_valid_blocks(values: np.ndarray, description: str) -> np.ndarray:
    """Average the tracking array over blocks that contain finite values."""
    if values.ndim < 2:
        raise ValueError(f"Unexpected shape {values.shape} for {description}")

    reduction_axes = tuple(range(1, values.ndim))
    finite_counts = np.sum(np.isfinite(values), axis=reduction_axes)
    valid_blocks = np.flatnonzero(finite_counts > 0)
    if valid_blocks.size == 0:
        raise ValueError(f"No finite blocks were found for {description}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values[valid_blocks], axis=0)


def _read_trajectories(
    path: Path,
    cycle_index: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read physical spatial-mean and tracked-grid-point trajectories.

    Returns
    -------
    var_names
        Variable codes in the NetCDF file.
    mean_data
        Array with shape (pseudo-time, variable, ensemble).
    point_data
        Array with shape (pseudo-time, variable, point, ensemble).
    """
    if Dataset is None:
        raise ImportError("Reading sde_tracking.nc requires netCDF4")
    if not path.exists():
        raise FileNotFoundError(path)

    with Dataset(path, "r") as nc:
        if "var_names" not in nc.variables:
            raise KeyError(f"var_names was not found in {path}")
        var_names = _decode_var_names(nc["var_names"][:])

        mean_name = "xt_state_mean" if "xt_state_mean" in nc.variables else "xt_state"
        required = (mean_name, "xt_state_gridpoint")
        missing = [name for name in required if name not in nc.variables]
        if missing:
            raise KeyError(f"Missing {', '.join(missing)} in {path}")

        mean_var = nc[mean_name]
        point_var = nc["xt_state_gridpoint"]
        if not 0 <= cycle_index < mean_var.shape[0]:
            raise IndexError(
                f"CYCLE_INDEX={cycle_index} is outside the available range "
                f"0--{mean_var.shape[0] - 1}"
            )

        mean_cycle = _to_float(
            mean_var[cycle_index], getattr(mean_var, "_FillValue", None)
        )
        point_cycle = _to_float(
            point_var[cycle_index], getattr(point_var, "_FillValue", None)
        )

    # After selecting a cycle, expected shapes are:
    # mean:  (block, pseudo-time, variable, ensemble)
    # point: (block, pseudo-time, variable, point, ensemble)
    if mean_cycle.ndim != 4:
        raise ValueError(
            f"Expected four dimensions after selecting the mean cycle; "
            f"got {mean_cycle.shape}"
        )
    if point_cycle.ndim != 5:
        raise ValueError(
            f"Expected five dimensions after selecting the point cycle; "
            f"got {point_cycle.shape}"
        )

    mean_data = _average_valid_blocks(mean_cycle, mean_name)
    point_data = _average_valid_blocks(point_cycle, "xt_state_gridpoint")
    return var_names, mean_data, point_data


def _finite_last_step(values: np.ndarray) -> int:
    """Return the last pseudo-time index containing at least one finite member."""
    valid = np.flatnonzero(np.any(np.isfinite(values), axis=1))
    if valid.size == 0:
        raise ValueError("Trajectory contains no finite ensemble values")
    return int(valid[-1])


def _set_robust_limits(ax: plt.Axes, values: np.ndarray) -> None:
    """Set stable limits without allowing isolated solver values to dominate."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return
    low, high = np.nanpercentile(finite, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high):
        return
    if np.isclose(low, high):
        pad = max(abs(float(low)) * 0.05, 1.0e-6)
    else:
        pad = 0.06 * float(high - low)
    ax.set_ylim(float(low - pad), float(high + pad))


def _plot_panel(ax: plt.Axes, values: np.ndarray) -> None:
    """Draw ensemble reverse-SDE paths and their ensemble mean."""
    if values.ndim != 2:
        raise ValueError(f"Expected (pseudo-time, ensemble), got {values.shape}")
    if not np.any(np.isfinite(values)):
        raise ValueError("Requested trajectory panel contains only NaNs")

    final_index = _finite_last_step(values)
    values = values[: final_index + 1]
    # sequential_methods.py records the state before every Euler update.  With
    # N saved steps, the last recorded state is therefore at tau=1/N rather
    # than exactly zero.
    pseudo_time = np.arange(values.shape[0], 0, -1, dtype=float) / values.shape[0]

    for member in range(values.shape[1]):
        ax.plot(
            pseudo_time,
            values[:, member],
            color=MEMBER_COLOR,
            linewidth=0.65,
            alpha=0.24,
            rasterized=True,
            zorder=1,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ensemble_mean = np.nanmean(values, axis=1)
    ax.plot(
        pseudo_time,
        ensemble_mean,
        color=MEAN_COLOR,
        linewidth=2.25,
        zorder=3,
    )
    ax.scatter(
        [pseudo_time[0]],
        [ensemble_mean[0]],
        s=34,
        marker="o",
        color=START_COLOR,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax.scatter(
        [pseudo_time[-1]],
        [ensemble_mean[-1]],
        s=38,
        marker="s",
        color=ANALYSIS_COLOR,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )

    _set_robust_limits(ax, values)
    ax.set_xlim(1.0, 0.0)
    ax.set_xticks((1.0, 0.75, 0.5, 0.25, 0.0))
    ax.grid(True, color="0.82", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5)


def _panel_title(panel_letter: str, var: str, location: str) -> str:
    metadata = VARIABLES[var]
    return (
        f"({panel_letter}) {metadata['name']} [{metadata['symbol']}] "
        f"({metadata['units']})\n{location}"
    )


def make_figure(
    var_names: list[str],
    mean_data: np.ndarray,
    point_data: np.ndarray,
) -> Path:
    """Create the selected experiment's single flagship trajectory figure."""
    if EXPERIMENT not in EXPERIMENTS:
        choices = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown EXPERIMENT={EXPERIMENT!r}; choose from {choices}")
    if REPRESENTATIVE_POINT not in TRACKED_POINTS:
        choices = ", ".join(TRACKED_POINTS)
        raise ValueError(
            f"Unknown REPRESENTATIVE_POINT={REPRESENTATIVE_POINT!r}; "
            f"choose from {choices}"
        )

    experiment = EXPERIMENTS[EXPERIMENT]
    flagship = experiment["flagship"]
    missing_vars = [var for var in flagship if var not in var_names]
    if missing_vars:
        raise KeyError(
            f"Flagship variable(s) {', '.join(missing_vars)} were not found. "
            f"Available names: {', '.join(var_names)}"
        )

    point = TRACKED_POINTS[REPRESENTATIVE_POINT]
    point_index = int(point["index"])
    if not 0 <= point_index < point_data.shape[2]:
        raise IndexError(
            f"Tracked point index {point_index} is outside the available range "
            f"0--{point_data.shape[2] - 1}"
        )

    rows = len(flagship)
    figure_height = 3.9 if rows == 1 else 6.8
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(10.8, figure_height),
        squeeze=False,
        sharex=True,
    )

    panel_letters = iter(string.ascii_lowercase)
    for row, var in enumerate(flagship):
        var_index = var_names.index(var)
        panels = (
            (axes[row, 0], mean_data[:, var_index, :], "Spatial Mean"),
            (
                axes[row, 1],
                point_data[:, var_index, point_index, :],
                f"{point['label']} Grid Point "
                f"(Grid Index {point['latitude_index']}, "
                f"{point['longitude_index']})",
            ),
        )

        for column, (ax, values, location) in enumerate(panels):
            _plot_panel(ax, values)
            ax.set_title(
                _panel_title(next(panel_letters), var, location),
                fontsize=10.7,
                pad=7,
            )
            if column == 0:
                ax.set_ylabel("Physical SDE State", fontsize=10.5)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Reverse-SDE Pseudo-Time, $\tau$", fontsize=10.5)

    legend_handles = (
        Line2D([0], [0], color=MEMBER_COLOR, linewidth=1.0, alpha=0.45),
        Line2D([0], [0], color=MEAN_COLOR, linewidth=2.25),
        Line2D(
            [0], [0], marker="o", linestyle="None", markersize=6.5,
            markerfacecolor=START_COLOR, markeredgecolor="white",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="None", markersize=6.5,
            markerfacecolor=ANALYSIS_COLOR, markeredgecolor="white",
        ),
    )
    legend_labels = (
        "EnSF Ensemble Members",
        "Ensemble Mean",
        r"Start ($\tau=1$)",
        r"Analysis Endpoint ($\tau\approx0$)",
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
        columnspacing=1.55,
    )

    displayed_cycle = CYCLE_INDEX + 1
    fig.suptitle(
        f"EnSF Reverse-SDE Trajectories: {experiment['name']}, "
        f"Assimilation Cycle {displayed_cycle}",
        fontsize=12.2,
        y=0.985,
    )
    bottom_margin = 0.19 if rows == 1 else 0.12
    fig.tight_layout(rect=(0.02, bottom_margin, 0.98, 0.94), h_pad=1.3, w_pad=1.2)

    output_dir = Path(OUTPUT_DIR).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ensf_sde_trajectories_{EXPERIMENT}.png"
    fig.savefig(
        output_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return output_path


def main() -> None:
    path = Path(NC_PATH).expanduser()
    print(f"Reading physical EnSF trajectories from {path}")
    var_names, mean_data, point_data = _read_trajectories(path, CYCLE_INDEX)
    output_path = make_figure(var_names, mean_data, point_data)
    point = TRACKED_POINTS[REPRESENTATIVE_POINT]
    print(
        f"Used tracked point {point['index']}: {point['label']} "
        f"at grid index ({point['latitude_index']}, {point['longitude_index']})"
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
