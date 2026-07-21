#!/usr/bin/env python3
"""Generate paper-quality EnSF-versus-LETKF RMSE figures.

This script reads the same cycle NetCDF files as error_plots_dual_nc.py, but it
only produces the figures needed for the paper:

1. One flagship figure per experiment. The WDG/WSG/TPH flagship contains
   level-averaged zonal and meridional wind as two panels.
2. One multi-panel figure containing the remaining variables.

All non-pressure variables are averaged over their valid SPEEDY levels. Surface
pressure is evaluated only at level 0 and is never described as level-averaged.
Each figure is saved as a high-resolution PNG and a vector PDF by default.

Edit only the USER SETTINGS section when switching experiments.
"""

from __future__ import annotations

###############################################################################
# USER SETTINGS
###############################################################################

# Choose one of: "all_linear", "all_arctangent", "wdg_wsg_tph",
# or "pressure_only".
EXPERIMENT = "wdg_wsg_tph"

# Directories containing the EnSF and LETKF cycle NetCDF files.
# The EnSF directory may still contain files named reverseSDE_cycle<k>.nc;
# "EnSF" is used only as the scientific label in the generated figures.
ENSF_DIR = (
    "/gpfs/home/jjs21b/AMLCS/runs/t21_80_0.05_30_ReverseSDE_1_1_100/wdg_wsg_tph_inflation/data"
)
LETKF_DIR = (
    "/gpfs/home/jjs21b/AMLCS/runs/t21_80_0.05_30_LETKF_2_1_115/wdg_wsg_tph_inflation/data"
)

# Contains snapshots/reference_solution_<k>.nc and free_run/free_run_<k>.nc.
REFERENCE_DIR = "/gpfs/home/jjs21b/AMLCS/LETKF_tuning/t21_80_0.05_30"

# Figures are written to OUTPUT_DIR/log and/or OUTPUT_DIR/linear.
OUTPUT_DIR = (
    "/gpfs/home/jjs21b/AMLCS/paper_figures/wdg_wsg_tph"
)

# Thirty archived cycles, indexed 0--29, correspond to assimilation cycles
# 1--30. When ANCHOR_AT_CYCLE_ZERO is True, the initial NoDA error is prepended
# to every curve and displayed at cycle 0.
CYCLES = list(range(30))
ANCHOR_AT_CYCLE_ZERO = True

# "log", "linear", or "both". The pressure-only flagship is normally taken
# from the linear directory; most other paper figures use the log directory.
SCALE_MODE = "both"

# Saving a PDF preserves vector text and curves for LaTeX. The PNG is useful
# for quick inspection and is rendered at publication-quality resolution.
SAVE_FORMATS = ("png", "pdf")
PNG_DPI = 400

###############################################################################
# END USER SETTINGS
###############################################################################

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from netCDF4 import Dataset


VARIABLES = {
    "UG1": {
        "name": "Zonal Wind",
        "symbol": r"$u$",
        "units": r"$\mathrm{m\,s^{-1}}$",
        "levels": tuple(range(8)),
    },
    "VG1": {
        "name": "Meridional Wind",
        "symbol": r"$v$",
        "units": r"$\mathrm{m\,s^{-1}}$",
        "levels": tuple(range(8)),
    },
    "TG1": {
        "name": "Temperature",
        "symbol": r"$T$",
        "units": r"$\mathrm{K}$",
        "levels": tuple(range(8)),
    },
    "TRG1": {
        "name": "Specific Humidity",
        "symbol": r"$q$",
        "units": r"$\mathrm{g\,kg^{-1}}$",
        "levels": tuple(range(2, 8)),
    },
    "PSG1": {
        "name": "Surface Pressure",
        "symbol": r"$p_s$",
        "units": r"$\log(p_s/P_0)$",
        "levels": (0,),
    },
}

# The first list forms one flagship figure. The second list forms one combined
# secondary figure. This encodes the scientific emphasis of each experiment.
EXPERIMENT_LAYOUTS = {
    "all_linear": {
        "flagship": ("TRG1",),
        "secondary": ("UG1", "VG1", "TG1", "PSG1"),
    },
    "all_arctangent": {
        "flagship": ("TRG1",),
        "secondary": ("UG1", "VG1", "TG1", "PSG1"),
    },
    "wdg_wsg_tph": {
        "flagship": ("UG1", "VG1"),
        "secondary": ("TG1", "TRG1", "PSG1"),
    },
    "pressure_only": {
        "flagship": ("PSG1",),
        "secondary": ("UG1", "VG1", "TG1", "TRG1"),
    },
}

# Preserve the established color scheme while making every curve explicit.
ENSF_COLOR = "tab:blue"
LETKF_COLOR = "tab:orange"
NODA_COLOR = "black"

CURVE_STYLES = (
    ("noda", "NoDA", NODA_COLOR, "-", 2.15),
    ("ensf_analysis", "EnSF Analysis", ENSF_COLOR, "-", 2.25),
    ("ensf_background", "EnSF Background", ENSF_COLOR, "--", 1.95),
    ("letkf_analysis", "LETKF Analysis", LETKF_COLOR, "-", 2.25),
    ("letkf_background", "LETKF Background", LETKF_COLOR, "--", 1.95),
)


@dataclass(frozen=True)
class ErrorSeries:
    """The five curves shown in every panel."""

    cycles: np.ndarray
    noda: np.ndarray
    ensf_analysis: np.ndarray
    ensf_background: np.ndarray
    letkf_analysis: np.ndarray
    letkf_background: np.ndarray


def _as_float_array(values) -> np.ndarray:
    """Convert NetCDF or masked values to a regular float array."""
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)


def _read_field(nc_path: Path, var: str, level: int) -> np.ndarray:
    """Read a model or prefixed two-dimensional field from a NetCDF file."""
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    with Dataset(nc_path, "r") as nc:
        for prefix in ("xa_mean", "xb_mean", "truth", "noda", "obs"):
            field_name = f"{prefix}_{var}_lev{level}"
            if field_name in nc.variables:
                return _as_float_array(nc.variables[field_name][:])

        if var not in nc.variables:
            raise KeyError(f"{var} level {level} was not found in {nc_path}")

        data = nc.variables[var]
        if data.ndim == 2:
            return _as_float_array(data[:])
        if data.ndim == 3:
            return _as_float_array(data[level, :, :])
        if data.ndim == 4:
            return _as_float_array(data[0, level, :, :])

        raise ValueError(
            f"Unsupported shape {data.shape} for {var} in {nc_path}"
        )


def _read_cycle_field(
    nc_path: Path, field_type: str, var: str, level: int
) -> np.ndarray:
    """Read an analysis or background ensemble-mean field."""
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    field_name = f"{field_type}_{var}_lev{level}"
    with Dataset(nc_path, "r") as nc:
        if field_name not in nc.variables:
            raise KeyError(f"{field_name} was not found in {nc_path}")
        return _as_float_array(nc.variables[field_name][:])


def _cycle_file(directory: Path, method: str, cycle: int) -> Path:
    """Resolve a cycle file while retaining the existing ReverseSDE filenames."""
    if method == "ensf":
        candidates = (
            directory / f"reverseSDE_cycle{cycle}.nc",
            directory / f"reversesde_cycle{cycle}.nc",
            directory / f"unified_cycle{cycle}.nc",
        )
    elif method == "letkf":
        candidates = (
            directory / f"letkf_cycle{cycle}.nc",
            directory / f"unified_cycle{cycle}.nc",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"No {method.upper()} file for cycle {cycle} in {directory}. "
        f"Tried: {names}"
    )


def _rmse(field: np.ndarray, truth: np.ndarray) -> float:
    """Calculate unweighted horizontal RMSE, matching the archived workflow."""
    difference = _as_float_array(field) - _as_float_array(truth)
    if not np.any(np.isfinite(difference)):
        return np.nan
    return float(np.sqrt(np.nanmean(difference**2)))


def _method_error_series(
    directory: Path,
    method: str,
    var: str,
    level: int,
    field_type: str,
    cycles: Iterable[int],
) -> np.ndarray:
    truth_dir = Path(REFERENCE_DIR).expanduser() / "snapshots"
    values = []

    for cycle in cycles:
        cycle_file = _cycle_file(directory, method, cycle)
        truth_file = truth_dir / f"reference_solution_{cycle}.nc"
        field = _read_cycle_field(cycle_file, field_type, var, level)
        truth = _read_field(truth_file, var, level)
        values.append(_rmse(field, truth))

    return np.asarray(values, dtype=float)


def _noda_error_series(
    var: str, level: int, cycles: Iterable[int]
) -> np.ndarray:
    reference_dir = Path(REFERENCE_DIR).expanduser()
    truth_dir = reference_dir / "snapshots"
    noda_dir = reference_dir / "free_run"
    values = []

    for cycle in cycles:
        truth = _read_field(
            truth_dir / f"reference_solution_{cycle}.nc", var, level
        )
        noda = _read_field(noda_dir / f"free_run_{cycle}.nc", var, level)
        values.append(_rmse(noda, truth))

    return np.asarray(values, dtype=float)


def _average_levels(level_series: list[np.ndarray], var: str) -> np.ndarray:
    """Average the independently calculated level RMSE values."""
    if not level_series:
        raise ValueError(f"No level series were calculated for {var}")

    lengths = {len(series) for series in level_series}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent cycle counts while averaging {var}")

    stacked = np.vstack(level_series)
    with np.errstate(invalid="ignore"):
        averaged = np.nanmean(stacked, axis=0)

    if not np.all(np.isfinite(averaged)):
        bad_cycles = np.flatnonzero(~np.isfinite(averaged)).tolist()
        raise ValueError(
            f"{var} has no finite RMSE value at cycle positions {bad_cycles}"
        )
    return averaged


def _prepend_anchor(series: np.ndarray, anchor: float) -> np.ndarray:
    return np.concatenate(([anchor], np.asarray(series, dtype=float)))


def _calculate_variable(var: str) -> ErrorSeries:
    """Calculate the five paper curves for one variable."""
    if var not in VARIABLES:
        raise KeyError(f"No metadata have been defined for {var}")

    ensf_dir = Path(ENSF_DIR).expanduser()
    letkf_dir = Path(LETKF_DIR).expanduser()
    levels = VARIABLES[var]["levels"]

    ensf_analysis = _average_levels(
        [
            _method_error_series(
                ensf_dir, "ensf", var, level, "xa_mean", CYCLES
            )
            for level in levels
        ],
        var,
    )
    ensf_background = _average_levels(
        [
            _method_error_series(
                ensf_dir, "ensf", var, level, "xb_mean", CYCLES
            )
            for level in levels
        ],
        var,
    )
    letkf_analysis = _average_levels(
        [
            _method_error_series(
                letkf_dir, "letkf", var, level, "xa_mean", CYCLES
            )
            for level in levels
        ],
        var,
    )
    letkf_background = _average_levels(
        [
            _method_error_series(
                letkf_dir, "letkf", var, level, "xb_mean", CYCLES
            )
            for level in levels
        ],
        var,
    )
    noda = _average_levels(
        [_noda_error_series(var, level, CYCLES) for level in levels], var
    )

    if ANCHOR_AT_CYCLE_ZERO:
        anchor = float(noda[0])
        ensf_analysis = _prepend_anchor(ensf_analysis, anchor)
        ensf_background = _prepend_anchor(ensf_background, anchor)
        letkf_analysis = _prepend_anchor(letkf_analysis, anchor)
        letkf_background = _prepend_anchor(letkf_background, anchor)
        noda = _prepend_anchor(noda, anchor)
        plot_cycles = np.arange(len(CYCLES) + 1)
    else:
        plot_cycles = np.arange(1, len(CYCLES) + 1)

    return ErrorSeries(
        cycles=plot_cycles,
        noda=noda,
        ensf_analysis=ensf_analysis,
        ensf_background=ensf_background,
        letkf_analysis=letkf_analysis,
        letkf_background=letkf_background,
    )


def _panel_title(var: str, panel_letter: str | None = None) -> str:
    info = VARIABLES[var]
    prefix = f"({panel_letter}) " if panel_letter else ""
    if var == "PSG1":
        return f"{prefix}{info['name']} [{info['symbol']}]"
    return f"{prefix}{info['name']} [{info['symbol']}]\nLevel-Averaged"


def _set_row_ylabels(axes: list[plt.Axes], variables: tuple[str, ...]) -> None:
    """Label RMSE once per row while retaining variable-specific units.

    The first panel receives ``RMSE [units]``. A later panel in the same row
    receives no label when its units match, or a short units-only label when
    they differ. This avoids repeating RMSE between adjacent panels without
    implying that temperature, humidity, and pressure share units.
    """
    if len(axes) != len(variables):
        raise ValueError("Each plotted axis must have a corresponding variable")

    first_units = VARIABLES[variables[0]]["units"]
    axes[0].set_ylabel(f"RMSE [{first_units}]")

    for ax, var in zip(axes[1:], variables[1:]):
        units = VARIABLES[var]["units"]
        ax.set_ylabel("")
        if units != first_units:
            ax.set_ylabel(f"[{units}]")
            ax.yaxis.set_label_position("right")


def _plot_panel(
    ax: plt.Axes,
    series: ErrorSeries,
    var: str,
    scale: str,
    panel_letter: str | None = None,
) -> None:
    for attribute, _label, color, linestyle, linewidth in CURVE_STYLES:
        values = getattr(series, attribute)
        if scale == "log":
            values = np.where(values > 0, values, np.nan)
        ax.plot(
            series.cycles,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=3 if "analysis" in attribute else 2,
        )

    if scale == "log":
        ax.set_yscale("log")

    ax.set_title(_panel_title(var, panel_letter), pad=10, fontweight="semibold")
    ax.set_xlabel("Assimilation Cycle")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(True, which="major", color="0.84", linewidth=0.75)
    ax.grid(True, which="minor", color="0.92", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.margins(x=0.015)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        for _attribute, label, color, linestyle, linewidth in CURVE_STYLES
    ]


def _add_shared_legend(fig: plt.Figure) -> None:
    handles = _legend_handles()
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.8,
        fontsize=8.8,
    )


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for file_format in SAVE_FORMATS:
        output_path = output_stem.with_suffix(f".{file_format}")
        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if file_format.lower() == "png":
            kwargs["dpi"] = PNG_DPI
        fig.savefig(output_path, **kwargs)
        print(f"Saved {output_path}")


def _flagship_figure(
    variables: tuple[str, ...],
    all_series: dict[str, ErrorSeries],
    scale: str,
) -> plt.Figure:
    if len(variables) == 1:
        fig, ax = plt.subplots(figsize=(7.2, 4.35))
        axes = [ax]
    elif len(variables) == 2:
        fig, axes_array = plt.subplots(1, 2, figsize=(7.4, 3.55), sharex=True)
        axes = list(np.atleast_1d(axes_array))
    else:
        raise ValueError("A flagship figure supports one or two variables")

    for index, (ax, var) in enumerate(zip(axes, variables)):
        panel_letter = chr(ord("a") + index) if len(variables) > 1 else None
        _plot_panel(ax, all_series[var], var, scale, panel_letter)

    _set_row_ylabels(axes, variables)

    _add_shared_legend(fig)
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.09, right=0.98, wspace=0.27)
    return fig


def _secondary_grid(count: int) -> tuple[int, int, tuple[float, float]]:
    if count == 1:
        return 1, 1, (7.2, 4.35)
    if count == 2:
        return 1, 2, (7.4, 3.55)
    if count == 4:
        return 2, 2, (7.4, 5.65)
    raise ValueError("The secondary figure supports one to four variables")


def _secondary_figure(
    variables: tuple[str, ...],
    all_series: dict[str, ErrorSeries],
    scale: str,
) -> plt.Figure:
    count = len(variables)
    if count == 3:
        # Two panels above and one centered panel below preserve readable text
        # at normal LaTeX page width without leaving an empty fourth panel.
        rows = 2
        fig = plt.figure(figsize=(7.4, 5.65))
        grid = fig.add_gridspec(2, 4)
        axes = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 1:3]),
        ]
    else:
        rows, columns, size = _secondary_grid(count)
        fig, axes_array = plt.subplots(
            rows, columns, figsize=size, squeeze=False, sharex=True
        )
        axes = list(axes_array.ravel())

    for index, (ax, var) in enumerate(zip(axes, variables)):
        panel_letter = chr(ord("a") + index) if len(variables) > 1 else None
        _plot_panel(ax, all_series[var], var, scale, panel_letter)

    if rows == 1:
        _set_row_ylabels(axes[:count], variables)
    elif count == 3:
        _set_row_ylabels(axes[:2], variables[:2])
        _set_row_ylabels([axes[2]], (variables[2],))
    else:
        _set_row_ylabels(axes[:2], variables[:2])
        _set_row_ylabels(axes[2:4], variables[2:4])

    if rows == 2:
        # The upper row shares the cycle coordinate with the lower row. Omitting
        # its repeated x label prevents collisions with the lower panel titles.
        for upper_ax in axes[:2]:
            upper_ax.set_xlabel("")
            upper_ax.tick_params(axis="x", labelbottom=False)

    for unused_ax in axes[count:]:
        unused_ax.set_visible(False)

    _add_shared_legend(fig)
    # Reserve enough space for the shared legend and the two-line panel titles.
    top = 0.72 if rows == 1 else 0.80
    fig.subplots_adjust(
        top=top,
        bottom=0.12,
        left=0.075,
        right=0.985,
        wspace=0.28,
        hspace=0.38,
    )
    return fig


def _requested_scales() -> tuple[str, ...]:
    normalized = SCALE_MODE.strip().lower()
    if normalized == "both":
        return ("log", "linear")
    if normalized in {"log", "linear"}:
        return (normalized,)
    raise ValueError('SCALE_MODE must be "log", "linear", or "both"')


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.7,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.linewidth": 0.9,
            "savefig.transparent": False,
        }
    )


def run() -> None:
    if EXPERIMENT not in EXPERIMENT_LAYOUTS:
        choices = ", ".join(EXPERIMENT_LAYOUTS)
        raise ValueError(f"Unknown EXPERIMENT {EXPERIMENT!r}. Choose: {choices}")

    _configure_matplotlib()
    layout = EXPERIMENT_LAYOUTS[EXPERIMENT]
    flagship_vars = layout["flagship"]
    secondary_vars = layout["secondary"]
    requested_vars = tuple(dict.fromkeys(flagship_vars + secondary_vars))

    print(f"Experiment: {EXPERIMENT}")
    print(f"Flagship variables: {', '.join(flagship_vars)}")
    print(f"Secondary variables: {', '.join(secondary_vars)}")

    all_series = {}
    for var in requested_vars:
        print(f"Calculating {var}...")
        all_series[var] = _calculate_variable(var)

    output_root = Path(OUTPUT_DIR).expanduser()
    for scale in _requested_scales():
        scale_dir = output_root / scale

        flagship = _flagship_figure(flagship_vars, all_series, scale)
        _save_figure(
            flagship,
            scale_dir / f"{EXPERIMENT}_flagship_{scale}",
        )
        plt.close(flagship)

        secondary = _secondary_figure(secondary_vars, all_series, scale)
        _save_figure(
            secondary,
            scale_dir / f"{EXPERIMENT}_secondary_{scale}",
        )
        plt.close(secondary)


if __name__ == "__main__":
    run()
