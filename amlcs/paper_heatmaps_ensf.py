#!/usr/bin/env python3
"""Generate flagship EnSF analysis-error heatmaps for the project paper.

The original heatmap_gifs_nc.py script calculates analysis error as

    abs(analysis - truth)

for the SPEEDY state variables used here. This paper-specific script preserves
that calculation and its zero-inclusive symmetric-log color normalization, but
replaces animated GIFs with one static, publication-quality PNG:

* All-linear and all-arctangent experiments: specific humidity at cycles 1--2.
* Pressure-only experiment: surface pressure at cycles 1--2.
* WDG/WSG/TPH experiment: zonal and meridional wind at cycles 1--2.

For multilevel variables, the heatmaps show the configured SPEEDY level (level
7 by default), matching the original script. Pressure is read at level 0.

Edit only the USER SETTINGS section when switching experiments.
"""

from __future__ import annotations

###############################################################################
# USER SETTINGS
###############################################################################

# Choose one of: "all_linear", "all_arctangent", "wdg_wsg_tph",
# or "pressure_only".
EXPERIMENT = "wdg_wsg_tph"

# Directory containing reverseSDE_cycle<k>.nc or unified_cycle<k>.nc files.
ENSF_DIR = (
    "/gpfs/home/jjs21b/AMLCS/runs/"
    "t21_80_0.05_30_ReverseSDE_1_1_100/wdg_wsg/data"
)

# Contains snapshots/reference_solution_<k>.nc.
REFERENCE_DIR = "/gpfs/home/jjs21b/AMLCS/LETKF_tuning/t21_80_0.05_30"

OUTPUT_DIR = "/gpfs/home/jjs21b/AMLCS/paper_figures/wdg_wsg_tph"

# Archived cycles 0 and 1 are displayed as assimilation cycles 1 and 2.
CYCLES = (0, 1)

# Applied to UG1, VG1, TG1, and TRG1. PSG1 always uses level 0.
LEVEL_INDEX = 7

# The original GIF script used the full finite range. A value below 100 can be
# used later if an isolated outlier makes the maps difficult to interpret.
COLOR_LIMIT_PERCENTILE = 100.0

PNG_DPI = 400

###############################################################################
# END USER SETTINGS
###############################################################################

from pathlib import Path
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    HAS_CARTOPY = True
except ImportError:
    ccrs = None
    cfeature = None
    LATITUDE_FORMATTER = None
    LONGITUDE_FORMATTER = None
    HAS_CARTOPY = False

try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None


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

EXPERIMENT_VARIABLES = {
    "all_linear": ("TRG1",),
    "all_arctangent": ("TRG1",),
    "wdg_wsg_tph": ("UG1", "VG1"),
    "pressure_only": ("PSG1",),
}


def _as_float_array(values) -> np.ndarray:
    """Convert a NetCDF or masked array to a regular floating-point array."""
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)


def _ensure_2d(values: np.ndarray, description: str) -> np.ndarray:
    values = np.squeeze(_as_float_array(values))
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D field for {description}; got {values.shape}")
    return values


def _cycle_number(path: Path) -> int:
    match = re.search(r"cycle(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _cycle_file(cycle: int) -> Path:
    """Resolve an EnSF cycle file while retaining existing ReverseSDE names."""
    directory = Path(ENSF_DIR).expanduser()
    candidates = (
        directory / f"reverseSDE_cycle{cycle}.nc",
        directory / f"reversesde_cycle{cycle}.nc",
        directory / f"unified_cycle{cycle}.nc",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Handle minor filename-capitalization differences without relying on the
    # filesystem's case sensitivity.
    available = sorted(directory.glob("*cycle*.nc"), key=_cycle_number)
    for candidate in available:
        if _cycle_number(candidate) == cycle:
            return candidate

    tried = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"No EnSF file for archived cycle {cycle} in {directory}. Tried: {tried}"
    )


def _analysis_field(cycle_file: Path, var: str) -> np.ndarray:
    """Read the ensemble-mean analysis field from an EnSF cycle file."""
    if Dataset is None:
        raise ImportError("Reading the experiment files requires netCDF4")
    level = 0 if var == "PSG1" else LEVEL_INDEX
    field_name = f"xa_mean_{var}_lev{level}"

    with Dataset(cycle_file, "r") as nc:
        if field_name not in nc.variables:
            raise KeyError(f"{field_name} was not found in {cycle_file}")
        return _ensure_2d(nc.variables[field_name][:], field_name)


def _truth_field(cycle: int, var: str) -> np.ndarray:
    """Read the matching truth field from the reference snapshots."""
    if Dataset is None:
        raise ImportError("Reading the reference files requires netCDF4")
    truth_file = (
        Path(REFERENCE_DIR).expanduser()
        / "snapshots"
        / f"reference_solution_{cycle}.nc"
    )
    if not truth_file.exists():
        raise FileNotFoundError(truth_file)

    level = 0 if var == "PSG1" else LEVEL_INDEX
    with Dataset(truth_file, "r") as nc:
        if var not in nc.variables:
            raise KeyError(f"{var} was not found in {truth_file}")

        data = nc.variables[var]
        if data.ndim == 2:
            values = data[:]
        elif data.ndim == 3:
            values = data[level, :, :]
        elif data.ndim == 4:
            values = data[0, level, :, :]
        else:
            raise ValueError(
                f"Unsupported shape {data.shape} for {var} in {truth_file}"
            )

    return _ensure_2d(values, f"truth {var}, cycle {cycle}")


def _absolute_analysis_error(cycle: int, var: str) -> np.ndarray:
    """Calculate |analysis - truth| exactly as in the source heatmap script."""
    analysis = _analysis_field(_cycle_file(cycle), var)
    truth = _truth_field(cycle, var)
    if analysis.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch for {var}, cycle {cycle}: "
            f"analysis {analysis.shape}, truth {truth.shape}"
        )
    return np.abs(analysis - truth)


def _color_norm(frames: list[np.ndarray]) -> mcolors.SymLogNorm:
    """Build the zero-inclusive symmetric-log normalization used previously."""
    finite_values = np.concatenate(
        [frame[np.isfinite(frame)].ravel() for frame in frames]
    )
    if finite_values.size == 0:
        raise ValueError("Every requested analysis-error field contains only NaNs")

    percentile = float(COLOR_LIMIT_PERCENTILE)
    if not 0 < percentile <= 100:
        raise ValueError("COLOR_LIMIT_PERCENTILE must be in (0, 100]")

    vmax = float(np.nanpercentile(finite_values, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    linthresh = max(vmax * 0.01, np.finfo(float).tiny)
    return mcolors.SymLogNorm(linthresh=linthresh, vmin=0.0, vmax=vmax)


def _configure_axis(
    ax: plt.Axes,
    row: int,
    column: int,
    rows: int,
    columns: int,
) -> None:
    if HAS_CARTOPY:
        projection = ccrs.PlateCarree()
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=0.65, color="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.28, edgecolor="0.35")

        gridlines = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.35,
            color="0.35",
            alpha=0.38,
            linestyle=":",
            x_inline=False,
            y_inline=False,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.bottom_labels = row == rows - 1
        gridlines.left_labels = column == 0
        gridlines.xformatter = LONGITUDE_FORMATTER
        gridlines.yformatter = LATITUDE_FORMATTER
        gridlines.xlabel_style = {"size": 8}
        gridlines.ylabel_style = {"size": 8}
    else:
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(0, 361, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, linewidth=0.35, color="0.35", alpha=0.38, linestyle=":")
        ax.tick_params(
            labelbottom=row == rows - 1,
            labelleft=column == 0,
            labelsize=8,
        )

    if row == rows - 1:
        ax.set_xlabel("Longitude", labelpad=14)
    if column == 0:
        ax.set_ylabel("Latitude", labelpad=18)


def _panel_title(
    var: str,
    cycle: int,
    panel_index: int,
    include_variable: bool,
) -> str:
    letter = chr(ord("a") + panel_index)
    cycle_label = f"Assimilation Cycle {cycle + 1}"
    if include_variable:
        info = VARIABLES[var]
        return f"({letter}) {info['name']} [{info['symbol']}]\n{cycle_label}"
    return f"({letter}) {cycle_label}"


def _figure_title(variables: tuple[str, ...]) -> str:
    if len(variables) == 1:
        var = variables[0]
        info = VARIABLES[var]
        level_text = "" if var == "PSG1" else f", SPEEDY Level {LEVEL_INDEX}"
        return (
            f"EnSF $|$Analysis $-$ Truth$|$: "
            f"{info['name']} [{info['symbol']}]{level_text}"
        )

    return (
        f"EnSF $|$Analysis $-$ Truth$|$: Wind Components, "
        f"SPEEDY Level {LEVEL_INDEX}"
    )


def _plot_heatmaps(
    variables: tuple[str, ...],
    errors: dict[tuple[str, int], np.ndarray],
) -> plt.Figure:
    if len(variables) == 1:
        rows, columns = 1, 2
        figsize = (9.4, 3.85)
    elif len(variables) == 2:
        rows, columns = 2, 2
        figsize = (9.4, 6.65)
    else:
        raise ValueError("Paper heatmaps support one or two flagship variables")

    ordered_frames = [
        errors[(var, cycle)] for var in variables for cycle in CYCLES
    ]
    norm = _color_norm(ordered_frames)
    projection = ccrs.PlateCarree() if HAS_CARTOPY else None
    subplot_kw = {"projection": projection} if HAS_CARTOPY else {}
    fig, axes_array = plt.subplots(
        rows,
        columns,
        figsize=figsize,
        squeeze=False,
        subplot_kw=subplot_kw,
    )
    axes = list(axes_array.ravel())

    image = None
    panel_index = 0
    for row, var in enumerate(variables):
        for column, cycle in enumerate(CYCLES):
            ax = axes_array[row, column]
            _configure_axis(ax, row, column, rows, columns)
            image_kwargs = {"transform": projection} if HAS_CARTOPY else {}
            image = ax.imshow(
                errors[(var, cycle)],
                origin="lower",
                extent=[0, 360, -90, 90],
                interpolation="nearest",
                cmap="viridis",
                norm=norm,
                **image_kwargs,
            )
            ax.set_title(
                _panel_title(
                    var,
                    cycle,
                    panel_index,
                    include_variable=len(variables) > 1,
                ),
                fontsize=10.5,
                fontweight="semibold",
                pad=8,
            )
            panel_index += 1

    if image is None:
        raise RuntimeError("No heatmap panels were created")

    units = {VARIABLES[var]["units"] for var in variables}
    if len(units) != 1:
        raise ValueError("A shared colorbar requires all variables to use one unit")
    unit = next(iter(units))

    fig.suptitle(
        _figure_title(variables),
        fontsize=12.5,
        fontweight="semibold",
        y=0.985,
    )
    if rows == 1:
        fig.subplots_adjust(
            left=0.055,
            right=0.865,
            bottom=0.16,
            top=0.80,
            wspace=0.12,
        )
    else:
        fig.subplots_adjust(
            left=0.055,
            right=0.865,
            bottom=0.09,
            top=0.86,
            wspace=0.12,
            hspace=0.22,
        )

    # A dedicated axis keeps the shared colorbar outside the rightmost map.
    colorbar_axis = fig.add_axes([0.895, 0.18, 0.018, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="vertical")
    colorbar.set_label(f"Absolute Analysis Error [{unit}]", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8.5)
    return fig


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "savefig.transparent": False,
        }
    )


def run() -> None:
    if EXPERIMENT not in EXPERIMENT_VARIABLES:
        choices = ", ".join(EXPERIMENT_VARIABLES)
        raise ValueError(f"Unknown EXPERIMENT {EXPERIMENT!r}. Choose: {choices}")
    if len(CYCLES) != 2:
        raise ValueError("CYCLES must contain exactly two archived cycle indices")

    _configure_matplotlib()
    variables = EXPERIMENT_VARIABLES[EXPERIMENT]
    print(f"Experiment: {EXPERIMENT}")
    print(f"Flagship variables: {', '.join(variables)}")
    print(f"Archived cycles: {CYCLES} (displayed as cycles 1 and 2)")

    errors = {}
    for var in variables:
        for cycle in CYCLES:
            print(f"Calculating |analysis - truth| for {var}, cycle {cycle}...")
            errors[(var, cycle)] = _absolute_analysis_error(cycle, var)

    figure = _plot_heatmaps(variables, errors)
    output_dir = Path(OUTPUT_DIR).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{EXPERIMENT}_ensf_analysis_error_cycles_1_2.png"
    figure.savefig(
        output_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    run()
