#!/usr/bin/env python3
"""
Dual-run error plotting: method1 vs method2 (five curves per figure).
Reads NetCDF cycle files directly instead of CSV exports.

NO COMMAND-LINE ARGUMENTS. Everything is configured in the USER SETTINGS
section below.

Generates two families of plots for each variable:
  (A) Per-level plots (each level per figure)
  (B) Level-averaged plots
"""

###############################################################################
# ======================= USER SETTINGS (EDIT THESE) ==========================
###############################################################################

# FULL PATHS to the two experiment directories you want to compare:

EXP1 = "/gpfs/home/jjs21b/AMLCS/runs/t21_80_0.05_30_ReverseSDE_1_1_100/wdg_wsg_tph_inflation/data"
EXP2 = "/gpfs/home/jjs21b/AMLCS/runs/t21_80_0.05_30_LETKF_2_1_115/wdg_wsg_tph_inflation/data"
# SPEEDY resolution:
RESOLUTION = "t21"

# Number of assimilation cycles to read (0-based indexing)
# Number of assimilation cycles to read (0-based indexing)
CYCLES = list(range(30))

# Base directory for reference solutions (Truth/NoDA)
REFERENCE_DIR = "/gpfs/home/jjs21b/AMLCS/LETKF_tuning/t21_80_0.05_30"

# Variables to compare (only model variables):
VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]

# Anchor mode: "step0" or "step1"
ANCHOR = "step1"

# Scale mode: "log", "linear", or "both"
SCALE_MODE = "both"

# Generate log plots? (Set False for absolute-only plots)
GENERATE_LOG_PLOTS = True

# Explicit Output Directory (optional)
# If set, plots will be saved here directly.
OUTPUT_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_80_0.05_30_ReverseSDE_1_1_100/wdg_wsg_tph_inflation/letkf_i115r2_vs_reverseSDE_wdg_wsg_tph"

# Output directory name (ignored if OUTPUT_DIR is set)
PLOT_DIR_NAME = None  

###############################################################################
# ======================= END USER SETTINGS ==================================
###############################################################################

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# imports from AMLCS
from grid_resolution import grid_resolution

# Formatting
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})

# Pretty variable names
VAR_CODES = {
    "TG0": "T_0", "UG0": "u_0", "VG0": "v_0", "TRG0": "Hq_0", "PSG0": "PS_0",
    "TG1": "T_1", "UG1": "u_1", "VG1": "v_1", "TRG1": "Hq_1", "PSG1": "PS_1",
    "WDG1": r"\theta_1",
}
PS_LEVELS_MB = [30, 100, 200, 300, 500, 700, 850, 925]

# Auto-assigned color palette (NoDA is always black)
COLOR_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
COLOR_NAMES = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]

###############################################################################
# --------------------- Utility Functions ------------------------------------
###############################################################################

def _parse_experiment_path(exp_path: Path) -> dict:
    """
    Parse experiment path to extract method and metadata.
    
    Examples:
        "t21_50_0.05_5_ReverseSDE_1_1_100" → {"method": "ReverseSDE", "metadata": ""}
        "t21_50_0.05_5_ReverseSDE_1_1_100/linear_normalization_results" → 
            {"method": "ReverseSDE", "metadata": "linear obs, normalization"}
    
    Returns:
        dict with keys: "method", "metadata", "full_label"
    """
    # Check if we're in a subdirectory with metadata
    full_path_str = str(exp_path)
    
    # Look for common metadata patterns in the path
    metadata_hints = []
    
    # Check for subdirectory metadata
    if exp_path.parent != exp_path:
        subdir = exp_path.name
        if "linear" in subdir.lower() and "nonlinear" not in subdir.lower():
            metadata_hints.append("linear obs")
        elif "nonlinear" in subdir.lower():
            metadata_hints.append("nonlinear obs")
        
        if "normalization" in subdir.lower() or "normalize" in subdir.lower():
            metadata_hints.append("normalization")
        elif "no_norm" in subdir.lower() or "unnorm" in subdir.lower():
            metadata_hints.append("no normalization")
    
    # Extract method name from base directory name
    # Extract method name from base directory name
    run_dir = exp_path
    # Search for the run directory (usually contains "t21")
    if "t21" not in run_dir.name:
        for parent in exp_path.parents:
            if "t21" in parent.name:
                run_dir = parent
                break
    
    # Fallback to immediate parent if logic fails
    if "t21" not in run_dir.name:
         run_dir = exp_path if exp_path.is_dir() else exp_path.parent

    name = run_dir.name.rstrip("/")
    parts = name.split("_")
    
    method = None
    if len(parts) > 7:
        mid = parts[4:-3]
        if mid:
            method = "_".join(mid)
    
    # Fallback method extraction
    if not method:
        for token in reversed(parts):
            try:
                float(token)
            except ValueError:
                method = token
                break
    
    if not method:
        method = name
    
    metadata = ", ".join(metadata_hints) if metadata_hints else ""
    
    return {
        "method": method,
        "metadata": metadata,
        "full_label": f"{method} ({metadata})" if metadata else method
    }


def _extract_method_name(exp_path: Path) -> str:
    """Legacy function for backwards compatibility."""
    return _parse_experiment_path(exp_path)["method"]


def _find_cycle_files(exp_path: Path, method: str):
    """
    Find all NetCDF cycle files for a given method.
    
    Patterns:
        ReverseSDE: reverseSDE_cycle<k>.nc
        EnKF_MC_obs: enkf_cycle<k>.nc
        Others: <method>_cycle<k>.nc
    """
    patterns = {
        "ReverseSDE": "reverseSDE_cycle*.nc",
        "EnKF_MC_obs": "unified_cycle*.nc",
    }
    pattern = patterns.get(method, f"{method.lower()}_cycle*.nc")
    files = sorted(exp_path.glob(pattern))
    
    # Fallback for ReverseSDE (or others) to unified_cycle if specific not found
    if not files:
        files = sorted(exp_path.glob("unified_cycle*.nc"))
        
    return files


def _compute_l2_error(field1: np.ndarray, field2: np.ndarray) -> float:
    """Compute L2 error between two fields."""
    diff = field1 - field2
    return np.sqrt(np.mean(diff**2))


def _read_nc_field(nc_path: Path, var: str, lev: int) -> np.ndarray:
    """
    Read a specific field from a NetCDF file.
    
    Args:
        nc_path: Path to NetCDF file
        var: Variable name (e.g., "TG1")
        lev: Level index
        
    Returns:
        2D array (lat, lon)
    """
    with Dataset(nc_path, 'r') as nc:
        # 1. Try split/prefixed fields (e.g. xa_mean_TG1_lev0)
        for prefix in ["xa_mean", "xb_mean", "truth", "noda", "obs"]:
            field_name = f"{prefix}_{var}_lev{lev}"
            if field_name in nc.variables:
                return nc.variables[field_name][:]
        
        # 2. Try raw variable name (common in reference files)
        if var in nc.variables:
            data = nc.variables[var]
            # Handle dimensions based on rank
            if data.ndim == 3:  # (nlev, lat, lon)
                return data[lev, :, :]
            if data.ndim == 2: # (lat, lon) e.g. PSG
                return data[:]
            if data.ndim == 4: # (ntr, lev, lat, lon) e.g. TRG
                return data[0, lev, :, :]

        # Helper to find component
        def find_comp(cname, target_lev):
            # Try prefixed
            for prefix in ["xa_mean", "xb_mean", "truth", "noda", "obs"]:
                fn = f"{prefix}_{cname}_lev{target_lev}"
                if fn in nc.variables: return nc.variables[fn][:]
            # Try raw
            if cname in nc.variables:
                d = nc.variables[cname]
                if d.ndim == 3: return d[target_lev, :, :]
                if d.ndim == 2: return d[:]
                if d.ndim == 4: return d[0, target_lev, :, :]
            return None

        # Special handling for PSG1 (Surface Pressure - only at lev0)
        if var == "PSG1":
            # Just try to find it at level 0, ignore requested 'lev' if distinct
            return find_comp("PSG1", 0)

        # Special handling for TRG1 (Humidity - only at levels 2-7)
        if var == "TRG1":
             # If requested level is < 2, return all zeros or NaN to indicate missing
             if lev < 2:
                 return np.zeros((32, 64)) # Hack: Return zeros for invalid levels
             return find_comp("TRG1", lev)
        
        # Standard variables
        res = find_comp(var, lev)
        if res is not None: return res

        # 3. Fallback for WDG1 (Derived from U/V if not present)
        if var == "WDG1":
            u_data = find_comp("UG1", lev)
            v_data = find_comp("VG1", lev)
            
            if u_data is not None and v_data is not None:
                return np.arctan2(v_data, u_data)

        # 4. Fallback for WSG1 (Derived from U/V if not present)
        if var == "WSG1":
            u_data = find_comp("UG1", lev)
            v_data = find_comp("VG1", lev)
            
            if u_data is not None and v_data is not None:
                return np.sqrt(u_data**2 + v_data**2)

        raise KeyError(f"Field {var} (lev {lev}) not found in {nc_path}")


def _compute_noda_series(exp_path: Path, var: str, lev: int, cycles: list) -> np.ndarray:
    """
    Compute NoDA error series from free_run snapshots.
    """
    errors = []
    # Use global REFERENCE_DIR
    ref_dir = Path(REFERENCE_DIR)
    free_run_dir = ref_dir / "free_run"
    truth_dir = ref_dir / "snapshots"
    
    for cycle_k in cycles:
        try:
            truth_file = truth_dir / f"reference_solution_{cycle_k}.nc"
            noda_file = free_run_dir / f"free_run_{cycle_k}.nc"
            
            truth = _read_nc_field(truth_file, var, lev)
            noda = _read_nc_field(noda_file, var, lev)
            
            error = _compute_l2_error(noda, truth)
            errors.append(error)
        except (FileNotFoundError, KeyError):
            # Suppress warning for excessive prints
            errors.append(np.nan)
    
    return np.array(errors)


def _compute_error_series(exp_path: Path, method: str, var: str, lev: int, cycles: list, field_type: str) -> np.ndarray:
    """
    Compute error series for analysis or background.
    
    Args:
        exp_path: Experiment directory
        method: Method name
        var: Variable name
        lev: Level index
        cycles: List of cycle indices
        field_type: "xa_mean" or "xb_mean"
        
    Returns:
        Array of L2 errors
    """
    errors = []
    
    ref_dir = Path(REFERENCE_DIR)
    truth_dir = ref_dir / "snapshots"

    for cycle_k in cycles:
        try:
            # Find the cycle file
            if method == "ReverseSDE":
                # Try specific first, then unified
                cycle_file = exp_path / f"reverseSDE_cycle{cycle_k}.nc"
                if not cycle_file.exists():
                    cycle_file = exp_path / f"unified_cycle{cycle_k}.nc"
            elif method == "EnKF_MC_obs":
                cycle_file = exp_path / f"unified_cycle{cycle_k}.nc"
            else:
                cycle_file = exp_path / f"{method.lower()}_cycle{cycle_k}.nc"
                
            # Generic fallback to unified_cycle
            if not cycle_file.exists():
                cycle_file = exp_path / f"unified_cycle{cycle_k}.nc"
            
            if not cycle_file.exists():
                # print(f"Warning: {cycle_file} not found") # Suppress
                errors.append(np.nan)
                continue
            
            # Read truth
            truth_file = truth_dir / f"reference_solution_{cycle_k}.nc"
            truth = _read_nc_field(truth_file, var, lev)
            
            # Read analysis/background field
            with Dataset(cycle_file, 'r') as nc:
                field_name = f"{field_type}_{var}_lev{lev}"
                field = nc.variables[field_name][:]
            
            error = _compute_l2_error(field, truth)
            errors.append(error)
            
        except (FileNotFoundError, KeyError) as e:
            # print(f"Warning: Error computing {field_type} for cycle {cycle_k}: {e}") # Suppress
            errors.append(np.nan)
    
    return np.array(errors)


def _make_anchor(series, anchor_val):
    if anchor_val is None:
        return series
    return np.concatenate([np.array([anchor_val]), series])


def _five_curves(ana1, bkg1, ana2, bkg2, noda, anchor_mode, scale, m1, m2):
    eps = 1e-12
    L = min(len(ana1), len(bkg1), len(ana2), len(bkg2), len(noda))
    if L == 0:
        return None, {}

    ana1, bkg1, ana2, bkg2, noda = (
        ana1[:L], bkg1[:L], ana2[:L], bkg2[:L], noda[:L]
    )
    anchor_val = noda[0] if anchor_mode == "step1" else None

    ana1 = _make_anchor(ana1, anchor_val)
    bkg1 = _make_anchor(bkg1, anchor_val)
    ana2 = _make_anchor(ana2, anchor_val)
    bkg2 = _make_anchor(bkg2, anchor_val)
    noda = _make_anchor(noda, anchor_val)

    xs = np.arange(len(noda))

    curves = {
        "NoDA": noda,
        f"{m1} Analysis": ana1,
        f"{m1} Background": bkg1,
        f"{m2} Analysis": ana2,
        f"{m2} Background": bkg2,
    }

    return xs, curves


def _plot_curves(xs, curves, title, out_path, scale, method1_info, method2_info):
    """
    Plot curves with color-coded methods in title and simplified legend.
    
    Args:
        method1_info: dict with 'method', 'metadata', 'color', 'color_name'
        method2_info: dict with 'method', 'metadata', 'color', 'color_name'
    """
    plt.figure(figsize=(9, 4))
    
    # Define line styles
    style_analysis = "-"
    style_background = "--"
    
    # Build labels
    m1_label = f"{method1_info['method']}"
    if method1_info['metadata']:
        m1_label += f" ({method1_info['metadata']})"
    
    m2_label = f"{method2_info['method']}"
    if method2_info['metadata']:
        m2_label += f" ({method2_info['metadata']})"
    
    # Plot in specific order
    order = [
        ("NoDA", "NoDA", "k", style_analysis),
        (f"{method1_info['method']} Analysis", "Analysis", method1_info['color'], style_analysis),
        (f"{method1_info['method']} Background", "Background", method1_info['color'], style_background),
        (f"{method2_info['method']} Analysis", "Analysis", method2_info['color'], style_analysis),
        (f"{method2_info['method']} Background", "Background", method2_info['color'], style_background),
    ]
    
    for curve_key, legend_label, color, linestyle in order:
        if curve_key not in curves:
            continue
        y = curves[curve_key]
        plt.plot(xs, y, label=legend_label, color=color, linestyle=linestyle)
    
    # Construct title with color info
    title_with_colors = (
        f"{title}\n"
        f"{m1_label} ({method1_info['color_name']}) vs {m2_label} ({method2_info['color_name']})"
    )
    
    plt.title(title_with_colors, fontsize=12)
    plt.xlabel("Assimilation Step")
    if scale == "log":
        plt.yscale("log")
        plt.ylabel("RMSE (Log Scale)")
    else:
        plt.ylabel("RMSE")
    
    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()



def _levels_for_var(var):
    if "PSG" in var:
        return [0]
    if var.startswith("TRG"):
        return list(range(2, 8))
    return list(range(8))


def _get_available_vars(exp_path: Path, method: str, candidates: list) -> set:
    """
    Check which of the candidate variables exist in the first available cycle file.
    """
    files = _find_cycle_files(exp_path, method)
    if not files:
        print(f"Warning: No cycle files found in {exp_path}")
        return set()
    
    # Check first file
    found = set()
    try:
        with Dataset(files[0], 'r') as nc:
            keys = set(nc.variables.keys())
            for var in candidates:
                # Check presence of first level for this var
                levels = _levels_for_var(var)
                if not levels:
                    continue
                first_lev = levels[0]
                
                # Look for standard prefixes
                for prefix in ["xa_mean", "xb_mean", "truth", "noda"]:
                    if f"{prefix}_{var}_lev{first_lev}" in keys:
                        found.add(var)
                        break
    except Exception as e:
        print(f"Warning: Could not check variables in {files[0]}: {e}")
        return set()
        
    return found

###############################################################################
# --------------------------- Main Work ---------------------------------------
###############################################################################

def run_dual_plots():
    exp1 = Path(EXP1).resolve()
    exp2 = Path(EXP2).resolve()

    # Parse experiment paths for method and metadata
    exp1_info = _parse_experiment_path(exp1)
    exp2_info = _parse_experiment_path(exp2)
    
    # Auto-assign colors
    exp1_info['color'] = COLOR_PALETTE[0]
    exp1_info['color_name'] = COLOR_NAMES[0]
    exp2_info['color'] = COLOR_PALETTE[1]
    exp2_info['color_name'] = COLOR_NAMES[1]

    method1 = exp1_info['method']
    method2 = exp2_info['method']

    # Filter variables based on existence in files
    print("Checking available variables...", flush=True)
    vars1 = _get_available_vars(exp1, method1, VARS)
    vars2 = _get_available_vars(exp2, method2, VARS)
    
    print(f"Vars in EXP1 ({exp1}): {vars1}", flush=True)
    print(f"Vars in EXP2 ({exp2}): {vars2}", flush=True)
    
    common_vars = vars1.intersection(vars2)
    print(f"Common Vars: {common_vars}", flush=True)
    
    active_vars = [v for v in VARS if v in common_vars]
    
    if not active_vars:
        print("Error: No common variables found from the requested list!")
        return

    out_name = PLOT_DIR_NAME or f"{method1}_vs_{method2}"

    print(f"Comparing:")
    print(f"  EXP1={exp1}")
    print(f"    Method: {method1}")
    print(f"    Metadata: {exp1_info['metadata'] or '(none)'}")
    print(f"    Color: {exp1_info['color_name']}")
    print(f"  EXP2={exp2}")
    print(f"    Method: {method2}")
    print(f"    Metadata: {exp2_info['metadata'] or '(none)'}")
    print(f"    Color: {exp2_info['color_name']}")
    print(f"Output directory tag: {out_name}")

    def _ensure(scale):
        if OUTPUT_DIR:
            d = Path(OUTPUT_DIR)
            if scale == "log":
                d = d / "log"
            elif scale == "linear" and GENERATE_LOG_PLOTS: # split if both
                 d = d / "linear"
            # else: just use d directly if only one mode
            d.mkdir(parents=True, exist_ok=True)
            return d

        # Legacy behavior
        if scale == "log":
            d1 = exp1 / out_name
            d2 = exp2 / out_name
        else:
            d1 = exp1 / f"{out_name}_abs"
            d2 = exp2 / f"{out_name}_abs"
        d1.mkdir(parents=True, exist_ok=True)
        d2.mkdir(parents=True, exist_ok=True)
        return d1  # only need one for saving

    # Determine scales to generate

    if SCALE_MODE == "both":
        scales = ["log", "linear"] if GENERATE_LOG_PLOTS else ["linear"]
    elif SCALE_MODE == "log" and not GENERATE_LOG_PLOTS:
        scales = ["linear"]  # Override to linear if log is disabled
    else:
        scales = [SCALE_MODE]

    print("Full output Plot Directories:")
    for scale in scales:
        if OUTPUT_DIR:
             base = Path(OUTPUT_DIR)
             if scale == "log":
                 print(f"  Log plots:     {base / 'log'}")
             elif scale == "linear" and GENERATE_LOG_PLOTS:
                 print(f"  Linear plots:  {base / 'linear'}")
             else:
                 print(f"  Plots:         {base}")
        else:
            if scale == "log":
                print(f"  Log plots:     {exp1 / out_name}")
            else:
                print(f"  Linear plots:  {exp1 / f'{out_name}_abs'}")

    # ------------------- LEVEL BY LEVEL -------------------
    for var in active_vars:
        lvls = _levels_for_var(var)

        for lvl in lvls:
            print(f"Processing {var} level {lvl}...")
            
            # Compute error series
            s_ana1 = _compute_error_series(exp1, method1, var, lvl, CYCLES, "xa_mean")
            s_bkg1 = _compute_error_series(exp1, method1, var, lvl, CYCLES, "xb_mean")
            s_ana2 = _compute_error_series(exp2, method2, var, lvl, CYCLES, "xa_mean")
            s_bkg2 = _compute_error_series(exp2, method2, var, lvl, CYCLES, "xb_mean")
            s_noda = _compute_noda_series(exp1, var, lvl, CYCLES)

            for scale in scales:
                out_dir = _ensure(scale)
                xs, curves = _five_curves(
                    s_ana1, s_bkg1, s_ana2, s_bkg2, s_noda,
                    ANCHOR, scale, method1, method2
                )
                if xs is None:
                    continue

                mb = PS_LEVELS_MB[lvl] if lvl < len(PS_LEVELS_MB) else lvl
                title = f"{VAR_CODES.get(var,var)} (lev {lvl}, {mb} mb)"
                out_file = out_dir / f"dual_single_{var}_lev{lvl}.png"
                _plot_curves(xs, curves, title, out_file, scale, exp1_info, exp2_info)

    # ------------------- LEVEL-AVERAGED -------------------
    for var in active_vars:
        lvls = _levels_for_var(var)
        
        print(f"Processing {var} (level-averaged)...")
        
        # Compute error series for each level
        ana1_levels = [_compute_error_series(exp1, method1, var, L, CYCLES, "xa_mean") for L in lvls]
        bkg1_levels = [_compute_error_series(exp1, method1, var, L, CYCLES, "xb_mean") for L in lvls]
        ana2_levels = [_compute_error_series(exp2, method2, var, L, CYCLES, "xa_mean") for L in lvls]
        bkg2_levels = [_compute_error_series(exp2, method2, var, L, CYCLES, "xb_mean") for L in lvls]
        noda_levels = [_compute_noda_series(exp1, var, L, CYCLES) for L in lvls]
        
        # Average across levels
        Lmin = min(len(x) for x in ana1_levels + bkg1_levels + ana2_levels + bkg2_levels + noda_levels)
        s_ana1 = np.vstack([x[:Lmin] for x in ana1_levels]).mean(axis=0)
        s_bkg1 = np.vstack([x[:Lmin] for x in bkg1_levels]).mean(axis=0)
        s_ana2 = np.vstack([x[:Lmin] for x in ana2_levels]).mean(axis=0)
        s_bkg2 = np.vstack([x[:Lmin] for x in bkg2_levels]).mean(axis=0)
        s_noda = np.vstack([x[:Lmin] for x in noda_levels]).mean(axis=0)

        for scale in scales:
            out_dir = _ensure(scale)
            xs, curves = _five_curves(
                s_ana1, s_bkg1, s_ana2, s_bkg2, s_noda,
                ANCHOR, scale, method1, method2
            )
            if xs is None:
                continue

            title = f"{VAR_CODES.get(var,var)} (Level Average)"
            out_file = out_dir / f"dual_levelavg_{var}.png"
            _plot_curves(xs, curves, title, out_file, scale, exp1_info, exp2_info)


###############################################################################
# ------------------------------ ENTRY POINT ----------------------------------
###############################################################################

if __name__ == "__main__":
    run_dual_plots()
