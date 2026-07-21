#!/usr/bin/env python3
"""
DA Output Inspector
===================
Quickly checks `unified_cycle*.nc` files to verify if Data Assimilation (DA) worked.

Configuration:
    Modify PATHS_TO_INSPECT below.
"""

import os
import glob
import numpy as np
from netCDF4 import Dataset

###############################################################################
# ========================= CONFIGURATION ====================================
###############################################################################

# List of files or directories to inspect
PATHS_TO_INSPECT = ["/gpfs/home/jjs21b/AMLCS/runs/t21_80_0.05_30_LETKF_1_1_100"]
    # Add more paths here (can be directory or specific file)


# Set to True for more detailed output
VERBOSE = False

###############################################################################
# ========================= END CONFIGURATION ================================
###############################################################################

def get_stats(arr):
    """Compute robust statistics."""
    if arr is None: return None
    arr = np.asanyarray(arr)
    if arr.size == 0: return None
    
    # Check for non-finites
    if not np.isfinite(arr).all():
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        return {'valid': False, 'n_nan': n_nan, 'n_inf': n_inf}
    
    return {
        'valid': True,
        'min': np.min(arr),
        'max': np.max(arr),
        'mean': np.mean(arr),
        'rms': np.sqrt(np.mean(arr**2)),
        'abs_max': np.max(np.abs(arr))
    }

def inspect_file(filepath, verbose=False, label=""):
    print(f"\n>> Inspecting: {label} ({os.path.basename(filepath)})")
    print(f"   Full Path: {filepath}")
    try:
        with Dataset(filepath, 'r') as nc:
            if 'sde_tracking' in os.path.basename(filepath):
                print(f"   [SDE Tracking File]")
                n_cycle = len(nc.dimensions.get('cycle', []))
                n_block = len(nc.dimensions.get('block', []))
                n_psteps = len(nc.dimensions.get('psteps', []))
                n_var = len(nc.dimensions.get('var', []))
                n_pts = len(nc.dimensions.get('pts', [])) if 'pts' in nc.dimensions else 0
                n_ens = len(nc.dimensions.get('ens', []))
                print(f"   Dimensions -> Cycles: {n_cycle}, Blocks: {n_block}, SDE Steps: {n_psteps}, Vars: {n_var}, Ens: {n_ens}, Pts: {n_pts}")
                
                print(f"   {'Variable':<25} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Status':<10}")
                print("   " + "-"*70)
                for vname in nc.variables:
                    if vname == 'var_names': continue
                    data = nc.variables[vname][:]
                    stats = get_stats(data)
                    if not stats['valid']:
                        print(f"   {vname:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | NaN/Inf! ({stats['n_nan']} NaNs, {stats['n_inf']} Infs)")
                    else:
                        print(f"   {vname:<25} | {stats['min']:<10.2e} | {stats['max']:<10.2e} | {stats['mean']:<10.2e} | OK")
                return

            # Check Dimensions (Standard DA output)
            if 'lat' in nc.dimensions and 'lon' in nc.dimensions:
                nlat = len(nc.dimensions['lat'])
                nlon = len(nc.dimensions['lon'])
                print(f"   Grid: {nlat}x{nlon}")
            
            # Identify Variables (looking for pairs like xb_mean_VAR_levL and xa_mean_VAR_levL)
            vars_found = {} # (var, lev) -> {xb, xa, truth}
            
            for vname in nc.variables:
                parts = vname.split('_')
                if len(parts) >= 4 and parts[1] == 'mean':
                    prefix = parts[0] # xb or xa
                    var_name = parts[2]
                    lev_tag = parts[3]
                    key = (var_name, lev_tag)
                    if key not in vars_found: vars_found[key] = {}
                    vars_found[key][prefix] = vname
                elif parts[0] in ['truth', 'obs']:
                    var_name = parts[1]
                    lev_tag = parts[2]
                    key = (var_name, lev_tag)
                    if key not in vars_found: vars_found[key] = {}
                    vars_found[key][parts[0]] = vname

            if not vars_found:
                # Fallback: check for raw state variables (e.g. UG0, UG1...) in Free Run files
                # If this is a free run file, we try to find the reference solution
                is_free_run = "free_run_" in os.path.basename(filepath)
                if is_free_run:
                     # Attempt to construct reference path
                     # ../free_run/free_run_X.nc  ->  ../snapshots/reference_solution_X.nc
                     dirname = os.path.dirname(filepath)
                     parent = os.path.dirname(dirname)
                     basename = os.path.basename(filepath)
                     # assume format free_run_X.nc
                     cycle_idx = basename.split('_')[-1].split('.')[0]
                     ref_path = os.path.join(parent, "snapshots", f"reference_solution_{cycle_idx}.nc")
                     
                     if os.path.exists(ref_path):
                         print(f"   Comparison: vs {os.path.basename(ref_path)}")
                         try:
                             with Dataset(ref_path, 'r') as nc_ref:
                                 # List variables common to both
                                 print(f"   {'Variable':<10} {'Level':<6} | {'Incr(RMS)':<10} {'Incr(Max)':<10} | {'Err(RMS)':<10} | {'Status':<10}")
                                 print("   " + "-"*75)
                                 
                                 # We just look for standard vars: UG0, UG1, VG0, VG1, etc.
                                 std_vars = ['UG0', 'UG1', 'VG0', 'VG1', 'TG0', 'TG1', 'PSG0', 'PSG1', 'TRG0', 'TRG1']
                                 # Also wind if present?
                                 
                                 for var in std_vars:
                                     if var in nc.variables and var in nc_ref.variables:
                                         # Get data
                                         res_val = nc.variables[var][:]
                                         ref_val = nc_ref.variables[var][:]
                                         
                                         # Compute Diff (Error)
                                         diff = res_val - ref_val
                                         s_diff = get_stats(diff)
                                         
                                         err_rms = f"{s_diff['rms']:.2e}" if s_diff['valid'] else "NaN"
                                         
                                         # For free run, Incr is N/A (no analysis update)
                                         # OR we could show change from previous step? No, just show Error.
                                         incr_rms, incr_max = "N/A", "N/A"
                                         
                                         # Find level if applicable
                                         lev_str = "lev?"
                                         if res_val.ndim == 3: # (lev, lat, lon)
                                             # Just loop levels? Or average?
                                             # Original script showed per level.
                                             # Let's show RMSE per level
                                             nlev = res_val.shape[0]
                                             for l in range(nlev):
                                                 diff_l = diff[l]
                                                 s_l = get_stats(diff_l)
                                                 err_l = f"{s_l['rms']:.2e}"
                                                 print(f"   {var:<10} lev{l:<3} | {incr_rms:<10} {incr_max:<10} | {err_l:<10} | {'OK':<10}")
                                         elif res_val.ndim == 4: # (1, lev, lat, lon) or (lev, lat, lon, ?)
                                              # Humidity TRG is (1, 8, 32, 64)
                                              if var.startswith('TRG'):
                                                 nlev = res_val.shape[1]
                                                 for l in range(nlev):
                                                     diff_l = diff[0, l]
                                                     s_l = get_stats(diff_l)
                                                     err_l = f"{s_l['rms']:.2e}"
                                                     print(f"   {var:<10} lev{l:<3} | {incr_rms:<10} {incr_max:<10} | {err_l:<10} | {'OK':<10}")
                                         else:
                                             # 2D var like PSG
                                             print(f"   {var:<10} {'sfc':<6} | {incr_rms:<10} {incr_max:<10} | {err_rms:<10} | {'OK':<10}")
                                             
                                 return # Done with free run file
                         except Exception as e:
                             print(f"   Failed to open reference: {e}")
                     else:
                         print(f"   Reference file not found: {ref_path}")
                         return
                
                print("   No standard DA variables found (xb_mean_*, xa_mean_*).")
                return

            print(f"   {'Variable':<10} {'Level':<6} | {'Mean(XB)':<10} {'Mean(XA)':<10} | {'Incr(RMS)':<10} {'Innov(RMS)':<11} | {'Err(RMS)':<10} | {'Status':<10}")
            print("   " + "-"*96)

            sorted_keys = sorted(vars_found.keys(), key=lambda x: (x[1], x[0]))
            for var, lev in sorted_keys:
                vinfo = vars_found[(var, lev)]
                xb_name, xa_name, tr_name, obs_name = vinfo.get('xb'), vinfo.get('xa'), vinfo.get('truth'), vinfo.get('obs')
                
                status, incr_rms, innov_rms, err_rms = "SKIP", "N/A", "N/A", "N/A"
                mean_xb, mean_xa = "N/A", "N/A"
                
                if xb_name and xa_name:
                    try:
                        xb = nc.variables[xb_name][:]
                        xa = nc.variables[xa_name][:]
                        s_xb, s_xa = get_stats(xb), get_stats(xa)
                        
                        if not s_xb['valid'] or not s_xa['valid']:
                            status = "NaN/Inf!"
                        else:
                            mean_xb = f"{s_xb['mean']:.2e}"
                            mean_xa = f"{s_xa['mean']:.2e}"
                            
                            incr = xa - xb
                            s_incr = get_stats(incr)
                            incr_rms = f"{s_incr['rms']:.2e}"
                            status = "No Update" if s_incr['abs_max'] == 0.0 else "OK"
                            
                            if obs_name in nc.variables:
                                obs = nc.variables[obs_name][:]
                                mask = np.isfinite(obs)
                                if np.any(mask):
                                    innov = obs[mask] - xb[mask]
                                    innov_rms = f"{np.sqrt(np.mean(innov**2)):.2e}"
                            
                            if tr_name:
                                tr = nc.variables[tr_name][:]
                                s_err = get_stats(xa - tr)
                                err_rms = f"{s_err['rms']:.2e}" if s_err['valid'] else "NaN(Tr)"
                    except Exception:
                        status = "Error"
                print(f"   {var:<10} {lev:<6} | {mean_xb:<10} {mean_xa:<10} | {incr_rms:<10} {innov_rms:<11} | {err_rms:<10} | {status:<10}")
    except Exception as e:
        print(f"   Failed to open/read: {e}")

def main():
    all_files = []
    # Identify labels from paths
    labeled_files = []
    
    for path in PATHS_TO_INSPECT:
        if os.path.isfile(path):
            labeled_files.append((path, path))
        elif os.path.isdir(path):
            files = glob.glob(os.path.join(path, "unified_cycle*.nc"))
            sde_files = glob.glob(os.path.join(path, "sde_tracking.nc"))
            if files:
                files.sort(key=lambda x: int(os.path.basename(x).split('cycle')[1].split('.')[0]) if 'cycle' in os.path.basename(x) else 0)
            files.extend(sde_files)
            if not files:
                files = glob.glob(os.path.join(path, "*.nc"))
                files.sort()
            for f in files:
                labeled_files.append((f, path)) # Label is directory
        else:
            files = glob.glob(path)
            files.sort()
            for f in files:
                labeled_files.append((f, path))

    if not labeled_files:
        print("No files discovered to inspect. Please check PATHS_TO_INSPECT.")
        return

    print(f"Discovered {len(labeled_files)} files.")
    for f, label in labeled_files:
        inspect_file(f, VERBOSE, label=label)

if __name__ == "__main__":
    main()
