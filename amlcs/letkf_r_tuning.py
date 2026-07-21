#!/usr/bin/env python3
"""
LETKF localization-radius (r) tuning sweep.

Sweeps a 2D grid of localization radius (r) x multiplicative inflation
(infla). Two subcommands:

  submit   Generate one runner CSV per (r, infla) cell from a fixed
           observation template (only `r`, `infla`, and `exp_settings` are
           overridden), submit one sbatch job per cell running amlcs_da.py
           followed by an organize step, record a manifest, and submit a
           dependent collection job that runs once all sweeps finish.
           Use --max-concurrent N to cap simultaneous runs (batches are chained
           with Slurm afterany dependencies).

  organize Move one run's unified_cycle NetCDF files into
           <run_folder>/<name>/data/. Runs automatically at the end of each r's
           sbatch job (can also be run standalone).

  collect  Read the manifest, move unified_cycle NetCDF files into
           <run_folder>/<name>/data/ (postprocessing), compute the
           level-averaged analysis RMSE per variable for each run (the same way
           error_plots_dual_nc.py does it), reduce each per-cycle series to its
           average (mean over cycles) and lowest (min over cycles) value, and
           write a summary CSV plus print the best r.

The idea: hold the observation configuration fixed (e.g. everything observed
with arctan) and sweep r to find the value that minimizes analysis RMSE.

Examples
--------
    python letkf_r_tuning.py submit --template letkf_runner_wind_vars.csv --r-values 1,2,3,4 --infla-values 1.0,1.15,1.3,1.45,1.6 --exp-settings ../LETKF_tuning/t21_80_0.05_30/ --name wdg_wsg_tph_inflation --max-concurrent 5

    # --max-concurrent 5 (default 0 = all at once) chains batches with Slurm
    # afterany dependencies so the next batch starts only after the previous
    # batch finishes. One submit call still schedules the full grid plus collect.

    python letkf_r_tuning.py collect letkf_tuning_runs/arctan/manifest.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# netCDF4 is imported lazily inside the collect helpers so that `submit` works
# in environments where netCDF4 is not installed.

# Directory containing this script (amlcs/). All relative paths used by
# amlcs_da.py (e.g. ../runs/, ../LETKF_tuning/) are resolved against it.
SCRIPT_DIR = Path(__file__).resolve().parent

# Variables compared for RMSE (model variables), matching error_plots_dual_nc.py
VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]

# Default sbatch options, mirroring sbatch_runner.sh
DEFAULT_SBATCH = {
    "account": "chipilskigroup_q",
    "partition": "chipilskigroup_q",
    "time": "12:00:00",
    "mem": "12G",
}


###############################################################################
# ----------------------------- shared helpers -------------------------------
###############################################################################

def _levels_for_var(var):
    """Levels analysed for a given variable (matches error_plots_dual_nc.py)."""
    if "PSG" in var:
        return [0]
    if var.startswith("TRG"):
        return list(range(2, 8))
    return list(range(8))


def _compute_l2_error(field1, field2):
    """L2 (RMSE) error between two fields."""
    diff = np.asarray(field1) - np.asarray(field2)
    return float(np.sqrt(np.mean(diff ** 2)))


def _read_truth_field(nc_path, var, lev):
    """
    Read a truth field from a reference_solution NetCDF file.

    Reference files store raw variable names (TG1 as 3D, PSG1 as 2D, etc.).
    Mirrors the raw-variable branch of error_plots_dual_nc.py::_read_nc_field.
    """
    from netCDF4 import Dataset
    with Dataset(nc_path, "r") as nc:
        if var in nc.variables:
            data = nc.variables[var]
            if data.ndim == 3:      # (nlev, lat, lon)
                return data[lev, :, :]
            if data.ndim == 2:      # (lat, lon) e.g. PSG
                return data[:]
            if data.ndim == 4:      # (ntr, lev, lat, lon)
                return data[0, lev, :, :]
        raise KeyError(f"Truth field {var} (lev {lev}) not found in {nc_path}")


def _read_analysis_field(nc_path, var, lev):
    """Read the analysis-mean field (xa_mean_<var>_lev<lev>) from a cycle file."""
    from netCDF4 import Dataset
    with Dataset(nc_path, "r") as nc:
        field_name = f"xa_mean_{var}_lev{lev}"
        if field_name not in nc.variables:
            raise KeyError(f"{field_name} not found in {nc_path}")
        return nc.variables[field_name][:]


def _resolve(path_str):
    """Resolve a path relative to the script directory (amlcs/)."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (SCRIPT_DIR / p)
    return p.resolve()


def _cycle_data_dir(run_folder, campaign_name):
    """Target directory for unified_cycle files: <run_folder>/<name>/data/."""
    return Path(run_folder) / campaign_name / "data"


def _organize_run_cycle_files(run_folder, campaign_name):
    """
    Move unified_cycle*.nc from the run directory root into <name>/data/.

    amlcs_da writes cycles directly under the run folder; this step mirrors the
    layout used elsewhere (e.g. wdg_only/data/unified_cycle0.nc).
    """
    run_folder = Path(run_folder)
    dest = _cycle_data_dir(run_folder, campaign_name)
    dest.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in sorted(run_folder.glob("unified_cycle*.nc")):
        if src.parent != run_folder:
            continue
        target = dest / src.name
        if target.exists():
            continue
        shutil.move(str(src), str(target))
        moved += 1
    return dest, moved


def organize(args):
    """Move one run's unified_cycle files into <run_folder>/<name>/data/.

    Intended to run inside each r's sbatch job right after amlcs_da.py finishes,
    so subfolders are created per r as soon as that assimilation completes.
    """
    run_folder = Path(args.run_folder)
    dest, moved = _organize_run_cycle_files(run_folder, args.name)
    print(f"organize: moved {moved} unified_cycle file(s) -> {dest}")


###############################################################################
# -------------------------------- submit ------------------------------------
###############################################################################

def _read_config_value(exp_settings_dir, key):
    cfg = pd.read_csv(Path(exp_settings_dir) / "config.csv")
    return cfg[key].iloc[0]


def _infla_token(infla):
    """Inflation token used in run-folder names (matches amlcs_da.py)."""
    return int(round(100 * float(infla)))


def _run_folder_name(template_df, code_path):
    """Reproduce amlcs_da.py's method_path naming as a function of (r, infla)."""
    s = int(template_df["s"].iloc[0])
    method = str(template_df["method"].iloc[0]).strip()

    def _make(r, infla):
        return f"{code_path}_{method}_{int(r)}_{s}_{_infla_token(infla)}"

    return _make


def _parse_job_id(sbatch_stdout):
    """Extract the numeric job id from `sbatch` output."""
    # Typical output: "Submitted batch job 1234567"
    for token in sbatch_stdout.split():
        if token.isdigit():
            return token
    return None


def _batched(items, batch_size):
    """Split *items* into consecutive chunks; batch_size <= 0 yields one chunk."""
    if not batch_size or batch_size <= 0 or batch_size >= len(items):
        yield items
        return
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _submit_sweep_job(args, cfg_rel, run_folder, r, infla, prev_batch_job_ids):
    """Submit one sbatch job; optionally wait for the previous batch (afterany)."""
    ii = _infla_token(infla)
    organize_cmd = (f'./run_py.sh {Path(__file__).name} organize '
                    f'"{run_folder}" --name {args.name}')
    wrap = f'./run_py.sh amlcs_da.py {cfg_rel} && {organize_cmd}'
    cmd = [
        "sbatch",
        f"--account={args.account}",
        f"--partition={args.partition}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        "--nodes=1",
        "--ntasks-per-node=1",
        f"--job-name={args.name}_r{r}_i{ii}",
    ]
    if prev_batch_job_ids:
        cmd.append(f"--dependency=afterany:{':'.join(prev_batch_job_ids)}")
    cmd.append(f"--wrap={wrap}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(f"sbatch submission failed for r={r}, infla={infla}")
    return _parse_job_id(result.stdout)


def submit(args):
    template_path = _resolve(args.template)
    if not template_path.exists():
        sys.exit(f"Template runner CSV not found: {template_path}")

    exp_settings_resolved = _resolve(args.exp_settings)
    if not (exp_settings_resolved / "config.csv").exists():
        sys.exit(f"exp_settings/config.csv not found under: {exp_settings_resolved}")

    r_values = [int(v.strip()) for v in args.r_values.split(",") if v.strip() != ""]
    if not r_values:
        sys.exit("No r values provided (use --r-values 2,4,6,...).")

    infla_values = [float(v.strip()) for v in args.infla_values.split(",") if v.strip() != ""]
    if not infla_values:
        sys.exit("No inflation values provided (use --infla-values 1.0,1.15,...).")

    code_path = str(_read_config_value(exp_settings_resolved, "code_path"))
    M = int(_read_config_value(exp_settings_resolved, "M"))

    template_df = pd.read_csv(template_path)
    folder_for = _run_folder_name(template_df, code_path)

    # Campaign output layout: amlcs/<campaign_root>/<name>/
    campaign_root = getattr(args, "campaign_root", "letkf_tuning_runs")
    campaign_dir = SCRIPT_DIR / campaign_root / args.name
    configs_dir = campaign_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    runs_root = (SCRIPT_DIR / ".." / "runs").resolve()

    have_sbatch = shutil.which("sbatch") is not None
    if not have_sbatch:
        print("WARNING: `sbatch` not found on PATH. Configs and manifest will be "
              "generated, but no jobs will be submitted.")

    max_concurrent = int(getattr(args, "max_concurrent", 0) or 0)

    runs = []
    pending = []
    # 2D grid sweep: every (r, infla) cell is an independent run with its own
    # config and run folder (the inflation token in the folder name keeps
    # parallel runs from overwriting each other).
    for r in r_values:
        for infla in infla_values:
            ii = _infla_token(infla)
            df = template_df.copy()
            df.loc[:, "r"] = r
            df.loc[:, "infla"] = infla
            # exp_settings is stored as-is (relative to amlcs) so amlcs_da.py
            # resolves it the same way it resolves ../runs/.
            df.loc[:, "exp_settings"] = args.exp_settings

            cfg_path = configs_dir / f"r{r}_i{ii}.csv"
            df.to_csv(cfg_path, index=False)

            run_folder = runs_root / folder_for(r, infla)
            cfg_rel = os.path.relpath(cfg_path, SCRIPT_DIR)
            data_dir = run_folder / args.name / "data"

            pending.append({
                "r": r,
                "infla": infla,
                "config": str(cfg_path),
                "config_rel": cfg_rel,
                "run_folder": str(run_folder),
                "data_dir": str(data_dir),
            })

    run_job_ids = []
    if have_sbatch and pending:
        batches = list(_batched(pending, max_concurrent))
        if max_concurrent > 0 and len(batches) > 1:
            print(f"Submitting {len(pending)} run(s) in {len(batches)} batch(es) "
                  f"of up to {max_concurrent} concurrent job(s).")
        prev_batch_job_ids = []
        for batch_idx, batch in enumerate(batches, start=1):
            batch_job_ids = []
            for cell in batch:
                job_id = _submit_sweep_job(
                    args, cell["config_rel"], cell["run_folder"],
                    cell["r"], cell["infla"], prev_batch_job_ids,
                )
                cell["job_id"] = job_id
                if job_id:
                    run_job_ids.append(job_id)
                    batch_job_ids.append(job_id)
                print(f"  r={cell['r']} infla={cell['infla']}: config={cell['config_rel']} "
                      f"-> run_folder={cell['run_folder']} job_id={job_id}")
            if max_concurrent > 0 and len(batches) > 1:
                print(f"  Batch {batch_idx}/{len(batches)}: "
                      f"{len(batch_job_ids)} job(s) submitted.")
            prev_batch_job_ids = batch_job_ids
        runs = pending
    else:
        for cell in pending:
            cell["job_id"] = None
            runs.append(cell)
            print(f"  r={cell['r']} infla={cell['infla']}: config={cell['config_rel']} "
                  f"-> run_folder={cell['run_folder']} job_id=None")

    manifest = {
        "name": args.name,
        "template": str(template_path),
        "exp_settings": args.exp_settings,
        "exp_settings_resolved": str(exp_settings_resolved),
        "snapshots_dir": str(exp_settings_resolved / "snapshots"),
        "code_path": code_path,
        "M": M,
        "vars": VARS,
        "r_values": r_values,
        "infla_values": infla_values,
        "max_concurrent": max_concurrent,
        "runs": runs,
    }
    manifest_path = campaign_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")

    # Dependent collection job: runs once every sweep job completes OK.
    if have_sbatch and run_job_ids:
        manifest_rel = os.path.relpath(manifest_path, SCRIPT_DIR)
        dep = ":".join(run_job_ids)
        wrap = f'./run_py.sh {Path(__file__).name} collect {manifest_rel}'
        cmd = [
            "sbatch",
            f"--account={args.account}",
            f"--partition={args.partition}",
            "--time=01:00:00",
            f"--mem={args.mem}",
            "--nodes=1",
            "--ntasks-per-node=1",
            f"--job-name=collect_{args.name}",
            f"--dependency=afterok:{dep}",
            f"--wrap={wrap}",
        ]
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            print("WARNING: failed to submit dependent collection job. Run "
                  f"`python {Path(__file__).name} collect {manifest_rel}` manually "
                  "after the sweeps finish.", file=sys.stderr)
        else:
            print(f"Collection job submitted (afterok:{dep}).")
    else:
        manifest_rel = os.path.relpath(manifest_path, SCRIPT_DIR)
        print("\nNo collection job submitted. After the runs finish, run:")
        print(f"  python {Path(__file__).name} collect {manifest_rel}")


###############################################################################
# -------------------------------- collect -----------------------------------
###############################################################################

def _available_cycles(data_dir, M):
    """Cycle indices for which a unified_cycle file exists (0..M-1 superset)."""
    data_dir = Path(data_dir)
    cycles = []
    for k in range(M):
        if (data_dir / f"unified_cycle{k}.nc").exists():
            cycles.append(k)
    return cycles


def _level_averaged_analysis_series(data_dir, snapshots_dir, var, cycles):
    """
    Per-cycle level-averaged analysis RMSE series for one variable.

    For each level: L2(xa_mean, truth); average across levels per cycle.
    Mirrors the level-averaged path of error_plots_dual_nc.py.
    """
    levels = _levels_for_var(var)
    data_dir = Path(data_dir)
    snapshots_dir = Path(snapshots_dir)

    per_level = []
    for lev in levels:
        errs = []
        for k in cycles:
            cycle_file = data_dir / f"unified_cycle{k}.nc"
            truth_file = snapshots_dir / f"reference_solution_{k}.nc"
            try:
                ana = _read_analysis_field(cycle_file, var, lev)
                truth = _read_truth_field(truth_file, var, lev)
                errs.append(_compute_l2_error(ana, truth))
            except (FileNotFoundError, KeyError, OSError):
                errs.append(np.nan)
        per_level.append(np.array(errs, dtype=float))

    if not per_level:
        return np.array([])

    return np.nanmean(np.vstack(per_level), axis=0)


def collect(args):
    manifest_path = _resolve(args.manifest)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    snapshots_dir = manifest["snapshots_dir"]
    M = int(manifest["M"])
    vars_list = manifest.get("vars", VARS)
    campaign_name = manifest["name"]
    campaign_dir = manifest_path.parent

    rows = []
    for run in sorted(manifest["runs"], key=lambda d: (d["r"], d.get("infla", 1.0))):
        r = run["r"]
        infla = run.get("infla", 1.0)
        run_folder = run["run_folder"]
        data_dir, moved = _organize_run_cycle_files(run_folder, campaign_name)
        if moved:
            print(f"r={r} infla={infla}: moved {moved} unified_cycle file(s) -> {data_dir}")
        cycles = _available_cycles(data_dir, M)

        row = {"r": r, "infla": infla, "run_folder": run_folder,
               "data_dir": str(data_dir), "n_cycles": len(cycles)}
        if not cycles:
            print(f"r={r} infla={infla}: no unified_cycle*.nc files found in {data_dir} (skipping).")
            for var in vars_list:
                row[f"{var}_avg"] = np.nan
                row[f"{var}_min"] = np.nan
            row["overall_avg"] = np.nan
            rows.append(row)
            continue

        var_avgs = []
        for var in vars_list:
            series = _level_averaged_analysis_series(data_dir, snapshots_dir, var, cycles)
            if series.size == 0 or np.all(np.isnan(series)):
                avg = np.nan
                lowest = np.nan
            else:
                avg = float(np.nanmean(series))
                lowest = float(np.nanmin(series))
            row[f"{var}_avg"] = avg
            row[f"{var}_min"] = lowest
            if not np.isnan(avg):
                var_avgs.append(avg)

        row["overall_avg"] = float(np.mean(var_avgs)) if var_avgs else np.nan
        rows.append(row)
        print(f"r={r} infla={infla}: cycles={len(cycles)} overall_avg={row['overall_avg']:.6g}")

    df = pd.DataFrame(rows)
    summary_path = campaign_dir / "rmse_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary written: {summary_path}")

    # Best (r, infla) cell by per-variable wins (scale-invariant): the cell that
    # achieves the lowest average RMSE for the greatest number of variables.
    # Ties are broken by mean rank across variables, then smaller r, then
    # smaller infla.
    def _cell(run_row):
        return (int(run_row["r"]), float(run_row["infla"]))

    cells = [_cell(r) for _, r in df.iterrows()]
    wins = {c: 0 for c in cells}
    rank_sums = {c: 0.0 for c in cells}
    rank_counts = {c: 0 for c in cells}
    n_scored = 0
    for var in vars_list:
        col = f"{var}_avg"
        if col not in df.columns:
            continue
        sub = df[["r", "infla", col]].dropna(subset=[col])
        if sub.empty:
            continue
        n_scored += 1
        ranks = sub[col].rank(method="min", ascending=True)
        for (_, srow), rk in zip(sub.iterrows(), ranks):
            c = (int(srow["r"]), float(srow["infla"]))
            rank_sums[c] += float(rk)
            rank_counts[c] += 1
        best_row = sub.loc[sub[col].idxmin()]
        wins[(int(best_row["r"]), float(best_row["infla"]))] += 1

    if n_scored > 0 and any(wins.values()):
        mean_rank = {c: (rank_sums[c] / rank_counts[c]) if rank_counts[c] else np.inf
                     for c in wins}
        best_cell = sorted(wins, key=lambda c: (-wins[c], mean_rank[c], c[0], c[1]))[0]
        br, bi = best_cell
        print(f"Best (r, infla) (most per-variable wins): r={br} infla={bi} "
              f"(wins {wins[best_cell]}/{n_scored} variables, mean rank {mean_rank[best_cell]:.3g})")
        print(f"  Summary CSV: {os.path.relpath(summary_path, SCRIPT_DIR)}")
    else:
        print("No valid runs to rank (no readable cycle files).")


###############################################################################
# --------------------------------- CLI --------------------------------------
###############################################################################

def build_parser(default_template=None,
                 default_infla="1.0,1.15,1.3,1.45,1.6",
                 default_r=None,
                 default_name="arctan",
                 campaign_root="letkf_tuning_runs",
                 default_exp_settings="../LETKF_tuning/t21_80_0.05_30/"):
    """Build the CLI parser.

    Defaults are parameterized so a sibling script (e.g. reversesde_tuning.py)
    can reuse the same submit/organize/collect engine with its own template,
    inflation grid, localization values, campaign name, and campaign output
    root. When `default_r` is set, `--r-values` becomes optional (used by the
    score filter, where r does not affect localization and is held fixed).
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sub = sub.add_parser("submit", help="Generate configs and submit the (r x infla) grid sweep.")
    p_sub.add_argument("--template", required=(default_template is None), default=default_template,
                       help="Path to a template runner CSV (fixed obs config).")
    p_sub.add_argument("--r-values", required=(default_r is None), default=default_r,
                       help="Comma-separated list of localization radii, e.g. 1,2,3,4,5.")
    p_sub.add_argument("--infla-values", default=default_infla,
                       help="Comma-separated list of inflation factors, e.g. 1.0,1.15,1.3,1.45,1.6.")
    p_sub.add_argument("--exp-settings", default=default_exp_settings,
                       help="exp_settings folder (truth/free_run source), relative to amlcs/.")
    p_sub.add_argument("--name", default=default_name,
                       help=f"Campaign name; outputs go to {campaign_root}/<name>/.")
    p_sub.add_argument("--account", default=DEFAULT_SBATCH["account"])
    p_sub.add_argument("--partition", default=DEFAULT_SBATCH["partition"])
    p_sub.add_argument("--time", default=DEFAULT_SBATCH["time"])
    p_sub.add_argument("--mem", default=DEFAULT_SBATCH["mem"])
    p_sub.add_argument("--max-concurrent", type=int, default=0,
                       help="Cap simultaneous sbatch jobs (0 = no limit). Later batches "
                            "wait for the previous batch via Slurm afterany dependencies.")
    p_sub.set_defaults(func=submit, campaign_root=campaign_root)

    p_org = sub.add_parser("organize",
                           help="Move a single run's unified_cycle NetCDFs into <run_folder>/<name>/data/.")
    p_org.add_argument("run_folder", help="Run directory containing unified_cycle*.nc files.")
    p_org.add_argument("--name", required=True, help="Campaign name (subfolder under the run directory).")
    p_org.set_defaults(func=organize)

    p_col = sub.add_parser("collect",
                           help="Organize cycle NetCDFs into <run>/<name>/data and compute RMSE summary.")
    p_col.add_argument("manifest", help="Path to manifest.json produced by submit.")
    p_col.set_defaults(func=collect)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
