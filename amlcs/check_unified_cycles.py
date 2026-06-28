#!/usr/bin/env python3
"""
Validate unified_cycle*.nc outputs for a LETKF tuning campaign.

Scans runs from letkf_tuning_runs/<campaign>/manifest.json (or LETKF folders
under ../runs/). For each unified_cycle*.nc file found, checks that it opens,
has the expected variable layout, and that state fields are readable (no NaN/Inf
in xb_mean, xa_mean, truth, noda).

By default the script only validates files that exist — it does not complain
about missing cycle indices. Use --expect-cycles M (or read M from the manifest)
if you want warnings about incomplete runs.

Note: list_snapshots in the runner CSV controls separate xa{k}.nc / xb{k}.nc
snapshot files, not unified_cycle*.nc. Unified cycle files are written every
assimilation cycle in sequential_methods.perform_assimilation.

NetCDF layout (from sequential_methods.py):
    Dimensions: lat, lon  (t21: 32 x 64)
    Variables per observed block:  {prefix}_{VAR}_{levN}
        prefix in xb_mean, xa_mean, truth, noda, obs, sigma, is_obs, idx
        e.g. xa_mean_TG1_lev7, obs_UG1_lev7, is_obs_PSG1_lev0
    obs/sigma are NaN away from observing stations by design.

Examples
--------
    python check_unified_cycles.py arctan_inflation

    python check_unified_cycles.py arctan_inflation --verbose

    python check_unified_cycles.py arctan_inflation --verbose -o arctan_inflation_check.txt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = (SCRIPT_DIR / ".." / "runs").resolve()
DEFAULT_CAMPAIGN_ROOT = SCRIPT_DIR / "letkf_tuning_runs"

EXPECTED_LAT = 32
EXPECTED_LON = 64

# Multi-part prefixes — longest first when matching (see _parse_field_name).
FIELD_PREFIXES = (
    "xb_mean", "xa_mean", "truth", "noda", "obs", "sigma", "is_obs", "idx",
)

# Fields that must be finite everywhere (used for RMSE / plots).
REQUIRED_STATE_PREFIXES = ("xb_mean", "xa_mean", "truth", "noda")

# Optional sanity bounds (WARN only) on state fields.
FIELD_BOUNDS = {
    "UG": (-250.0, 250.0),
    "VG": (-250.0, 250.0),
    "TG": (80.0, 400.0),
    "PSG": (1.0e4, 1.2e5),
    "WDG": (-4.0, 4.0),
    "WSG": (0.0, 200.0),
}

MIN_FILE_BYTES = 10_240


@dataclass
class Issue:
    severity: str  # ERROR | WARN | INFO
    run_label: str
    where: str     # human-readable location
    check: str     # what was tested
    detail: str


@dataclass
class RunReport:
    label: str
    run_folder: str
    data_dir: str
    config_path: str | None = None
    r: int | None = None
    infla: float | None = None
    cycles_found: list[int] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "ERROR" for i in self.issues)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def _parse_cycle_index(path: Path) -> int:
    m = re.search(r"unified_cycle(\d+)\.nc$", path.name)
    if not m:
        raise ValueError(f"not a unified_cycle file: {path}")
    return int(m.group(1))


def _parse_field_name(vname: str) -> tuple[str, str, str] | None:
    """
    Parse NetCDF variable name into (prefix, var, lev_tag).

    Matches sequential_methods naming: xb_mean_TG1_lev7, is_obs_PSG1_lev0.
    """
    for prefix in sorted(FIELD_PREFIXES, key=len, reverse=True):
        head = prefix + "_"
        if not vname.startswith(head):
            continue
        rest = vname[len(head):]
        m = re.match(r"^(.+)_(lev\d+)$", rest)
        if m:
            return prefix, m.group(1), m.group(2)
    return None


def _group_da_variables(nc) -> dict[tuple[str, str], dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for vname in nc.variables:
        parsed = _parse_field_name(vname)
        if parsed is None:
            continue
        prefix, var_name, lev_tag = parsed
        grouped.setdefault((var_name, lev_tag), {})[prefix] = vname
    return grouped


def _expected_m_from_manifest(manifest: dict | None) -> int | None:
    """Assimilation cycle count M from manifest (exp_settings config.csv)."""
    if manifest is None:
        return None
    m = manifest.get("M")
    if m is not None:
        return int(m)
    return None


def _load_manifest(campaign: str, manifest_arg: str | None) -> dict | None:
    if manifest_arg:
        path = _resolve(manifest_arg)
    else:
        path = DEFAULT_CAMPAIGN_ROOT / campaign / "manifest.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_data_dir(run_folder: Path, campaign: str) -> Path:
    """Prefer <run>/<campaign>/data, then <run>/<campaign>, then run root."""
    for candidate in (
        run_folder / campaign / "data",
        run_folder / campaign,
        run_folder,
    ):
        if candidate.is_dir() and list(candidate.glob("unified_cycle*.nc")):
            return candidate
    return run_folder / campaign / "data"


def _discover_runs(campaign: str, manifest: dict | None, runs_root: Path) -> list[RunReport]:
    reports: list[RunReport] = []

    if manifest is not None:
        for run in manifest.get("runs", []):
            run_folder = Path(run["run_folder"])
            if "LETKF" not in run_folder.name.upper():
                continue
            config_path = run.get("config") or run.get("config_rel")
            config_resolved = _resolve(config_path) if config_path else None
            data_dir = _find_data_dir(run_folder, campaign)
            label = f"r={run.get('r')} infla={run.get('infla', 1.0)}"
            reports.append(RunReport(
                label=label,
                run_folder=str(run_folder),
                data_dir=str(data_dir),
                config_path=str(config_resolved) if config_resolved else None,
                r=run.get("r"),
                infla=run.get("infla", 1.0),
            ))
        if reports:
            return reports

    for run_folder in sorted(runs_root.glob("*_LETKF_*")):
        if not run_folder.is_dir():
            continue
        data_dir = _find_data_dir(run_folder, campaign)
        if not list(data_dir.glob("unified_cycle*.nc")):
            continue
        reports.append(RunReport(
            label=run_folder.name,
            run_folder=str(run_folder),
            data_dir=str(data_dir),
        ))
    return reports


def _add(report: RunReport, severity: str, where: str, check: str, detail: str) -> None:
    report.issues.append(Issue(severity, report.label, where, check, detail))


def _var_family(var_name: str) -> str | None:
    for key in FIELD_BOUNDS:
        if var_name.startswith(key):
            return key
    return None


def _finite_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    return {
        "size": arr.size,
        "n_finite": int(finite.sum()),
        "n_nan": int(np.isnan(arr).sum()),
        "n_inf": int(np.isinf(arr).sum()),
        "min": float(np.nanmin(arr)) if finite.any() else np.nan,
        "max": float(np.nanmax(arr)) if finite.any() else np.nan,
    }


def _check_cycle_file(report: RunReport, nc_path: Path, cycle_k: int) -> None:
    where_base = f"cycle {cycle_k} ({nc_path.name})"
    size = nc_path.stat().st_size
    if size < MIN_FILE_BYTES:
        _add(
            report, "ERROR", where_base, "file size",
            f"{size} bytes — likely truncated/corrupt NetCDF (healthy files are usually >> 10 KB)",
        )
        return

    try:
        from netCDF4 import Dataset
    except ImportError:
        sys.exit("netCDF4 is required: pip install netCDF4")

    try:
        with Dataset(nc_path, "r") as nc:
            if "lat" not in nc.dimensions or "lon" not in nc.dimensions:
                _add(report, "ERROR", where_base, "dimensions", "missing lat/lon dimensions")
                return

            nlat = len(nc.dimensions["lat"])
            nlon = len(nc.dimensions["lon"])
            if (nlat, nlon) != (EXPECTED_LAT, EXPECTED_LON):
                _add(
                    report, "WARN", where_base, "grid shape",
                    f"lat/lon = {nlat}x{nlon}, expected {EXPECTED_LAT}x{EXPECTED_LON} for t21",
                )

            grouped = _group_da_variables(nc)
            state_groups = {
                k: v for k, v in grouped.items()
                if any(p in v for p in REQUIRED_STATE_PREFIXES)
            }
            if not state_groups:
                vars_sample = ", ".join(sorted(nc.variables.keys())[:8])
                _add(
                    report, "ERROR", where_base, "variable layout",
                    f"no xb_mean/xa_mean/truth/noda fields found "
                    f"(expected names like xa_mean_TG1_lev7; file has: {vars_sample}...)",
                )
                return

            n_with_incr = 0
            n_zero_incr = 0

            for (var_name, lev_tag), vmap in sorted(state_groups.items()):
                field_where = f"{where_base} | {var_name} {lev_tag}"

                for prefix in REQUIRED_STATE_PREFIXES:
                    vname = vmap.get(prefix)
                    if not vname:
                        _add(
                            report, "WARN", field_where, f"missing {prefix}",
                            f"no variable {prefix}_{var_name}_{lev_tag} in this cycle file",
                        )
                        continue

                    data = np.asarray(nc.variables[vname][:], dtype=float)
                    if data.shape != (nlat, nlon):
                        _add(
                            report, "ERROR", field_where, f"{vname} shape",
                            f"shape {data.shape}, expected ({nlat}, {nlon})",
                        )
                        continue

                    stats = _finite_stats(data)
                    if stats["n_finite"] == 0:
                        _add(
                            report, "ERROR", field_where, f"{vname} finiteness",
                            "all values NaN or Inf",
                        )
                        continue
                    if stats["n_nan"] or stats["n_inf"]:
                        _add(
                            report, "ERROR", field_where, f"{vname} finiteness",
                            f"{stats['n_nan']} NaN and {stats['n_inf']} Inf "
                            f"of {stats['size']} grid cells",
                        )

                    family = _var_family(var_name)
                    if family and family in FIELD_BOUNDS:
                        lo, hi = FIELD_BOUNDS[family]
                        if stats["min"] < lo or stats["max"] > hi:
                            _add(
                                report, "WARN", field_where, f"{vname} range",
                                f"[{stats['min']:.4g}, {stats['max']:.4g}] outside "
                                f"typical [{lo}, {hi}] — may still be valid",
                            )

                xb_name = vmap.get("xb_mean")
                xa_name = vmap.get("xa_mean")
                if xb_name and xa_name:
                    xb = np.asarray(nc.variables[xb_name][:], dtype=float)
                    xa = np.asarray(nc.variables[xa_name][:], dtype=float)
                    if np.isfinite(xb).any() and np.isfinite(xa).any():
                        n_with_incr += 1
                        incr = xa - xb
                        if np.nanmax(np.abs(incr)) == 0.0:
                            n_zero_incr += 1

                obs_name = vmap.get("obs")
                iso_name = vmap.get("is_obs")
                if obs_name and iso_name:
                    obs = np.asarray(nc.variables[obs_name][:], dtype=float)
                    iso = np.asarray(nc.variables[iso_name][:], dtype=int)
                    observed = iso.astype(bool)
                    if observed.any():
                        at_obs = obs[observed]
                        n_bad = int((~np.isfinite(at_obs)).sum())
                        if n_bad:
                            _add(
                                report, "ERROR", field_where, f"{obs_name} at stations",
                                f"{n_bad} of {observed.sum()} observed cells are NaN/Inf "
                                f"(obs should be finite where is_obs=1)",
                            )
                    # NaN in obs away from stations is normal — not flagged.

            if n_with_incr > 0 and n_zero_incr == n_with_incr:
                _add(
                    report, "WARN", where_base, "increments",
                    "xa_mean identical to xb_mean for every field with both present "
                    "(zero analysis increment)",
                )

    except OSError as exc:
        _add(report, "ERROR", where_base, "open NetCDF", str(exc))
    except Exception as exc:
        _add(report, "ERROR", where_base, "read NetCDF", f"{type(exc).__name__}: {exc}")


def _check_run(report: RunReport, expected_m: int | None) -> None:
    data_dir = Path(report.data_dir)
    run_folder = Path(report.run_folder)

    if not data_dir.is_dir():
        # Also mention if files exist in run root instead
        root_cycles = list(run_folder.glob("unified_cycle*.nc"))
        hint = ""
        if root_cycles:
            hint = (f" — found {len(root_cycles)} unified_cycle file(s) in run root "
                    f"{run_folder} (not organized into {data_dir}); run organize or "
                    f"point checker at that folder")
        _add(
            report, "ERROR", f"run {report.label}", "data directory",
            f"{data_dir} not found{hint}",
        )
        return

    cycle_files = sorted(data_dir.glob("unified_cycle*.nc"), key=_parse_cycle_index)
    if not cycle_files:
        _add(
            report, "ERROR", f"run {report.label} | {data_dir}", "cycle files",
            "no unified_cycle*.nc files in data directory",
        )
        return

    report.cycles_found = [_parse_cycle_index(p) for p in cycle_files]
    found_set = set(report.cycles_found)

    if expected_m is not None:
        expected_set = set(range(expected_m))
        missing = sorted(expected_set - found_set)
        if missing:
            _add(
                report, "WARN",
                f"run {report.label} | {data_dir}",
                "incomplete cycle set",
                f"exp_settings M={expected_m} expects unified_cycle0..{expected_m - 1}; "
                f"missing {missing} ({len(found_set)}/{expected_m} present) — "
                f"run may have failed partway, or files not organized yet",
            )
        extra = sorted(found_set - expected_set)
        if extra:
            _add(
                report, "INFO",
                f"run {report.label} | {data_dir}",
                "extra cycles",
                f"found cycle indices {extra} beyond M={expected_m}",
            )

    for nc_path in cycle_files:
        _check_cycle_file(report, nc_path, _parse_cycle_index(nc_path))


def _format_issue(issue: Issue) -> str:
    return f"{issue.severity}: [{issue.where}] {issue.check} — {issue.detail}"


def _print_report(reports: list[RunReport], verbose: bool) -> None:
    n_err = sum(1 for r in reports for i in r.issues if i.severity == "ERROR")
    n_warn = sum(1 for r in reports for i in r.issues if i.severity == "WARN")

    print(f"\n{'=' * 72}")
    print(f"Runs checked: {len(reports)}  |  ERRORs: {n_err}  |  WARNs: {n_warn}")
    print(f"{'=' * 72}")

    for report in reports:
        status = "OK" if report.ok else "FAIL"
        cycles = (
            f"{report.cycles_found} ({len(report.cycles_found)} files)"
            if report.cycles_found else "none"
        )
        print(f"\n[{status}] {report.label}")
        print(f"       data: {report.data_dir}")
        print(f"       unified_cycle files found: {cycles}")
        if not report.issues:
            print("       no issues")
            continue
        for issue in report.issues:
            if issue.severity == "INFO" and not verbose:
                continue
            if issue.severity == "WARN" and not verbose:
                continue
            print(f"       {_format_issue(issue)}")

    hidden = sum(
        1 for r in reports for i in r.issues
        if i.severity in ("WARN", "INFO") and not verbose
    )
    if hidden:
        print(f"\n({hidden} WARN/INFO item(s) hidden; use --verbose)")

    print()
    if n_err:
        print("RESULT: FAILED — see ERROR lines above (run / cycle / field shown in brackets).")
    elif n_warn:
        print("RESULT: PASSED with warnings — use --verbose for details.")
    else:
        print("RESULT: PASSED — NetCDF files match expected layout and look readable.")


def _inspect_one_file(reports: list[RunReport]) -> None:
    """Print variable layout from the first cycle file found."""
    from netCDF4 import Dataset

    for report in reports:
        data_dir = Path(report.data_dir)
        files = sorted(data_dir.glob("unified_cycle*.nc"), key=_parse_cycle_index)
        if not files:
            continue
        nc_path = files[0]
        print(f"\nInspect: {nc_path}")
        print(f"  run: {report.label}")
        with Dataset(nc_path, "r") as nc:
            print("  dimensions:")
            for name, dim in nc.dimensions.items():
                print(f"    {name}: {len(dim)}")
            print("  variables (first 20):")
            for i, vname in enumerate(sorted(nc.variables.keys())):
                if i >= 20:
                    print(f"    ... and {len(nc.variables) - 20} more")
                    break
                var = nc.variables[vname]
                parsed = _parse_field_name(vname)
                tag = f" -> {parsed}" if parsed else ""
                print(f"    {vname}: shape={var.shape}{tag}")
        return
    print("No unified_cycle files found to inspect.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("campaign", help="Campaign subfolder name (e.g. arctan_inflation).")
    p.add_argument("--manifest", default=None,
                   help="manifest.json path (default: letkf_tuning_runs/<campaign>/manifest.json).")
    p.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT),
                   help=f"Runs directory (default: {DEFAULT_RUNS_ROOT}).")
    p.add_argument(
        "--expect-cycles", type=int, default=None,
        help="Warn if unified_cycle0..M-1 are not all present (default: read M "
             "from manifest; omit with --no-expect-cycles to skip completeness check).",
    )
    p.add_argument(
        "--no-expect-cycles", action="store_true",
        help="Only validate files that exist; do not warn about missing cycle indices.",
    )
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show WARN/INFO and per-field details.")
    p.add_argument("--inspect-one", action="store_true",
                   help="Print dimensions/variables from one sample file and exit.")
    p.add_argument("-o", "--output", default=None,
                   help="Write full report to this .txt file (path relative to amlcs/).")
    return p


def _run_checks(args, campaign: str, runs_root: Path, manifest: dict | None,
                reports: list[RunReport]) -> int:
    expected_m = None
    if not args.no_expect_cycles:
        expected_m = args.expect_cycles
        if expected_m is None:
            expected_m = _expected_m_from_manifest(manifest)

    print(f"Campaign: {campaign}")
    if manifest is not None:
        mpath = (_resolve(args.manifest) if args.manifest
                 else DEFAULT_CAMPAIGN_ROOT / campaign / "manifest.json")
        print(f"Manifest: {mpath}")
    print(f"Runs root: {runs_root}")
    if expected_m is not None:
        print(f"Completeness check: expect unified_cycle0..{expected_m - 1} per run (WARN if missing)")
    else:
        print("Completeness check: off (validating each file found only)")

    if args.inspect_one:
        _inspect_one_file(reports)
        return 0

    for report in reports:
        _check_run(report, expected_m=expected_m)

    _print_report(reports, verbose=args.verbose)
    n_err = sum(1 for r in reports for i in r.issues if i.severity == "ERROR")
    return 1 if n_err else 0


def main() -> int:
    args = build_parser().parse_args()
    campaign = args.campaign
    runs_root = _resolve(args.runs_root)

    manifest = _load_manifest(campaign, args.manifest)
    if manifest is None:
        print(f"Note: no manifest for '{campaign}'; scanning {runs_root}")

    reports = _discover_runs(campaign, manifest, runs_root)
    if not reports:
        print(f"No LETKF runs found for campaign '{campaign}'.", file=sys.stderr)
        return 2

    if args.output:
        out_path = _resolve(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                code = _run_checks(args, campaign, runs_root, manifest, reports)
        print(f"Report written to {out_path}")
        return code

    return _run_checks(args, campaign, runs_root, manifest, reports)


if __name__ == "__main__":
    sys.exit(main())
