#!/usr/bin/env python3
"""
Validate unified_cycle*.nc outputs for a LETKF tuning campaign.

Scans every run listed in letkf_tuning_runs/<campaign>/manifest.json (or LETKF
run folders under ../runs/ that contain <campaign>/data/). Reports truncated or
unreadable NetCDF files, NaNs/Infs in state fields, suspicious magnitudes, and
missing cycles.

Note: unified_cycle files are NetCDF (.nc), not CSV.

Examples
--------
    python check_unified_cycles.py arctan_inflation

    python check_unified_cycles.py arctan_inflation --verbose

    python check_unified_cycles.py arctan_inflation \\
        --manifest letkf_tuning_runs/arctan_inflation/manifest.json \\
        --runs-root ../runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = (SCRIPT_DIR / ".." / "runs").resolve()
DEFAULT_CAMPAIGN_ROOT = SCRIPT_DIR / "letkf_tuning_runs"

# t21 grid (lat x lon)
EXPECTED_LAT = 32
EXPECTED_LON = 64

# Generous physical / sanity bounds for SPEEDY fields (flag only, not hard fail).
# obs/sigma may be NaN away from stations — that is expected.
FIELD_BOUNDS = {
    "UG": (-250.0, 250.0),
    "VG": (-250.0, 250.0),
    "TG": (80.0, 400.0),
    "TRG": (-1.0e4, 1.0e4),
    "PSG": (1.0e4, 1.2e5),
    "WDG": (-4.0, 4.0),
    "WSG": (0.0, 200.0),
}

STATE_PREFIXES = ("xb_mean", "xa_mean", "truth", "noda")
MIN_FILE_BYTES = 10_240  # truncated NetCDF from I/O failures is often ~32 B


@dataclass
class Issue:
    severity: str  # ERROR | WARN
    run_label: str
    path: str
    message: str


@dataclass
class RunReport:
    label: str
    run_folder: str
    data_dir: str
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


def _var_family(var_name: str) -> str | None:
    for key in FIELD_BOUNDS:
        if var_name.startswith(key):
            return key
    return None


def _parse_cycle_index(path: Path) -> int:
    m = re.search(r"unified_cycle(\d+)\.nc$", path.name)
    if not m:
        raise ValueError(f"not a unified_cycle file: {path}")
    return int(m.group(1))


def _load_manifest(campaign: str, manifest_arg: str | None) -> dict | None:
    if manifest_arg:
        path = _resolve(manifest_arg)
    else:
        path = DEFAULT_CAMPAIGN_ROOT / campaign / "manifest.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_runs(campaign: str, manifest: dict | None, runs_root: Path) -> list[RunReport]:
    reports: list[RunReport] = []

    if manifest is not None:
        expected_m = int(manifest.get("M", 0))
        for run in manifest.get("runs", []):
            run_folder = Path(run["run_folder"])
            if "LETKF" not in run_folder.name.upper():
                continue
            data_dir = Path(run.get("data_dir", run_folder / campaign / "data"))
            if not data_dir.is_dir():
                data_dir = run_folder / campaign / "data"
            if not data_dir.is_dir():
                data_dir = run_folder
            label = f"r={run.get('r')} infla={run.get('infla', 1.0)}"
            reports.append(RunReport(
                label=label,
                run_folder=str(run_folder),
                data_dir=str(data_dir),
                r=run.get("r"),
                infla=run.get("infla", 1.0),
            ))
        if reports:
            return reports
        if expected_m:
            pass  # fall through to glob

    # Fallback: glob LETKF run folders with campaign subfolder.
    pattern = f"*_LETKF_*"
    for run_folder in sorted(runs_root.glob(pattern)):
        if not run_folder.is_dir():
            continue
        data_dir = run_folder / campaign / "data"
        if not data_dir.is_dir():
            data_dir = run_folder
        if not list(data_dir.glob("unified_cycle*.nc")):
            continue
        reports.append(RunReport(
            label=run_folder.name,
            run_folder=str(run_folder),
            data_dir=str(data_dir),
        ))
    return reports


def _group_da_variables(nc) -> dict[tuple[str, str], dict[str, str]]:
    """Map (var, lev) -> {prefix: netcdf_var_name}."""
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for vname in nc.variables:
        parts = vname.split("_")
        if len(parts) < 4:
            continue
        prefix, var_name, lev_tag = parts[0], parts[2], parts[3]
        if prefix in STATE_PREFIXES or prefix in ("obs", "sigma", "is_obs"):
            grouped.setdefault((var_name, lev_tag), {})[prefix] = vname
    return grouped


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
        "mean": float(np.nanmean(arr)) if finite.any() else np.nan,
        "std": float(np.nanstd(arr)) if finite.any() else np.nan,
    }


def _check_cycle_file(report: RunReport, nc_path: Path, expected_m: int | None) -> None:
    rel = str(nc_path)
    size = nc_path.stat().st_size
    if size < MIN_FILE_BYTES:
        report.issues.append(Issue(
            "ERROR", report.label, rel,
            f"file too small ({size} bytes) — likely truncated/corrupt",
        ))
        return

    try:
        from netCDF4 import Dataset
    except ImportError:
        sys.exit("netCDF4 is required: pip install netCDF4")

    try:
        with Dataset(nc_path, "r") as nc:
            if "lat" not in nc.dimensions or "lon" not in nc.dimensions:
                report.issues.append(Issue(
                    "ERROR", report.label, rel, "missing lat/lon dimensions",
                ))
                return
            nlat = len(nc.dimensions["lat"])
            nlon = len(nc.dimensions["lon"])
            if (nlat, nlon) != (EXPECTED_LAT, EXPECTED_LON):
                report.issues.append(Issue(
                    "WARN", report.label, rel,
                    f"unexpected grid {nlat}x{nlon} (expected {EXPECTED_LAT}x{EXPECTED_LON})",
                ))

            grouped = _group_da_variables(nc)
            if not grouped:
                report.issues.append(Issue(
                    "ERROR", report.label, rel, "no xb_mean/xa_mean/truth variables found",
                ))
                return

            n_state_bad = 0
            n_zero_incr = 0
            n_extreme = 0

            for (var_name, lev_tag), vmap in sorted(grouped.items()):
                for prefix in STATE_PREFIXES:
                    vname = vmap.get(prefix)
                    if not vname:
                        continue
                    data = np.asarray(nc.variables[vname][:], dtype=float)
                    if data.shape != (nlat, nlon):
                        report.issues.append(Issue(
                            "ERROR", report.label, rel,
                            f"{vname}: shape {data.shape}, expected ({nlat}, {nlon})",
                        ))
                        continue
                    stats = _finite_stats(data)
                    if stats["n_finite"] == 0:
                        report.issues.append(Issue(
                            "ERROR", report.label, rel,
                            f"{vname}: all NaN/Inf",
                        ))
                        n_state_bad += 1
                        continue
                    if stats["n_nan"] or stats["n_inf"]:
                        report.issues.append(Issue(
                            "ERROR", report.label, rel,
                            f"{vname}: {stats['n_nan']} NaN, {stats['n_inf']} Inf "
                            f"(of {stats['size']} cells)",
                        ))
                        n_state_bad += 1

                    family = _var_family(var_name)
                    if family and family in FIELD_BOUNDS:
                        lo, hi = FIELD_BOUNDS[family]
                        if stats["min"] < lo or stats["max"] > hi:
                            report.issues.append(Issue(
                                "WARN", report.label, rel,
                                f"{vname}: range [{stats['min']:.4g}, {stats['max']:.4g}] "
                                f"outside typical [{lo}, {hi}]",
                            ))
                            n_extreme += 1
                    if abs(stats["max"]) > 1e12 or abs(stats["min"]) > 1e12:
                        report.issues.append(Issue(
                            "WARN", report.label, rel,
                            f"{vname}: magnitude suspicious (min={stats['min']:.4g}, "
                            f"max={stats['max']:.4g})",
                        ))
                        n_extreme += 1

                xb_name = vmap.get("xb_mean")
                xa_name = vmap.get("xa_mean")
                if xb_name and xa_name:
                    xb = np.asarray(nc.variables[xb_name][:], dtype=float)
                    xa = np.asarray(nc.variables[xa_name][:], dtype=float)
                    if np.isfinite(xb).any() and np.isfinite(xa).any():
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
                        obs_at_stations = obs[observed]
                        if not np.isfinite(obs_at_stations).all():
                            report.issues.append(Issue(
                                "ERROR", report.label, rel,
                                f"{obs_name}: NaN/Inf at observed grid points",
                            ))
                    if (~observed).any():
                        away = obs[~observed]
                        n_bad_away = int(np.isfinite(away).sum())
                        if n_bad_away > 0 and n_bad_away > 0.01 * away.size:
                            report.issues.append(Issue(
                                "WARN", report.label, rel,
                                f"{obs_name}: {n_bad_away} finite values where is_obs=0 "
                                "(usually NaN away from stations)",
                            ))

            if n_zero_incr == len(grouped) and len(grouped) > 0:
                report.issues.append(Issue(
                    "WARN", report.label, rel,
                    "analysis identical to background for every field (zero increment)",
                ))

    except OSError as exc:
        report.issues.append(Issue(
            "ERROR", report.label, rel, f"cannot open NetCDF ({exc})",
        ))
    except Exception as exc:
        report.issues.append(Issue(
            "ERROR", report.label, rel, f"read failed: {exc}",
        ))


def _check_run(report: RunReport, expected_m: int | None) -> None:
    data_dir = Path(report.data_dir)
    if not data_dir.is_dir():
        report.issues.append(Issue(
            "ERROR", report.label, str(data_dir), "data directory not found",
        ))
        return

    cycle_files = sorted(data_dir.glob("unified_cycle*.nc"), key=_parse_cycle_index)
    if not cycle_files:
        report.issues.append(Issue(
            "ERROR", report.label, str(data_dir), "no unified_cycle*.nc files",
        ))
        return

    report.cycles_found = [_parse_cycle_index(p) for p in cycle_files]

    if expected_m is not None:
        expected = set(range(expected_m))
        found = set(report.cycles_found)
        missing = sorted(expected - found)
        if missing:
            report.issues.append(Issue(
                "ERROR", report.label, str(data_dir),
                f"missing cycles {missing} (expected 0..{expected_m - 1}, "
                f"found {len(found)})",
            ))
        extra = sorted(found - expected)
        if extra:
            report.issues.append(Issue(
                "WARN", report.label, str(data_dir),
                f"unexpected extra cycles {extra}",
            ))

    for nc_path in cycle_files:
        _check_cycle_file(report, nc_path, expected_m)


def _print_report(reports: list[RunReport], verbose: bool) -> None:
    n_err = sum(1 for r in reports for i in r.issues if i.severity == "ERROR")
    n_warn = sum(1 for r in reports for i in r.issues if i.severity == "WARN")

    print(f"\n{'=' * 72}")
    print(f"Runs checked: {len(reports)}  |  ERRORs: {n_err}  |  WARNs: {n_warn}")
    print(f"{'=' * 72}")

    for report in reports:
        status = "OK" if report.ok else "FAIL"
        cycles = (
            f"{min(report.cycles_found)}..{max(report.cycles_found)} "
            f"({len(report.cycles_found)} files)"
            if report.cycles_found else "no cycles"
        )
        print(f"\n[{status}] {report.label}")
        print(f"       {report.data_dir}")
        print(f"       cycles: {cycles}")
        if not report.issues:
            print("       no issues")
            continue
        for issue in report.issues:
            if not verbose and issue.severity == "WARN":
                continue
            print(f"       {issue.severity}: {issue.message}")
            if verbose:
                print(f"              ({Path(issue.path).name})")

    if n_warn and not verbose:
        print(f"\n({n_warn} WARN(s) hidden; use --verbose to show)")

    print()
    if n_err:
        print("RESULT: FAILED — at least one run has corrupt or incomplete cycle files.")
    elif n_warn:
        print("RESULT: PASSED with warnings — review WARN items with --verbose.")
    else:
        print("RESULT: PASSED — all checked runs look healthy.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "campaign",
        help="Campaign name (subfolder under each run, e.g. arctan_inflation).",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.json (default: letkf_tuning_runs/<campaign>/manifest.json).",
    )
    p.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help=f"Runs directory (default: {DEFAULT_RUNS_ROOT}).",
    )
    p.add_argument(
        "--expected-cycles",
        type=int,
        default=None,
        help="Expected number of cycles M (default: read from manifest).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every WARN and per-file detail.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    campaign = args.campaign
    runs_root = _resolve(args.runs_root)

    manifest = _load_manifest(campaign, args.manifest)
    expected_m = args.expected_cycles
    if expected_m is None and manifest is not None:
        expected_m = int(manifest.get("M", 0)) or None

    if manifest is None:
        print(f"Note: no manifest found for campaign '{campaign}'; "
              f"scanning {runs_root} for *_LETKF_*/{campaign}/data/")

    reports = _discover_runs(campaign, manifest, runs_root)
    if not reports:
        print(f"No LETKF runs found for campaign '{campaign}'.", file=sys.stderr)
        print("  Provide --manifest or ensure run folders exist under --runs-root.",
              file=sys.stderr)
        return 2

    print(f"Campaign: {campaign}")
    if manifest is not None:
        print(f"Manifest: {_resolve(args.manifest) if args.manifest else DEFAULT_CAMPAIGN_ROOT / campaign / 'manifest.json'}")
    print(f"Runs root: {runs_root}")
    if expected_m is not None:
        print(f"Expected cycles: 0..{expected_m - 1}")

    for report in reports:
        _check_run(report, expected_m)

    _print_report(reports, verbose=args.verbose)
    n_err = sum(1 for r in reports for i in r.issues if i.severity == "ERROR")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
