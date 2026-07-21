"""
Shapiro-Wilk gaussianity test of the prior (background) ensemble.

For each prognostic variable and each model level, a Shapiro-Wilk test is
performed on the 80-member ensemble sample at every horizontal grid point
(64 x 32). The p-values are then averaged over the grid to give one value
per level, and finally averaged over levels. Higher mean p-value = more
Gaussian; lower = more non-Gaussian.

Uses a single time step (the "1" time-level fields stored in the files).
"""

import glob
import os

import numpy as np
from scipy.io import netcdf_file
from scipy.stats import shapiro

ENS_DIR = (r"C:\Users\jack_schwartz\OneDrive\git_repos\SPEEDY-f77"
           r"\LETKF_tuning\t21_80_0.05_30\ensemble_0")

# variable name in file -> (label, is_3d)
VARIABLES = {
    "UG1":  ("U (zonal wind)",      True),
    "VG1":  ("V (meridional wind)", True),
    "TG1":  ("T (temperature)",     True),
    "TRG1": ("q (humidity tracer)", True),
    "PSG1": ("Ps (surface press.)", False),
}


def load_ensemble():
    """Return dict varname -> array (n_members, nlev, nlat, nlon) or
    (n_members, nlat, nlon) for surface fields."""
    files = sorted(
        glob.glob(os.path.join(ENS_DIR, "ensemble_member_*.nc")),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    print(f"Found {len(files)} ensemble members in {ENS_DIR}")
    data = {v: [] for v in VARIABLES}
    for path in files:
        f = netcdf_file(path, mmap=False)
        for v in VARIABLES:
            arr = f.variables[v][:].astype(np.float64)
            arr = np.squeeze(arr)  # drop the recname dim of TRG
            data[v].append(arr)
        f.close()
    return {v: np.stack(a) for v, a in data.items()}


def shapiro_pvalues_per_level(field):
    """field: (n_members, nlat, nlon) -> mean p-value over grid points."""
    n_mem, nlat, nlon = field.shape
    samples = field.reshape(n_mem, nlat * nlon)
    pvals = np.empty(nlat * nlon)
    for i in range(nlat * nlon):
        s = samples[:, i]
        # A numerically constant sample makes the test undefined
        if np.ptp(s) == 0 or np.std(s) < 1e-30 * max(1.0, abs(np.mean(s))):
            pvals[i] = np.nan
            continue
        pvals[i] = shapiro(s).pvalue
    if np.all(np.isnan(pvals)):
        return np.nan  # entire level constant across members (e.g. q aloft)
    return np.nanmean(pvals)


def main():
    data = load_ensemble()
    n_mem = next(iter(data.values())).shape[0]
    print(f"Ensemble size per test: {n_mem} members\n")

    results = {}
    for v, (label, is_3d) in VARIABLES.items():
        field = data[v]
        if is_3d:
            nlev = field.shape[1]
            level_p = np.array(
                [shapiro_pvalues_per_level(field[:, k]) for k in range(nlev)]
            )
        else:
            level_p = np.array([shapiro_pvalues_per_level(field)])
        results[v] = (label, level_p)

    print("Grid-mean Shapiro-Wilk p-values (n = %d members per test)" % n_mem)
    print("Level 0 = model top, level 7 = surface (sigma levels)\n")

    header = "level    " + "".join(f"{v:>10s}" for v in VARIABLES)
    print(header)
    max_lev = max(len(p) for _, p in results.values())
    for k in range(max_lev):
        row = f"{k:>5d}    "
        for v in VARIABLES:
            _, p = results[v]
            row += f"{p[k]:>10.4f}" if k < len(p) else " " * 10
        print(row)

    print("-" * len(header))
    row = "lev-avg  "
    for v in VARIABLES:
        _, p = results[v]
        row += f"{np.nanmean(p):>10.4f}"
    print(row)
    print("(levels with a constant field across members, e.g. humidity in "
          "the top layers, are excluded from the average)")

    print("\nLevel-averaged p-values (higher = more Gaussian):")
    ranked = sorted(results.items(), key=lambda kv: np.nanmean(kv[1][1]))
    for v, (label, p) in ranked:
        print(f"  {label:<22s} ({v:>4s}): {np.nanmean(p):.4f}")
    print(f"\nMost non-Gaussian prior: {ranked[0][1][0]}")


if __name__ == "__main__":
    main()
