# AMLCS Experiments: LETKF and ReverseSDE on SPEEDY

After SPEEDY is built (see the main [README](./README.md)), run data assimilation
experiments from the **`amlcs/`** directory at the repository root. This guide
explains how to set up runs, what the main configuration choices mean, and how
to turn outputs into figures.

All commands below assume your current working directory is **`amlcs/`** —
the data-assimilation workflow directory in your AMLCS checkout (for example
`~/AMLCS/amlcs` on a cluster login node).

Before running preprocessing or assimilation on the cluster, start an
interactive session and load the DA environment. From the repository root:

```bash
source start_da_session.sh
cd amlcs
```

That helper loads the compiler module, activates the `speedy_da_env` conda
environment, and sets `LD_LIBRARY_PATH` for the SPEEDY libraries.

The basic workflow is:

1. Choose an existing experiment input folder (`exp_settings`) or generate one
   with `amlcs_pre.py`: truth, ensemble initial conditions, NoDA baseline, and
   local SPEEDY support files.
2. Copy one runner CSV template and edit its single data row.
3. Launch `amlcs_da.py` through `./run_py.sh` so the run has a saved log.
4. Use the automatically created run folder under `../runs/` for diagnostics
   and post-processing.

---

## What you are actually doing

Every experiment follows the same **perfect-model twin** setup:

1. A **truth trajectory** is stored in `exp_settings/snapshots/`.
2. An **ensemble forecast** starts from perturbed initial conditions in
   `ensemble_0/`.
3. **Synthetic observations** are generated from the truth (with specified noise
   and, optionally, a nonlinear transform).
4. Each assimilation cycle: forecast → analyze → update ensemble → repeat.

You compare the analysis to truth and to a **no-data-assimilation (NoDA)**
free run in `free_run/`. The goal is usually to answer one of:

- Does the method beat NoDA?
- How does LETKF compare to the score filter (ReverseSDE)?
- Which localization radius or inflation works best?
- How do linear vs nonlinear vs wind observations change behaviour?

The runner CSV is a **single-row recipe** that tells `amlcs_da.py` which method
to use, which fields to observe, and how strong localization/inflation should be.

---

## Two assimilation methods (high level)

### LETKF (Local Ensemble Transform Kalman Filter)

Think of this as a **local Kalman update at every grid point**. Observations
within a square patch (controlled by **`r`**) influence the state at that point.
Inflation **`infla`** enters the solve as prior covariance inflation
`(k-1)I/ρ` (Szunyogh et al. 2008).

LETKF cares about:

- **`r`** — how far observations can reach (grid-index radius; see below)
- **`infla`** — how much ensemble spread is retained in the prior
- **`option_mask`** — whether U, V, T, etc. are updated together or separately
- **`wind_nonlinear_operator`** — whether wind direction/speed obs use a
  nonlinear U/V operator inside the filter

### ReverseSDE (ensemble score filter / EnSF)

Think of this as **drawing the posterior by reversing a stochastic differential
equation** in observation space. There is **no localization patch** in the same
sense as LETKF; the parameter **`r` in the CSV does not change the analysis**
(it only affects the run folder name).

ReverseSDE automatically:

1. Runs a pseudo-time SDE per observed block until convergence.
2. **Restores ensemble spread** to the prior standard deviation.
3. Applies a final multiplicative factor **`infla`** (`x̄ + infla·X′`).

So for ReverseSDE you usually tune **`infla` only**, not `r`.

During a ReverseSDE run the code also writes **`sde_tracking.nc`**, which records
how each ensemble member moves through pseudo-time. That file feeds the
**spaghetti plots** (Step 5 below).

---

## Step 1: Pick the experiment inputs

Point your runner CSV at an **`exp_settings`** directory (relative to `amlcs/`).
This directory is the fixed input dataset for the experiment. It is not the
same thing as the output run folder that `amlcs_da.py` creates later.

It must already contain truth, ensemble, and model support files.

Typical folders:

| Folder | Typical use |
|--------|-------------|
| `../LETKF_tuning/t21_80_0.05_30/` | 80 members, 30 cycles, current tuning campaigns |
| `../ENSF_gaussian_check/t21_50_0.05_20/` | Smaller ensemble, older linear/nonlinear tests |

Required contents:

```
exp_settings/
  config.csv          # Nens, M, resolution, perturbation size
  snapshots/          # reference_solution_0.nc ... reference_solution_{M-1}.nc
  ensemble_0/         # ensemble_member_0.nc ... + fort_*.3 restarts
  free_run/           # free_run_0.nc ... (NoDA baseline for plots)
  initial_condition/
  model_local/
  source_local/
```

The **`config.csv`** row defines ensemble size and cycle count. The runner CSV
does not repeat those; it only selects method, observations, and tuning knobs.

Before starting a new run, check two things:

- `config.csv` has the intended ensemble size (`Nens`) and number of cycles
  (`M`).
- `free_run/` exists if you plan to compare against the NoDA baseline in plots.

### Optional: generate inputs with `amlcs_pre.py`

If you do not already have an `exp_settings` folder, generate one from a
preprocessing CSV. Start from an existing template such as
`amlcs_pre_letkf_30.csv`, copy it, and edit the single data row.

Important columns:

| Column | Role |
|--------|------|
| `Nens` | Ensemble size |
| `M` | Number of assimilation cycles to prepare |
| `res_name` | Resolution name, normally `t21` here |
| `per` | Size of the initial ensemble perturbation |
| `folder_prep` | Parent folder where inputs are written |
| `code` | Optional explicit folder name; leave empty for the default name |
| `syn_tests` | `True` to create synthetic truth, ensemble, and free run inputs |

Run preprocessing from `amlcs/`:

```bash
cp amlcs_pre_letkf_30.csv my_pre.csv
# edit my_pre.csv
./run_py.sh amlcs_pre.py my_pre.csv
```

With `code` left empty, `amlcs_pre.py` writes:

```text
<folder_prep>/<res_name>_<Nens>_<per>_<M>/
```

For example, `folder_prep=../LETKF_tuning`, `res_name=t21`, `Nens=80`,
`per=0.05`, and `M=30` creates:

```text
../LETKF_tuning/t21_80_0.05_30/
```

That generated folder is the `exp_settings` path to put in your DA runner CSV.

---

## Step 2: Choose the kind of experiment

Start from a **template CSV** in `amlcs/`, copy it to a new name, and edit the
single data row. Treat this CSV as the run recipe. It tells `amlcs_da.py`:

- which method to run (`method`)
- which input folder to use (`exp_settings`)
- which quantities are observed (`obs_plc`) and with what noise (`err_obs`)
- which tuning values to use (`r`, `s`, `infla`, `option_mask`, etc.)

For example:

```bash
cp letkf_runner_normal.csv my_letkf_test.csv
```

Then edit `my_letkf_test.csv` before running it. The templates differ mainly
in **what you observe** and **whether observations are linear or nonlinear**.

### Linear observations (simplest baseline)

**Question:** Can the method assimilate standard model fields when the
observation operator is just "read the value at the station"?

- **LETKF:** copy `letkf_runner_normal.csv`
- **ReverseSDE:** copy `ensf_runner_linear_m1.csv`

Set `nonlinear_obs=False`. Every observed field in `obs_plc` gets
`y = Hx + noise`.

Good first sanity check before turning on nonlinear or wind cases.

### Nonlinear observations (arctan)

**Question:** What happens when observations are a **nonlinear function** of
the state (as in the score-filter prototype)?

- **LETKF:** `letkf_runner_nonlinear.csv` (`nonlinear_obs=True`)
- **ReverseSDE:** `ensf_runner_nonlinear.csv`

For the basic nonlinear case, just set `nonlinear_obs=True`. The default
operator is arctan, so synthetic obs become
`y = arctan(scalefact * Hx) + noise`.

`normalize_nonlinear` is a LETKF-only option and is not recommended for normal
experiments. ReverseSDE normalizes internally by default. Only touch
`nonlinear_operator_type` if you are adding or testing a custom nonlinear
operator; basic nonlinear runs do not need it.

This is the standard "harder than linear" test bed for comparing LETKF vs EnSF.

### Wind speed and wind direction

**Question:** Can the filter use **derived** wind quantities that are not direct
state variables?

Wind obs are **not** in the SPEEDY state vector. They are built from truth UG1/VG1:

- **WDG1** — direction, `arctan2(u, v)` (radians)
- **WSG1** — speed, `√(u² + v²)`

Enable them with indices **10** and **11** in `obs_plc` / `err_obs`.

| Goal | Template | Notes |
|------|----------|-------|
| Standard fields + wind (LETKF) | `letkf_runner_wind_vars.csv` | Linear T/U/V/H/P + wind; `option_mask=1` |
| Wind only, nonlinear wind op (LETKF) | `letkf_runner_only_wind.csv` | `wind_nonlinear_operator=True` |
| Wind (ReverseSDE) | `ensf_runner_wdg_wsg.csv` | Wind handled in score-filter likelihood |

For LETKF wind cases, use **`option_mask=1`** so U and V sit in the same
multivariate block. For fair LETKF vs ReverseSDE comparisons, match `s`,
`err_obs`, and which obs flags are on.

---

## Step 3: Configure the runner CSV

Each runner file has **one data row**. Think of the columns in groups:

### Identity and output location

| Column | Role |
|--------|------|
| `method` | `LETKF` or `ReverseSDE` |
| `exp_settings` | Path to truth/ensemble folder |
| `res_name` | Must be `t21` for this build |
| `code` | Leave empty to write under `../runs/` |

Outputs land in a folder named automatically:

```
<code_path>_<method>_<r>_<s>_<int(round(100*infla))>
```

Example: `t21_80_0.05_30_LETKF_3_1_115` → LETKF, r=3, s=1, infla=1.15.

### Observation network

| Column | Role |
|--------|------|
| `s` | Spacing of observing stations on the grid (`1` = every grid point eligible; larger = sparser) |
| `obs_plc` | Which quantities are observed (`1` = on, `0` = off) |
| `err_obs` | Observation **standard deviation** for each quantity (same order as below) |

**Index order** for `obs_plc` and `err_obs`:

| Index | Field |
|-------|-------|
| 0–4 | UG0, VG0, TG0, TRG0, PSG0 |
| 5–9 | UG1, VG1, TG1, TRG1, PSG1 |
| 10 | WDG1 (wind direction) |
| 11 | WSG1 (wind speed) |

You do not need to list all twelve entries if you only care about the first
few; trailing indices default to off.

### Nonlinear obs (when enabled)

| Column | Role |
|--------|------|
| `nonlinear_obs` | Turn on nonlinear standard observations; the basic/default operator is arctan |
| `nonlinear_operator_type` | For custom nonlinear operators; leave at the default for basic arctan runs |
| `scalefact` | Scale inside the nonlinear function |
| `normalize_nonlinear` | LETKF only; Z-score before arctan, not recommended for normal runs. ReverseSDE normalizes internally by default |

### LETKF-only tuning

| Column | Role |
|--------|------|
| `r` | Localization half-width in **grid indices**; patch is `(2r+1)×(2r+1)` cells |
| `infla` | Prior inflation ρ in `(k-1)I/ρ + …` (uniform at all grid points) |
| `option_mask` | How variables are grouped into local blocks (see below) |
| `wind_nonlinear_operator` | Nonlinear wind obs operator inside LETKF |

**`option_mask` in plain language:**

| Value | What it does | When to use |
|-------|--------------|-------------|
| **1** | All variables on a level updated together | Wind obs, paper-like multivariate LETKF; more memory |
| **2** | One variable per block | Default in older templates; lower memory |
| **3** | Pairs time levels | Rare; wind path disabled |

**`r` in plain language:** At mid-latitudes on t21, `r=1` is roughly a 3×3
cell patch (closest analogue to ~800 km outer radius in Szunyogh et al.).
`r=5` is a very large, weakly localized patch.

### ReverseSDE-only notes

| Column | Role |
|--------|------|
| `infla` | Final multiplier **after** spread restoration (typical values 0.6–1.4) |
| `r` | Ignored for analysis; leave at `1` |

Also produces **`sde_tracking.nc`** in the run folder root (not inside `data/`).

### Other

| Column | Role |
|--------|------|
| `list_snapshots` | Which cycles to write NetCDF diagnostics for |

---

## Step 4: Run

From `amlcs/`, run your edited CSV through the wrapper script:

```bash
./run_py.sh amlcs_da.py my_runner.csv
```

### What `run_py.sh` does

`run_py.sh` is a small logging wrapper around Python. It does not create the
experiment or choose parameters itself; your runner CSV does that. The wrapper
exists so long runs keep their stdout/stderr in a stable file that you can
inspect later.

Current script behavior:

- Argument 1 is the Python script to run (`amlcs_da.py`).
- Remaining arguments are passed through to that script (`my_runner.csv`).
- The log file is written under `LOG_DIR` inside `run_py.sh`.
- The log name includes the Python script, runner CSV name, timestamp, and
  either `SLURM_JOB_ID` or the local process id.
- The console prints only a short launch summary, including the exact log path.
- Python output is redirected to the log file with unbuffered output, so progress
  and failures are easier to debug while or after the run is executing.

Before running on a new machine or account, open `run_py.sh` and check:

```bash
LOG_DIR="/gpfs/home/jjs21b/AMLCS/logs"
```

Change that path if needed, and make sure the directory exists and is writable.
For example:

```bash
mkdir -p /gpfs/home/jjs21b/AMLCS/logs
```

When you launch a run, the console should look roughly like this:

```text
=== Running: amlcs_da.py
=== Params: my_runner.csv
=== Log file: /gpfs/home/jjs21b/AMLCS/logs/amlcs_da_my_runner_2026-06-11_14-03-12_123456.out
```

If the run fails, start with that `.out` file. It should contain the Python
traceback, the parsed runner CSV values printed by `amlcs_da.py`, and any model
or NetCDF errors emitted during setup or cycling.

Important: `run_py.sh` uses `SLURM_JOB_ID` in the log filename when Slurm set
that variable, but it does not submit a Slurm job by itself. If you call it from
an interactive shell, the experiment runs in that shell. If you call it inside a
Slurm batch script, the log filename will include the Slurm job id.

### What `amlcs_da.py` creates

Main outputs per run:

| File | Contents |
|------|----------|
| `unified_cycle<k>.nc` | Background/analysis means, obs, increments per cycle |
| `sde_tracking.nc` | **ReverseSDE only** — pseudo-time ensemble trajectories |

The run folder is named automatically from the input settings and key runner
values:

```text
../runs/<exp_settings_name>_<method>_<r>_<s>_<int(round(100*infla))>
```

For example, a LETKF run with `exp_settings=../LETKF_tuning/t21_80_0.05_30/`,
`r=3`, `s=1`, and `infla=1.15` writes under:

```text
../runs/t21_80_0.05_30_LETKF_3_1_115
```

### Organize into a named campaign folder (optional)

Post-processing scripts expect cycle NetCDFs under a `data/` subfolder. After a
run finishes, you can move them there with:

```bash
python letkf_r_tuning.py organize \
  "../runs/t21_80_0.05_30_LETKF_3_1_115" \
  --name my_campaign
```

This places files in `../runs/<run_folder>/my_campaign/data/`. You can also
create that folder and move `unified_cycle*.nc` by hand.

---

## Step 5: Post-processing — which tool when?

Different scripts answer different questions. Use this map:

| Question | Tool |
|----------|------|
| How does RMSE evolve over cycles for two methods? | `error_plots_dual_nc.py` |
| Where on the globe are increments / analysis errors large? | `heatmap_gifs_nc.py` |
| How did ReverseSDE ensemble members move in pseudo-time? | `spaghetti_plots_v2.py` |

All plotting scripts use a **USER SETTINGS block at the top of the file** —
edit paths there, then run with plain `python script.py`.

---

### Time-series method comparison (`error_plots_dual_nc.py`)

**Use when:** You want line plots of RMSE vs cycle for **two methods** plus
NoDA and truth.

**Inputs:** Organized `data/` folders with `unified_cycle*.nc`; truth and
free_run from `REFERENCE_DIR`.

Edit at top of file:

```python
EXP1 = ".../ReverseSDE_.../my_campaign/data"
EXP2 = ".../LETKF_.../my_campaign/data"
REFERENCE_DIR = ".../LETKF_tuning/t21_80_0.05_30"
CYCLES = list(range(30))
VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]
```

```bash
python error_plots_dual_nc.py
```

Produces per-level and level-averaged figures under `OUTPUT_DIR` (or next to
the experiment if unset).

---

### Spatial heatmaps and GIFs (`heatmap_gifs_nc.py`)

**Use when:** You want to **see the spatial pattern** of increments or analysis
error evolve cycle by cycle — not just a single scalar RMSE.

Reads `unified_cycle*.nc` from an experiment `data/` folder (works for **both
LETKF and ReverseSDE**).

For each variable and cycle it builds:

| Product | Meaning |
|---------|---------|
| `{var}_spatial_means.png` | Domain-mean increment and \|analysis − truth\| vs cycle |
| `{var}_increment.gif` | Animated map of **analysis − background** |
| `{var}_ana_truth.gif` | Animated map of **\|analysis − truth\|** |

Edit at top of file:

```python
EXP_DIR   = ".../runs/.../my_campaign/data"
REFERENCE_DIR = ".../LETKF_tuning/t21_80_0.05_30"
OUT_DIR_NAME = "../heatmaps_nc"
VARS = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]
LEVEL_IDX = 7          # bottom level for 3D vars; PSG1 ignores this
NONLINEAR_OBS = True   # match your runner if using arctan obs
SCALEFACT = 0.5        # match runner scalefact for innovation maps
```

**Dual mode** (`DUAL_MODE = True`): side-by-side LETKF vs ReverseSDE GIFs with
a shared color scale — useful for conference figures.

```python
DUAL_MODE = True
EXP_DIR   = ".../ReverseSDE_.../data"
EXP_DIR_2 = ".../LETKF_.../data"
EXP_LABEL_1 = "ReverseSDE"
EXP_LABEL_2 = "LETKF"
```

```bash
python heatmap_gifs_nc.py
```

Output: `EXP_DIR/../heatmaps_nc/` (or `heatmaps_nc_dual` if configured).

Requires **cartopy** on the cluster for coastlines. Innovation GIFs are
skipped by default because approximating nonlinear obs-space innovations from
NetCDF alone is misleading unless `NONLINEAR_OBS` / `SCALEFACT` match the run.

---

### ReverseSDE spaghetti plots (`spaghetti_plots_v2.py`)

**Use when:** You want to understand **how the score filter converges** — each
line is one ensemble member's trajectory through **pseudo-time** during a single
assimilation cycle.

**Requires:** `sde_tracking.nc` in the **run folder root** (written
automatically by ReverseSDE at cycle 0 onward). This file is **not** produced
by LETKF runs.

The SDE evolves a normalized state `x_t` from noise toward the posterior. The
script plots, for each cycle and variable:

- One colored line per ensemble member vs pseudo-time step
- Red = start of pseudo-time, green = end
- Side panel: kernel density of member values at start vs end

Default tracked grid points (must match `sequential_methods.py` unless you
change both):

| Point | Default label |
|-------|----------------|
| (lat=8, lon=31) | Largest initial background error |
| (lat=24, lon=36) | Largest analysis increment |

Edit at top of file:

```python
NC_PATH = ".../runs/t21_80_0.05_30_ReverseSDE_1_1_100/sde_tracking.nc"
OUT_DIR_BASE = ".../runs/t21_80_0.05_30_ReverseSDE_1_1_100/sde_plots"
PLOT_NORMALIZED = True   # xt_norm_* (observation-space normalization)
PLOT_PHYSICAL = False      # xt_state_* (physical units)
TARGET_VARS = None         # or ["TG1", "UG1"] to limit variables
```

```bash
python spaghetti_plots_v2.py
```

Outputs under `OUT_DIR_BASE/`:

```
normalized_mean/       # spatial mean trajectories
normalized_gridpoint/  # at tracked lat/lon points
  TG1/cycle3_TG1_pt0_lat8_lon31.png
  ...
```

Set `SPLIT_PLOTS = True` to split early vs late pseudo-time steps into separate
figures. Set `PLOT_PHYSICAL = True` to also plot denormalized `xt_state_*`
variables.

**How to read a spaghetti plot:** If members collapse to a narrow bundle by the
end of pseudo-time, the SDE is confidently pulling the ensemble toward a
consistent analysis. Wide spread at the end suggests noisy observations,
mis-tuned inflation, or difficult nonlinear obs.

Related scripts: `plot_sde_trajectories_conference.py` (conference-style
layouts), `inspect_sde.py` / `inspect_netcdf.py` (quick NetCDF peek).

---

## Quick template picker

| I want to… | Method | Start from |
|------------|--------|------------|
| Linear baseline | LETKF | `letkf_runner_normal.csv` |
| Linear baseline | ReverseSDE | `ensf_runner_linear_m1.csv` |
| Nonlinear arctan | LETKF | `letkf_runner_nonlinear.csv` |
| Nonlinear arctan | ReverseSDE | `ensf_runner_nonlinear.csv` |
| Standard fields + wind | LETKF | `letkf_runner_wind_vars.csv` |
| Wind only (nonlinear op) | LETKF | `letkf_runner_only_wind.csv` |
| Wind obs | ReverseSDE | `ensf_runner_wdg_wsg.csv` |

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `NetCDF: Unknown file format` on `ensemble_member.nc` | Disk quota full during setup — each run copies the full `exp_settings` tree into `../runs/`. Truncated files (~32 bytes) mean the copy failed. Free space, delete old runs, retry. |
| Spaghetti script: file not found | Point `NC_PATH` at `sde_tracking.nc` in run root; only ReverseSDE writes it. |
| Heatmap GIFs all NaN | Wrong `data/` path, or cycles not organized yet. |
| Wind obs seem ignored (LETKF) | Check `obs_plc` indices 10/11, `option_mask=1`, and `wind_nonlinear_operator` if needed. |
| LETKF `infla` has no effect on analysis | Need recent `sequential_methods.py` with `(Nens-1)/infla` in the LETKF solve. |

Further API detail: root [README.md](../../../README.md) and source under
`amlcs/`.
