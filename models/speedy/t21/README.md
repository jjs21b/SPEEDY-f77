# Building SPEEDY-f77 with a Spack-Managed Compiler

This guide documents the tested procedure for building the legacy SPEEDY
atmospheric model on a modern HPC cluster (AlmaLinux 8/9 with glibc ≥ 2.26).

The approach is a hybrid:

- **Spack** is used to build and manage a period-correct GCC 4.8.5 compiler.
  This avoids depending on the cluster's `gnu/4.8.5` module, which will
  disappear after the AlmaLinux 9 upgrade.
- **Manual source builds** are used for the dependency libraries (zlib,
  HDF5, NetCDF-C, NetCDF-Fortran) and for SPEEDY itself, following the
  original [SPEEDY-f77 README](https://github.com/jjs21b/SPEEDY-f77/blob/legacy-build-fix/models/speedy/t21/README.md).

---

## Phase 1: Install Spack and Set Up the Environment

On an HPC login node:

```bash
# Clear any loaded modules to avoid conflicts
module purge

# Install Spack (if not already installed)
git clone --depth=1 https://github.com/spack/spack.git ~/spack

# Activate Spack shell integration
# (add this line to your ~/.bashrc to initialize Spack on every HPC login)
source ~/spack/share/spack/setup-env.sh

# Detect system compilers (takes a minute or two)
spack compiler find
```

---

## Phase 1.5: Build an Intermediate Compiler (AlmaLinux 9 only)

On AlmaLinux 9, the system GCC (typically 11.x+) is too new to directly
compile GCC 4.8.5. You need to first build GCC 8.5.0 as an intermediate
compiler, then use it to build 4.8.5 in Phase 2.

**Skip this phase entirely if you are on AlmaLinux 8** — the system
GCC 8.5.0 is already available and can build GCC 4.8.5 directly.

```bash
# Create a temporary environment for building GCC 8.5.0
spack env create build_gcc850
spack env activate -p build_gcc850

# Install GCC 8.5.0
spack install --add gcc@8.5.0 languages=c,c++,fortran

# Register it as a known compiler
spack compiler find
spack compiler list   # verify gcc@8.5.0 appears

# Deactivate the temporary environment
spack env deactivate
```

Now create the main SPEEDY environment and continue to Phase 2:

```bash
spack env create speedy
spack env activate -p speedy
```

---

## Phase 2: Patch and Build GCC 4.8.5

SPEEDY's Fortran 77/90 hybrid source requires the older, more permissive
`gfortran` from GCC 4.8.5. Building GCC 4.8.5 on a modern system (AlmaLinux
8/9 with glibc ≥ 2.26) requires one source-level fix because the 4.8.5 code
references `struct ucontext`, which modern glibc removed in favor of the
POSIX-standard `ucontext_t` type.

If you are on **AlmaLinux 8** and skipped Phase 1.5, create the SPEEDY
environment now:

```bash
spack env create speedy
spack env activate -p speedy
```

If you are on **AlmaLinux 9** and completed Phase 1.5, the `speedy`
environment should already be created and active.

### Step 2a: Add the ucontext_t fix to Spack's GCC recipe

```bash
spack edit gcc
```

Find the existing `def patch(self):` method in the `Gcc` class and add the
following version-guarded block to it:

```python
        if self.spec.satisfies("@4.8.5"):
            # Fix struct ucontext for glibc >= 2.26
            filter_file(
                r"struct ucontext",
                "ucontext_t",
                "libgcc/config/i386/linux-unwind.h",
            )
```

Save and exit.

> **Why this can't be avoided:** There is no Spack variant or configure
> flag that controls this; `struct ucontext` is a hardcoded type name in
> a `libgcc` source file. The `filter_file()` call (equivalent to
> `sed -i 's/struct ucontext/ucontext_t/g'`) is the minimum necessary fix.
>
> **Do not** try to reuse Spack's existing `ucontext_t.patch` by extending
> its version range — it references architecture files (nios2, aarch64)
> absent from the 4.8.5 source tree. **Do not** use a standalone diff-style
> patch — `gcc-backport.patch` modifies the same file first, shifting line
> numbers and causing hunk failures. **Do not** use a `@when("@4.8.5")`
> decorator — it collides with Spack's `patch()` function and causes
> `'str' object has no attribute 'spec'` (I tried all these methods, and
> they caused their own problems).

### Step 2b: Install GCC 4.8.5

```bash
spack install --add gcc@4.8.5 languages=c,c++,fortran ~libsanitizer cxxflags="-std=c++14"
```

The `--add` flag adds GCC 4.8.5 to the environment's spec list and installs
it in one command. The `~libsanitizer` variant is critical:

- GCC 4.8.5's `libsanitizer` source has multiple incompatibilities with
  glibc ≥ 2.26 (missing `#include <signal.h>`, `__res_state` type changes,
  etc.). SPEEDY doesn't need sanitizers, so we disable them.
- Do **not** try to fix this by editing `configure_args` in `package.py` —
  the `+libsanitizer` variant appends `--enable-libsanitizer` later in the
  argument list, which overrides any earlier `--disable` (GCC's configure
  uses last-one-wins semantics).

### Step 2c: Load and verify GCC 4.8.5

```bash
spack load gcc@4.8.5

# Verify the correct compiler is active
which gfortran
gfortran --version
```

Keep GCC 4.8.5 loaded for all subsequent steps in this shell session.

---

## Phase 3: Build Dependencies from Source

The SPEEDY dependency libraries (zlib, HDF5, NetCDF-C, NetCDF-Fortran) are
built manually from source, following the original README's procedure. This
produces a working toolchain in `~/speedy_libs` that SPEEDY's makefile
expects.

### Step 3.1: Create directories

```bash
cd ~
mkdir software_builds
mkdir speedy_libs
cd software_builds
```

### Step 3.2: Download & build zlib

```bash
wget https://www.zlib.net/fossils/zlib-1.2.11.tar.gz
tar -xvf zlib-1.2.11.tar.gz
cd zlib-1.2.11
./configure --prefix=$HOME/speedy_libs
make
make install
cd ..
```

### Step 3.3: Download & build HDF5

```bash
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz
tar -xvf hdf5-1.8.12.tar.gz
cd hdf5-1.8.12
./configure --prefix=$HOME/speedy_libs --with-zlib=$HOME/speedy_libs --enable-fortran
make
make install
cd ..
```

### Step 3.4: Download & build NetCDF-C

```bash
wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.3.2.tar.gz
tar -xvf v4.3.2.tar.gz
cd netcdf-c-4.3.2
export CPPFLAGS="-I$HOME/speedy_libs/include"
export LDFLAGS="-L$HOME/speedy_libs/lib"
./configure --prefix=$HOME/speedy_libs --disable-dap
```

**IMPORTANT:** Before running `make`, apply a source fix to
`include/nc4internal.h` to correct a type mismatch that prevents compilation
on modern toolchains:

1. Open the file in a text editor (these instructions use `nano` but substitute whichever text editor you prefer):
   ```bash
   nano include/nc4internal.h
   ```

2. Go to line 351. Change:
   ```c
   NC_TYPE_INFO_T *nc4_rec_find_nc_type(const NC_GRP_INFO_T *start_grp, hid_t target_nc_typeid);
   ```
   to:
   ```c
   NC_TYPE_INFO_T *nc4_rec_find_nc_type(const NC_GRP_INFO_T *start_grp, nc_type target_nc_typeid);
   ```

3. Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

Then continue the build:

```bash
make
make install
cd ..
```

### Step 3.5: Download & build NetCDF-Fortran

```bash
wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/netcdf-fortran-4.2.tar.gz
tar -xvf netcdf-fortran-4.2.tar.gz
cd netcdf-fortran-netcdf-fortran-4.2
autoreconf -i   # Makes the configure script executable
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH
# The other environment variables should still be set from the last step.
./configure --prefix=$HOME/speedy_libs
```

**IMPORTANT:** Before running `make`, edit the generated Makefile:

1. Open the `Makefile` in a text editor:
   ```bash
   nano Makefile
   ```

2. Find the line that starts with `SUBDIRS =` (NOT `DISTSUBDIRS`). Use
   `Ctrl+W` in nano to search.

3. Delete the word `man4` from that line. Change:
   ```
   SUBDIRS = fortran f90 nf_test man4 examples
   ```
   to:
   ```
   SUBDIRS = fortran f90 nf_test examples
   ```

4. Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

Then continue the build:

```bash
make
# make check
make install
cd ..
```

All required libraries are now built and installed in `~/speedy_libs`.

---

## Phase 4: Build the SPEEDY Model

### Step 4.1: Clone the model repository

```bash
cd ~
git clone https://github.com/jjs21b/SPEEDY-f77.git
cd SPEEDY-f77/models/speedy/t21
```

### Step 4.2: Configure the makefile

This repository includes a `makefile` that is pre-configured for this build
process. You only need to edit the variable that points to your custom
library installation.

1. Open the `makefile` in a text editor.

2. Find the following configuration block near the top.

3. Replace the placeholder path `/gpfs/home/your_username` with the actual
   path to your home directory:

   ```makefile
   # --- START OF USER CONFIGURATION ---

   # TODO: Replace the path below with the full path to your home directory.
   NETCDF_INSTALL_PATH = "$HOME/speedy_libs"

   # These variables use the path defined above. No changes are needed here.
   NETCDF_FLAGS = -L$(NETCDF_INSTALL_PATH)/lib -lnetcdff
   NETCDF_INCLUDE = -I$(NETCDF_INSTALL_PATH)/include

   # --- END OF USER CONFIGURATION ---
   ```

> **Note:** If you started a new terminal session after installing the
> required libraries, re-run these commands before compilation to restore
> the GCC 4.8.5 compiler and library paths:

> ```bash
> source ~/spack/share/spack/setup-env.sh
> spack env activate speedy
> spack load gcc@4.8.5
> export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH
> ```

### Step 4.3: Compile the model

With the `makefile` configured, run the compile script:

```bash
bash compile.sh
```

This should create the executable file `imp.exe`.

---

## Running the Model with Slurm

Ensure you have the necessary input files in the model directory, then
create a Slurm job script `run_speedy.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=speedy
#SBATCH --output=speedy_%j.out
#SBATCH --error=speedy_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --partition=genacc_q       # ← or chipilskigroup_q 

# ---- Load Spack and the compiler ----
source ~/spack/share/spack/setup-env.sh    # ← this is okay to run multiple times; it is idempotent
spack env activate speedy                  # ← we don't use the `-p` for non-interactive jobs since it only adds a prefix to the shell prompt
spack load gcc@4.8.5

# ---- Set runtime library paths ----
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

# ---- Run the model ----
cd ~/SPEEDY-f77/models/speedy/t21
./imp.exe
```

Submit and monitor:

```bash
sbatch run_speedy.sh
squeue -u $USER
```

---

## Appendix A: Useful Spack Commands

| Task                              | Command                                           |
|-----------------------------------|---------------------------------------------------|
| List known compilers              | `spack compiler list`                             |
| Show environment's spack.yaml     | `spack config get`                                |
| Edit spack.yaml                   | `spack config edit`                               |
| Install and add in one step       | `spack install --add <spec>`                      |
| Find a package's install prefix   | `spack location -i <spec>`                        |
| Check what's installed            | `spack find`                                      |
| Deactivate environment            | `spack env deactivate`                            |
| List all environments             | `spack env list`                                  |
| Destroy an environment            | `spack env remove <name>`                         |
| Edit a package recipe             | `spack edit <package>`                            |

---

## Appendix B: Troubleshooting

**GCC 4.8.5 build fails with `dereferencing pointer to incomplete type`
at `md-unwind-support.h:65`:** The `struct ucontext` vs `ucontext_t`
incompatibility with glibc ≥ 2.26. Apply the `filter_file()` fix in
Phase 2, Step 2a. Do not reuse Spack's existing `ucontext_t.patch`
(wrong files), do not use a standalone diff (context mismatch with
`gcc-backport.patch`), do not use `@when` decorator (breaks Spack).

**GCC 4.8.5 build fails in `libsanitizer` (SIGSEGV undeclared,
`__res_state` errors, etc.):** Use the `~libsanitizer` variant flag on
the install command. Do not edit `configure_args` — the `+libsanitizer`
variant appends `--enable-libsanitizer` later, overriding any earlier
`--disable` (last-one-wins semantics).

**NetCDF-C `make` fails with `incompatible pointer type` for
`nc4_rec_find_nc_type`:** Edit `include/nc4internal.h` line 351 and
change `hid_t target_nc_typeid` to `nc_type target_nc_typeid` (see
Step 3.4).

**NetCDF-Fortran `make` fails looking for `man4` documentation tools:**
Remove `man4` from the `SUBDIRS =` line in the generated Makefile (see
Step 3.5).

**Runtime "library not found" for imp.exe:** Ensure
`LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH` is set before
running the executable. The Slurm script in Phase 5 handles this.

---

## Phase 5: AMLCS Data Assimilation Experiments

After SPEEDY is built, assimilation experiments (LETKF and ReverseSDE / EnSF)
are run from the **`amlcs/`** directory at the repository root, not from this
`t21/` model folder.

See **[AMLCS_EXPERIMENTS.md](./AMLCS_EXPERIMENTS.md)** for:

- How twin experiments work (truth, ensemble, synthetic obs, NoDA baseline)
- Choosing linear, nonlinear (arctan), and wind speed/direction setups
- LETKF vs ReverseSDE and which parameters actually matter for each
- Runner CSV fields, templates, and parameter sweeps
- Post-processing: RMSE summaries, dual error plots, **heatmap GIFs**
  (`heatmap_gifs_nc.py`), and **ReverseSDE spaghetti plots**
  (`spaghetti_plots_v2.py`)

Quick start (from `amlcs/` on the cluster):

```bash
./run_py.sh amlcs_da.py letkf_runner_nonlinear.csv
```
