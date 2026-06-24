#!/bin/bash
###############################################################################
# build_speedy.sh
#
# Automated build script for SPEEDY-f77 on modern HPC (AlmaLinux 8/9).
# Assumes Phase 1 (Spack install, env creation, compiler find) is complete.
# Will auto-load Spack and activate the 'speedy' environment if needed.
#
# Usage:
#   bash build_speedy.sh
#
###############################################################################

set -euo pipefail

# ---- Configuration --------------------------------------------------------
SPACK_ROOT="${SPACK_ROOT:-$HOME/spack}"
SPACK_ENV_NAME="speedy"

LIBS_PREFIX="$HOME/speedy_libs"
BUILD_DIR="$HOME/software_builds"
SPEEDY_REPO="https://github.com/jjs21b/SPEEDY-f77.git"
SPEEDY_DIR="$HOME/SPEEDY-f77"

ZLIB_VERSION="1.2.11"
HDF5_VERSION="1.8.12"
NETCDF_C_VERSION="4.3.2"
NETCDF_F_VERSION="4.2"

# Internet connectivity settings (for compute nodes behind a proxy)
INTERNET_CHECK_CMD="${INTERNET_CHECK_CMD:-curl -sf --max-time 10 -o /dev/null https://github.com}"
INTERNET_ACCESS_CMD="${INTERNET_ACCESS_CMD:-module load webproxy}"

# ---- Helper functions -----------------------------------------------------
info()  { echo "===> $*"; }
die()   { echo "ERROR: $*" >&2; exit 1; }

ensure_internet() {
    if eval "$INTERNET_CHECK_CMD" >/dev/null 2>&1; then
        return 0
    fi
    info "No internet access detected. Running: $INTERNET_ACCESS_CMD"
    eval "$INTERNET_ACCESS_CMD" >/dev/null 2>&1 || true
    if eval "$INTERNET_CHECK_CMD" >/dev/null 2>&1; then
        info "Internet access established."
        return 0
    fi
    die "No internet access. Check your proxy settings or set INTERNET_ACCESS_CMD."
}

# ---- Clean module environment ---------------------------------------------
module purge 2>/dev/null || true

# ---- Report nodename ------------------------------------------------------
info "Running on ${SLURMD_NODENAME:-$(hostname)}"

# ---- Install and load Spack if not already available ----------------------
if ! command -v spack >/dev/null 2>&1; then
    if [ -f "$SPACK_ROOT/share/spack/setup-env.sh" ]; then
        info "Loading Spack from $SPACK_ROOT..."
        source "$SPACK_ROOT/share/spack/setup-env.sh"
    else
        ensure_internet
        info "Spack not found. Installing to $SPACK_ROOT..."
        git clone --depth=1 https://github.com/spack/spack.git "$SPACK_ROOT" \
            || die "Failed to clone Spack"
        source "$SPACK_ROOT/share/spack/setup-env.sh"
        info "Spack installed and loaded."
    fi
fi

# ---- Detect OS version ----------------------------------------------------
IS_ALMA9=false
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "${ID:-}" == "almalinux" && "${VERSION_ID:-}" == 9* ]]; then
        IS_ALMA9=true
    fi
fi

# ---- Activate the Spack environment (create if needed) -------------------
if $IS_ALMA9; then
    # AlmaLinux 9: need an intermediate GCC 8.5.0 first, built in a
    # separate environment, before we can build GCC 4.8.5.
    BOOTSTRAP_ENV="build_gcc850"
    if spack find gcc@8.5.0 2>/dev/null | grep -q "gcc@8.5.0"; then
        info "GCC 8.5.0 already installed. Skipping bootstrap environment."
    else
        info "AlmaLinux 9 detected. Building intermediate GCC 8.5.0..."
        # Create/activate bootstrap environment
        if ! spack env list 2>&1 | grep -q "$BOOTSTRAP_ENV"; then
            spack env create "$BOOTSTRAP_ENV"
        fi
        spack env activate "$BOOTSTRAP_ENV"
        spack compiler find
        ensure_internet
        spack install --add gcc@8.5.0 languages=c,c++,fortran \
            || die "Failed to install GCC 8.5.0"
        spack compiler find
        spack compiler list
        spack env deactivate
        info "GCC 8.5.0 intermediate compiler ready."
    fi
fi

# Now create/activate the main speedy environment
CURRENT_ENV="$(spack env status 2>&1 || true)"
if echo "$CURRENT_ENV" | grep -q "$SPACK_ENV_NAME"; then
    info "Spack environment '$SPACK_ENV_NAME' is already active."
else
    if ! spack env list 2>&1 | grep -q "$SPACK_ENV_NAME"; then
        info "Creating Spack environment '$SPACK_ENV_NAME'..."
        spack env create "$SPACK_ENV_NAME" || die "Failed to create environment '$SPACK_ENV_NAME'"
        spack compiler find
    else
        info "Activating Spack environment '$SPACK_ENV_NAME'..."
    fi
    spack env activate "$SPACK_ENV_NAME" || die "Failed to activate environment '$SPACK_ENV_NAME'"
fi

###############################################################################
# PHASE 2: Patch and build GCC 4.8.5
###############################################################################

info "Phase 2: Patching Spack GCC recipe for 4.8.5 ucontext_t fix..."

GCC_PACKAGE_PY="$(spack location -p gcc)/package.py"
[ -f "$GCC_PACKAGE_PY" ] || die "Cannot find GCC package.py at: $GCC_PACKAGE_PY"

# Check if the patch is already applied
if grep -q 'struct ucontext' "$GCC_PACKAGE_PY" 2>/dev/null && \
   grep -q 'libgcc/config/i386/linux-unwind.h' "$GCC_PACKAGE_PY" 2>/dev/null; then
    info "ucontext_t fix already present in GCC recipe. Skipping."
else
    # Insert the filter_file block at the end of the existing patch() method.
    # We find 'def patch(self):' and inject our code before the next method.
    python3 -c "
import re, sys

pkg_file = '$GCC_PACKAGE_PY'
with open(pkg_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'libgcc/config/i386/linux-unwind.h' in content:
    print('Already patched.')
    sys.exit(0)

insert_block = '''
        # [SPEEDY] Fix struct ucontext for glibc >= 2.26 (GCC 4.8.5 only)
        if self.spec.satisfies(\"@4.8.5\"):
            filter_file(
                r\"struct ucontext\",
                \"ucontext_t\",
                \"libgcc/config/i386/linux-unwind.h\",
            )
'''

# Find the patch method and insert before the next 'def ' at the same indent
match = re.search(r'(    def patch\(self\):.*?)((?=\n    def )|(?=\nclass )|\Z)', content, re.DOTALL)
if match:
    insert_pos = match.end(1)
    content = content[:insert_pos] + insert_block + content[insert_pos:]
    with open(pkg_file, 'w') as f:
        f.write(content)
    print('Successfully inserted ucontext_t fix into patch() method.')
else:
    # No existing patch method — add one
    # Find the class body and insert after the first method
    match2 = re.search(r'(class Gcc\(.*?\):.*?\n)(    def )', content, re.DOTALL)
    if match2:
        insert_pos = match2.start(2)
        new_method = '''    def patch(self):
        # [SPEEDY] Fix struct ucontext for glibc >= 2.26 (GCC 4.8.5 only)
        if self.spec.satisfies(\"@4.8.5\"):
            filter_file(
                r\"struct ucontext\",
                \"ucontext_t\",
                \"libgcc/config/i386/linux-unwind.h\",
            )

'''
        content = content[:insert_pos] + new_method + content[insert_pos:]
        with open(pkg_file, 'w') as f:
            f.write(content)
        print('Created new patch() method with ucontext_t fix.')
    else:
        print('ERROR: Could not find insertion point in package.py', file=sys.stderr)
        sys.exit(1)
" || die "Failed to patch GCC recipe"
fi

info "Phase 2: Installing GCC 4.8.5 (this may take 30+ minutes)..."

# Check if already installed
if spack find gcc@4.8.5 2>/dev/null | grep -q "gcc@4.8.5"; then
    info "GCC 4.8.5 already installed. Skipping."
else
    ensure_internet
    spack install --add gcc@4.8.5 languages=c,c++,fortran ~libsanitizer cxxflags="-std=c++14" || \
    spack install --add gcc@4.8.5 languages=c,c++,fortran ~libsanitizer ~bootstrap cxxflags="-std=c++14" || \
    die "Failed to install GCC 4.8.5"
fi

info "Phase 2: Loading GCC 4.8.5..."
spack load gcc@4.8.5

# Verify
GF_PATH="$(which gfortran 2>/dev/null)" || die "gfortran not found after spack load"
GF_VER="$(gfortran --version | head -1)"
info "Compiler: $GF_PATH"
info "Version:  $GF_VER"

###############################################################################
# PHASE 3: Build dependencies from source
###############################################################################

info "Phase 3: Building dependencies in $BUILD_DIR -> $LIBS_PREFIX"

ensure_internet
mkdir -p "$BUILD_DIR" "$LIBS_PREFIX"
cd "$BUILD_DIR"

# ---- 3.2: zlib -----------------------------------------------------------
if [ -f "$LIBS_PREFIX/lib/libz.a" ]; then
    info "zlib already installed. Skipping."
else
    info "Building zlib $ZLIB_VERSION..."
    wget -q https://www.zlib.net/fossils/zlib-${ZLIB_VERSION}.tar.gz
    tar -xf zlib-${ZLIB_VERSION}.tar.gz
    cd zlib-${ZLIB_VERSION}
    ./configure --prefix="$LIBS_PREFIX"
    make -j"$(nproc)"
    make install
    cd "$BUILD_DIR"
fi

# ---- 3.3: HDF5 -----------------------------------------------------------
if [ -f "$LIBS_PREFIX/lib/libhdf5.a" ]; then
    info "HDF5 already installed. Skipping."
else
    info "Building HDF5 $HDF5_VERSION..."
    wget -q https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz
    tar -xf hdf5-${HDF5_VERSION}.tar.gz
    cd hdf5-${HDF5_VERSION}
    ./configure --prefix="$LIBS_PREFIX" --with-zlib="$LIBS_PREFIX" --enable-fortran
    make -j"$(nproc)"
    make install
    cd "$BUILD_DIR"
fi

# ---- 3.4: NetCDF-C -------------------------------------------------------
if [ -f "$LIBS_PREFIX/lib/libnetcdf.a" ]; then
    info "NetCDF-C already installed. Skipping."
else
    info "Building NetCDF-C $NETCDF_C_VERSION..."
    wget -q https://github.com/Unidata/netcdf-c/archive/refs/tags/v${NETCDF_C_VERSION}.tar.gz
    tar -xf v${NETCDF_C_VERSION}.tar.gz
    cd netcdf-c-${NETCDF_C_VERSION}

    export CPPFLAGS="-I$LIBS_PREFIX/include"
    export LDFLAGS="-L$LIBS_PREFIX/lib"
    ./configure --prefix="$LIBS_PREFIX" --disable-dap

    # Fix type mismatch in nc4internal.h (hid_t -> nc_type)
    info "Applying nc4internal.h fix (hid_t -> nc_type on line 351)..."
    sed -i 's/NC_TYPE_INFO_T \*nc4_rec_find_nc_type(const NC_GRP_INFO_T \*start_grp, hid_t target_nc_typeid)/NC_TYPE_INFO_T *nc4_rec_find_nc_type(const NC_GRP_INFO_T *start_grp, nc_type target_nc_typeid)/' \
        include/nc4internal.h

    make -j"$(nproc)"
    make install
    cd "$BUILD_DIR"
fi

# ---- 3.5: NetCDF-Fortran -------------------------------------------------
if [ -f "$LIBS_PREFIX/lib/libnetcdff.a" ]; then
    info "NetCDF-Fortran already installed. Skipping."
else
    info "Building NetCDF-Fortran $NETCDF_F_VERSION..."
    wget -q https://github.com/Unidata/netcdf-fortran/archive/refs/tags/netcdf-fortran-${NETCDF_F_VERSION}.tar.gz
    tar -xf netcdf-fortran-${NETCDF_F_VERSION}.tar.gz
    cd netcdf-fortran-netcdf-fortran-${NETCDF_F_VERSION}

    autoreconf -i

    export LD_LIBRARY_PATH="$LIBS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    # CPPFLAGS and LDFLAGS should still be set from the NetCDF-C step
    ./configure --prefix="$LIBS_PREFIX"

    # Remove 'man4' from SUBDIRS in Makefile (it references missing docs tooling)
    info "Removing 'man4' from Makefile SUBDIRS..."
    sed -i '/^SUBDIRS\s*=/ s/ man4//' Makefile

    make -j"$(nproc)"
    make install
    cd "$BUILD_DIR"
fi

info "All dependencies installed in $LIBS_PREFIX"

###############################################################################
# PHASE 4: Build the SPEEDY model
###############################################################################

info "Phase 4: Building SPEEDY model..."

# Clone if not already present
if [ -d "$SPEEDY_DIR" ]; then
    info "SPEEDY repo already exists at $SPEEDY_DIR. Skipping clone."
else
    ensure_internet
    cd ~
    git clone "$SPEEDY_REPO"
fi

cd "$SPEEDY_DIR/models/speedy/t21"

# Update the makefile with the correct library path
info "Configuring makefile with NETCDF_INSTALL_PATH = $LIBS_PREFIX"
sed -i "s|^NETCDF_INSTALL_PATH = .*|NETCDF_INSTALL_PATH = $LIBS_PREFIX|" makefile

# Ensure LD_LIBRARY_PATH is set for the compile step
export LD_LIBRARY_PATH="$LIBS_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Compile
info "Running compile.sh..."
bash compile.sh

if [ -f "imp.exe" ]; then
    info "SUCCESS: imp.exe has been built."
    info ""
    info "To run the model:"
    info "  cd $SPEEDY_DIR/models/speedy/t21"
    info "  ./imp.exe"
    info ""
    info "For Slurm submission, create a job script that includes:"
    info "  source ~/spack/share/spack/setup-env.sh"
    info "  spack env activate speedy"
    info "  spack load gcc@4.8.5"
    info "  export LD_LIBRARY_PATH=$LIBS_PREFIX/lib:\$LD_LIBRARY_PATH"
else
    die "compile.sh completed but imp.exe was not created."
fi
