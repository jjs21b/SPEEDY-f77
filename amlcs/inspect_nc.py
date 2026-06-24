import netCDF4
import sys

path = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/reverseSDE_cycle7.nc"
try:
    ds = netCDF4.Dataset(path)
    print("Dimensions:")
    for d in ds.dimensions:
        print(f"  {d}: {len(ds.dimensions[d])}")
    print("\nVariables:")
    for v in ds.variables:
        if "PSG" in v or "psg" in v:
            print(f"  {v}: {ds.variables[v].dimensions}, {ds.variables[v].shape}")
    ds.close()
except Exception as e:
    print(e)
