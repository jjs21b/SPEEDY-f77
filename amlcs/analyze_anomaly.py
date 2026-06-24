import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import os

path = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/reverseSDE_cycle7.nc"
var_name = "truth_PSG1_lev0"
out_path = "anomaly_check.png"

try:
    ds = netCDF4.Dataset(path)
    data = ds.variables[var_name][:]
    
    # Gridpoint
    lat_idx, lon_idx = 27, 32
    
    val = data[lat_idx, lon_idx]
    
    
    # Coordinates (Approximate or just indices)
    # lats/lons not in file. Using indices.
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    z_score = (val - mean_val) / std_val
    
    print(f"Variable: {var_name}")
    print(f"Shape: {data.shape}")
    print(f"Value at ({lat_idx}, {lon_idx}): {val:.4f}")
    print(f"Value at ({lat_idx}, {lon_idx}): {val:.4f}")
    # print(f"Coordinates: Lat={lat_val:.2f}, Lon={lon_val:.2f}")
    print(f"Field Mean: {mean_val:.4f}")
    print(f"Field Mean: {mean_val:.4f}")
    print(f"Field Std: {std_val:.4f}")
    print(f"Z-Score: {z_score:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(data, 
                origin='lower',  # Convention for lat/lon usually, but check index orientation. 
                                # Usually index 0 is lat (y), index 1 is lon (x).
                                # origin='lower' puts index 0 at bottom.
                                # If lat 0 is south, this is correct.
                                # If lat 0 is north (top), we might want origin='upper'.
                                # I'll assume standard imshow for now (origin='upper') matches array layout
                                # unless I know lat 0 is -90.
                                # Let's use default (upper) to match matrix indices.
                cmap='viridis')
    plt.colorbar(label='Pressure (PSG1)')
    
    # Mark the point
    # x = lon_idx, y = lat_idx
    plt.scatter([lon_idx], [lat_idx], color='red', marker='x', s=100, label='(27,32)')
    
    plt.title(f"{var_name} at Cycle 7\nAt (27,32): Val={val:.2f}, Z={z_score:.2f}")
    plt.xlabel("Longitude Index")
    plt.ylabel("Latitude Index")
    plt.legend()
    
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    ds.close()

except Exception as e:
    print(f"Error: {e}")
