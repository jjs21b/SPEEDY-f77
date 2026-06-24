import netCDF4
nc = netCDF4.Dataset('/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data_ps0001/unified_cycle0.nc')
for k in nc.variables.keys():
    print(k)
