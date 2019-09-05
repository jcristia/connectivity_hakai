# general scratch space for viewing netcdf files

import netCDF4 as nc

#######
# Explore
#######

filename = r''

dataset = nc.Dataset(filename, "r+")
variables = dataset.variables.keys()
for variable in variables:
    print (variable)
for dim in dataset.dimensions.values():
    print (dim)
for var in dataset.variables.values():
    print (var)

time = dataset.variables["time_counter"][:]