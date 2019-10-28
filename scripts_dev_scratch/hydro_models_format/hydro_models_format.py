# general scratch space for viewing netcdf files

import netCDF4 as nc
import numpy as np

#######
# Explore
#######

filename = r'D:\Hakai\models\fvcom_results\cal03brcl_21_0003_EDITED.nc'

dataset = nc.Dataset(filename, "r+")
variables = dataset.variables.keys()
for variable in variables:
    print (variable)
for dim in dataset.dimensions.values():
    print (dim)
for var in dataset.variables.values():
    print (var)

u = dataset.variables["u"][:]
