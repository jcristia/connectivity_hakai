# script to get the xy coords from a nc file

import netCDF4 as nc
import numpy as np
import itertools

filename = r'D:\Hakai\NEP36-N26_5d_20140524_20161228_grid_U_20160712-20160721.nc'
dataset = nc.Dataset(filename, "r+")

#print dataset.data_model
#variables = dataset.variables.keys()
#for variable in variables:
#    print variable
#for dim in dataset.dimensions.values():
#    print(dim)
#for var in dataset.variables.values():
#    print(var)

lat = dataset.variables["nav_lat"][:]
lon = dataset.variables["nav_lon"][:]

lat = np.array(lat).flatten()
lon = np.array(lon).flatten()

headers = ['lon', 'lat']
output = open(r'D:\Hakai\NEP_grid.csv', 'w')
for header in headers:
    output.write(header + ",")
output.write("\n")

for x,y in itertools.izip(lon,lat):
    if x != 0 and y != 0:
        # can only write strings
        output.write('%s' % x)
        output.write(',')
        output.write('%s' % y)
        output.write(',')
        output.write("\n")

output.close()