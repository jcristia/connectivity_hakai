# this script contains bits and pieces to explore the variables of a netcdf file

import netCDF4 as nc

#filename = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\fvcom_results\sample_20180928_edited\calvert03_12hours_tides_winds.nc'
filename = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED.nc'
dataset = nc.Dataset(filename, "r+")

print dataset.data_model

variables = dataset.variables.keys()
for variable in variables:
    print variable

for dim in dataset.dimensions.values():
    print(dim)
for var in dataset.variables.values():
    print(var)

for name in dataset.ncattrs():
    print name + ": " + getattr(dataset,name)

siglay = dataset.variables["siglay"][:]
siglay[:,0]
siglev = dataset.variables["siglev"][:]
h = dataset.variables["h"][:]


times = dataset.variables["time"][:]
xc = dataset.variables["xc"][:]
latc = dataset.variables["latc"][:]
lat = dataset.variables["lat"][:]
x = dataset.variables["x"][:]

u = dataset.variables["u"]
print len(dataset.variables["u"])
print dataset.variables["u"][0][0][0]
print dataset.variables["u"][0][0][0:5]
v = dataset.variables["v"]

print dataset.variables["km"][0]


print nc.Variable.ncattrs(u)
print nc.Variable.getncattr(u, "coordinates")

#nc.Variable.renameAttribute(u, "coordinates", "coordtest")
#nc.Variable.renameAttribute(u, "coordtest", "coordinates")

# The intitial file used lats and longs for the coordinates, which all had zero values
# using setncattr, I changed the coordinates attributes values to now use xc yc
#nc.Variable.setncattr(u, "coordinates", "time siglay xc yc")
#nc.Variable.setncattr(v, "coordinates", "time siglay xc yc")

# testing import issues from installing opendrift
#from mpl_toolkits import basemap
#import mpl_toolkits.basemap

x = np.array([1,2,3]) 
x[0] = np.nan 


import numpy as np
x = [0.,55595.29911269, 53906.2717006,0.,0.]
x = np.asarray(x)
deltax = 24674.199577145948

h = int((x.max()-x.min())/deltax)