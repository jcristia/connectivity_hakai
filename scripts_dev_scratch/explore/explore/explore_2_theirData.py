# compare their sample nc data to Pramod's

import netCDF4 as nc

filename = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\sample_20180815\calvert03_0001.nc'
filename_2 = r'C:\Python27\ArcGISx6410.3\Lib\site-packages\opendrift-master\tests\test_data\16Nov2015_NorKyst_z_surface/norkyst800_subset_16Nov2015.nc'
filename_3 = r'C:\Users\jcristia\Downloads\opendrift_telemacV2\correntes.nc'
filename_4 = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\explore\openoil_output.nc'

dataset = nc.Dataset(filename, "r+")
dataset_OD = nc.Dataset(filename_2, "r+")
dataset3 = nc.Dataset(filename_3, "r+")
dataset4 = nc.Dataset(filename_4, "r+")



print dataset.data_model
print dataset_OD.data_model

for attr in dataset4.ncattrs():
    print('{}: {}'.format(attr, dataset4.getncattr(attr)))
for dim in dataset4.dimensions.values():
    print(dim)

for var in dataset4.variables.values():
    print(var)

variables = dataset.variables.keys()
for variable in variables:
    print variable

variables = dataset.variables.keys()
for variable in variables:
    print variable

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

u = dataset_OD.variables["u"]
print nc.Variable.ncattrs(u)
print nc.Variable.getncattr(u, "coordinates")
# so the OD sample data coordinates is just based on lon and lat

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