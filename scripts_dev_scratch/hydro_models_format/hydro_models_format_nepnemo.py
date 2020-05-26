import netCDF4 as nc
import os
import numpy as np
import subprocess


#######
# Explore
#######

filename = r'D:\Hakai\models\nep_nemo\Li_sample\NEP36-OPM221_1h_20070101_20111205_grid_U_2D_20101231-20110109.nc'
filename = r'D:\Hakai\models\nep_nemo\Li_sample\NEP36-OPM221_1h_20070101_20111205_grid_V_2D_20101231-20110109.nc'

dataset = nc.Dataset(filename, "r+")
variables = dataset.variables.keys()
for variable in variables:
    print (variable)
for dim in dataset.dimensions.values():
    print (dim)
for var in dataset.variables.values():
    print (var)

time = dataset.variables["time_counter"][:]
#depth = dataset.variables["depthu"][:] # this seems to be the center of a slice
#depth_b = dataset.variables["depthu_bounds"][:] # these are the bounds of that slice
lon = dataset.variables["nav_lon"]
lat = dataset.variables["nav_lat"]

# From Pramod's version:
#lon[0] # notice how there are zeros at the end of every slice. This screws up how it gets read in by the reader. I need to crop these out.

##############################


#######
# put U and V variables into one file
#######

import subprocess
import os
import netCDF4 as nc
path = r'D:/Hakai/models/nep_nemo/Li_sample'
os.chdir(path)
os.getcwd()
u_file_copy = r'NEP36-OPM221_1h_20070101_20111205_grid_UV_2D_20101231-20110109_COMBINED.nc' # this can just be a renamed copy of the original U file
v_file = r'NEP36-OPM221_1h_20070101_20111205_grid_V_2D_20101231-20110109.nc'
command = '''ncks -A -v vos ''' + v_file + " " + u_file_copy
subprocess.call(command)
# check
dataset = nc.Dataset(u_file_copy, "r+")
for var in dataset.variables.values():
    print (var)
uos = dataset.variables["uos"]
vos = dataset.variables["vos"]

###########################################





#FROM OLD PRAMOD VERSION OF DATA

#######
# cropping a netCDF DEPTH
#######

# this works but you can't run it repeadetly for some reason. Had trouble overwriting itself.

import subprocess
import os
import netCDF4 as nc
path = r'D:/Hakai'
os.chdir(path)
os.getcwd()
infile = r'NEP36-N26_5d_20140524_20161228_grid_UV_20160712-20160721_COMBINED.nc'
outfile = r'NEP36-N26_5d_20140524_20161228_grid_UV_20160712-20160721_COMBINED_VCROPPED.nc'
command = '''ncks -d depthu,1,1 -d depthv,1,1 ''' + infile + " -O " + outfile
subprocess.call(command)
# check
dataset = nc.Dataset(outfile, "r+")
for dim in dataset.dimensions.values():
    print (dim)
for var in dataset.variables.values():
    print (var)

###########################################




#########
# cropping a netcdf X/Y
# INCOMPLETE
#########

# https://stackoverflow.com/questions/25731393/is-there-a-way-to-crop-a-netcdf-file

#outname = r'D:/Hakai/NEP36-N26_5d_20140524_20161228_grid_U_20160712-20160721_CROPPED.nc'
## find max of lat and lon values through all of data. Mask out the 0 values.
#ma_y = np.ma.masked_equal(lat, 0.0, copy=False)
#ymax = np.amax(ma_y)
#ymin = np.amin(ma_y)
#ma_x = np.ma.masked_equal(lon, 0.0, copy=False)
#xmax = np.amax(ma_x)
#xmin = np.amin(ma_x)
# this ended up being useless since I can only crop by x and y dimensions which is just a generic grid

#path = r'D:/Hakai'
#os.chdir(path)
#os.getcwd()
#command = '''ncks -d x,250,714 -d y,250,750 NEP36-N26_5d_20140524_20161228_grid_U_20160712-20160721.nc -O cropped_test.nc'''
## -d is dimension
## -O allows for overwrite of existing output file
#subprocess.call(command)

# this works, but all the 0 values are on the land side
# I think it was an issue in how Pramod originally clipped it
# Ideally all the 0 values could still be coordinates and they could just have 0 as their u/v values
# OR the 0 values could be considered missing values and maybe opendrift won't read them
# for now, I will just move on and see if it matters to opendrift.

################################
