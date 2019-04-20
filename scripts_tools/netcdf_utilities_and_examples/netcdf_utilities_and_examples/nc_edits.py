# edit netCDF4 file to match specific CF-convention for structured grids
# This script is the alternative to the NCO
# It is much slower than the nc_edits_NCO
# I'm only keeping it here for reference

import netCDF4 as nc
import numpy as np
filename = r'D:\Hakai\fvcom_results\calvert03_tides_verified_EDITED.nc'
dataset = nc.Dataset(filename, "r+")

for dim in dataset.dimensions.values():
    print(dim)
for var in dataset.variables.values():
    print(var)

# the unstructured reader says it can read xys if they have this standard name
# it read them, but it failed on other steps. I need latlon.
# therefore, change these standard names so they aren't recognized
xc = dataset.variables["xc"]
xc.standard_name = "projection_x_coordinate_TEMP"
yc = dataset.variables["yc"]
yc.standard_name = "projection_y_coordinate_TEMP"

# these will be the variables I use. They require an exact match on the standard name to be recognizable
lonc = dataset.variables["lonc"]
lonc.standard_name = "longitude"
latc = dataset.variables["latc"]
latc.standard_name = "latitude"

# change the standard names on the non-center latlons so they don't get confused with the ones I want above
lon = dataset.variables["lon"]
lon.standard_name = "longitude_TEMP"
lat = dataset.variables["lat"]
lat.standard_name = "latitude_TEMP"

# change the coordinates attributes
u = dataset.variables["u"]
u.coordinates = "time siglay latc lonc"
v = dataset.variables["v"]
v.coordinates = "time siglay latc lonc"

# change to match standard
ww = dataset.variables["ww"]
ww.long_name = "upward_seawater_velocity"

# convert xc,yc to lat lon
from pyproj import Proj, transform
inproj = Proj("+proj=utm +zone=9")
outproj = Proj("+proj=longlat +ellps=WGS84")

import itertools
lons = []
lats = []
for x,y in itertools.izip(xc,yc):
    lonx,laty = transform(inproj, outproj, x, y)
    lons.append(lonx)
    lats.append(laty)

# add data to lonc and latc variables
latc[:] = lats
lonc[:] = lons

# NOTE: something weird was happening for a while where not all the lats were being written. Once I reset the window and checked the values with the explore.py script, some values were zero. I ran it a few more times and it seemed to work. I'm wondering if it just need some dummy command after it.

#check
print latc
print lonc
variables = dataset.variables.keys()
latc_val = dataset.variables["latc"][:]
lonc_val = dataset.variables["lonc"][:]
print latc_val
print lonc_val

# fill dummy values for u and v
#import numpy as np
#u_fill = np.random.rand(13, 10, 45792)
#v_fill = np.random.rand(13, 10, 45792)

#u[:] = u_fill
#v[:] = v_fill


# time issue - hopefully this will get fixed by Pramod in next version so I don't have to do this
# time is saved in float32 variable and the interval is off
# for now, I am going to rename the time variable and create a new one

dataset.renameVariable("time", "timeOLD")
timeOLD = dataset.variables["timeOLD"]
timeOLD.long_name = "timeOLD"

dataset.createVariable('time', 'f8', ('time'))
time = dataset.variables['time']
time.long_name = 'time'
time.units = 'days since 1858-11-17 00:00:00'
time.format = 'modified julian day (MJD)'
time.time_zone = 'UTC'

# fill time values
# there was a one time issue where the time values were way off and clearly wrong
# also the difference between the 1st and 2nd value were different than all other values
t = []
start = 57844.0
interval = (1.0/24.0)
steps = 937
t.append(start)
count = 1
while len(t) < steps:
    value = start + (interval * count)
    t.append(value)
    count += 1
# convert to numpy float64 format
# actually this isn't necessary
#x = np.asarray(t)
#x.astype(float)
time[:] = t
