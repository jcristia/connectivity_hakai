# I am using nco instead of the netcdf4 python library to make file changes. It is much faster.

import os
import netCDF4 as nc
import numpy as np
from pyproj import Proj, transform
import itertools

# I kept getting error when I would try to create a copy with the same name, even though it creates an intermediate file and I had the overwrite on. SO, I need to use my own outname intermediate file
filename = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_TEMPTEST.nc'
outname = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_TEMPTEST_TEMP1.nc'



###### renaming ######
# the unstructured reader says it can read xys if they have this standard name
# it read them, but it failed on other steps. I need latlon.
# therefore, change these standard names so they aren't recognized
command = '''ncatted -O -a standard_name,xc,o,c,dummyname1 ''' + filename
os.system(command)
command = '''ncatted -O -a standard_name,yc,o,c,dummyname2 ''' + filename
os.system(command)
command = '''ncatted -O -a standard_name,lat,o,c,dummyname3 ''' + filename
os.system(command)
command = '''ncatted -O -a standard_name,lon,o,c,dummyname4 ''' + filename
os.system(command)
# these will be the variables I use. They require an exact match on the standard name to be recognizable
command = '''ncatted -O -a standard_name,lonc,o,c,longitude ''' + filename
os.system(command)
command = '''ncatted -O -a standard_name,latc,o,c,latitude ''' + filename
os.system(command)
# change to match standard
command = '''ncatted -O -a long_name,ww,o,c,upward_seawater_velocity ''' + filename
os.system(command)
# change the u/v coordinates attributes
command = '''ncatted -O -a coordinates,u,o,c,"time siglay latc lonc" ''' + filename
os.system(command)
command = '''ncatted -O -a coordinates,v,o,c,"time siglay latc lonc" ''' + filename
os.system(command)



###### convert xc,yc to lat lon ######
dataset = nc.Dataset(filename, "r+")
lonc = dataset.variables["lonc"]
latc = dataset.variables["latc"]
xc = dataset.variables["xc"]
yc = dataset.variables["yc"]
inproj = Proj("+proj=utm +zone=9")
outproj = Proj("+proj=longlat +ellps=WGS84")
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


###### delete variables I don't need ######
# cannot have spaces between variables
variables_delete = 'l,nprocs,nv,iint,zeta,nbe,ntsn,nbsn,ntve,nbve,a1u,a2u,aw0,awx,awy,art2,art1,tauc,omega,ua,va,temp,salinity,viscofm,viscofh,km,kh,kq,q2,q2l,l,partition,siglay_center,siglev,siglev_center,Itime,Itime2,Times'
# -O allows for overwriting so I dont have to respond to any prompts that I guess show up in the command line, -v extracts variables to new copy, and -x says to exclude the variables I list
command = '''ncks -O -x -v ''' + variables_delete + " " +  filename + " " + outname
os.system(command)
# Note: you can't get rid of all variables since many depend on many others and
# create a chain of dependencies



###### cleanup ######
# This might get hung up if you run it with the code above
# I've also noticed that even after you've cleared the window you still can't delete/rename
# You need to disconnect the drive and connect it again
import os
filename = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_TEMPTEST.nc'
outname = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_TEMPTEST_TEMP1.nc'
os.remove(filename)
os.rename(outname, filename)















