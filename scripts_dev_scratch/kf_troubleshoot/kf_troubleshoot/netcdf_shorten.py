import os
import netCDF4 as nc
import numpy as np

filename = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED_SHORTENED.nc'
outname = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED_SHORTENED1.nc'

dataset = nc.Dataset(filename, "r+")
variables = dataset.variables.keys()
for variable in variables:
    print variable
for dim in dataset.dimensions.values():
    print(dim)
for var in dataset.variables.values():
    print(var)

time = dataset.variables["time"][:]

# take a hyperslab of the data
command = '''ncks -d time,350,480 ''' + filename + ' ' + outname
os.system(command)


###### cleanup ######
# This might get hung up if you run it with the code above
# I've also noticed that even after you've cleared the window you still can't delete/rename
# You need to disconnect the drive and connect it again
import os
filename = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED_SHORTENED.nc'
outname = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED_SHORTENED1.nc'
os.remove(filename)
os.rename(outname, filename)