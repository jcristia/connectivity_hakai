import netCDF4 as nc
import numpy as np

filename = r'D:\Hakai\fvcom_results\cal03brcl_21_0003.nc'
dataset = nc.Dataset(filename, "r+")

variables = dataset.variables.keys()
for variable in variables:
    print variable

for dim in dataset.dimensions.values():
    print(dim)
for var in dataset.variables.values():
    print(var)

u = dataset.variables["u"][:]
np.shape(u)   # (481, 20, 45792)

siglay = dataset.variables["siglay"][:] # the middle of the sandwiches, this matches shape of u and v
siglay[:,0]
#siglev = dataset.variables["siglev"][:] # the ends of the sandwiches
#siglev[:,0]
h_center = dataset.variables["h_center"][:] # seafloor depth, need to use center to match shape of u and v

# pseudocode:
# make sure to use numpy where possible
# set target depth (e.g. 5m)
# get u into array
# for each time step
# for each node
# get h_center
# if less than target depth, remove point(?) or maybe just set to 0 so that it can still move by diffusion
# else, target_porportion = target_depth / h_center
#   get the two siglay levels that encompass this value
#       if target proportion less than or greater than 0.0025 or 0.9975 then just duplicate these levels so that the u value for those levels is chosen
#   get the two corresponding u values for these levels
#   prop = (target_proportion - siglay2) / (siglay1-siglay2)
#   u_new = ((u1 - u2) * prop) + u2

# for numpy
# go through each h_center, do target proportion into new array
# set to none where less than target? or to 0 so that it can move by diffusion
# then with target proportion, extract 2 siglays into new array and 2 u values into separate array

# then delete ones in siglay where u is none, then delete ones that are none in u, if not setting to 0
# then I guess you could copy the netcdf, then fill this into a new variable.  Would need to delete u and siglay, then edit siglay to have just 1 x z shape with the new target_proportion