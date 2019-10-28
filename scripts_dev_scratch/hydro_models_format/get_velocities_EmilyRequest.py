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


##############################################################

# Getting average current speed for Pruth Bay for Emily:
u = dataset.variables["u"][:]
v = dataset.variables["v"][:]
# shape for u and v: (481, 20, 45792) (time, siglay, nele)
# so I want all times, siglay 0, and the specific element
# pruth south: nele 20693 (index 20692)
# pruth north: nele 20946 (index 20945)
# wolf: nele 24038 (index 24037)
# sandspit: nele 25490 (index 25489)

# use numpy to slice
psu = u[:, 0, 20692]
psv = v[:, 0, 20692]

pnu = u[:, 0, 20945]
pnv = v[:, 0, 20945]

wu = u[:, 0, 24037]
wv = v[:, 0, 24037]

ssu = u[:, 0, 25489]
ssv = v[:, 0, 25489]

# calculate velocity
# taken from opendrift:
##azimuth = np.degrees(np.arctan2(x_vel, y_vel))  # Direction of motion
##velocity = np.sqrt(x_vel**2 + y_vel**2)  # Velocity in m/s

ps_vel = np.sqrt(psu**2 + psv**2)
pn_vel = np.sqrt(pnu**2 + pnv**2)
w_vel = np.sqrt(wu**2 + wv**2)
ss_vel = np.sqrt(ssu**2 + ssv**2)

# calc mean, min, max
ps_vel_mean = np.mean(ps_vel)
ps_vel_med = np.median(ps_vel)
ps_vel_min = np.min(ps_vel)
ps_vel_max = np.max(ps_vel)
ps_vel_std = np.std(ps_vel)

pn_vel_mean = np.mean(pn_vel)
pn_vel_med = np.median(pn_vel)
pn_vel_min = np.min(pn_vel)
pn_vel_max = np.max(pn_vel)
ps_vel_std = np.std(pn_vel)

w_vel_mean = np.mean(w_vel)
w_vel_med = np.median(w_vel)
w_vel_min = np.min(w_vel)
w_vel_max = np.max(w_vel)
w_vel_std = np.std(w_vel)

ss_vel_mean = np.mean(ss_vel)
ss_vel_med = np.median(ss_vel)
ss_vel_min = np.min(ss_vel)
ss_vel_max = np.max(ss_vel)
ss_vel_std = np.std(ss_vel)


print(ps_vel_mean)
print(ps_vel_med)
print(ps_vel_min)
print(ps_vel_max)
print(ps_vel_std)
print(pn_vel_mean)
print(pn_vel_med)
print(pn_vel_min)
print(pn_vel_max)
print(ps_vel_std)
print(w_vel_mean)
print(w_vel_med)
print(w_vel_min)
print(w_vel_max)
print(w_vel_std)
print(ss_vel_mean)
print(ss_vel_med)
print(ss_vel_min)
print(ss_vel_max)
print(ss_vel_std)

##############################################################