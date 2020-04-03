# To run, check the 'local_add' path, and change the file name and dates.
# This is set up to read from nc files with u/v info stored in 1 file
# If I need to use it for u/v stored in separate files (as they would be if you download them directly) then change the file name in the load loop (an example is commented out), and change the filename in the time loop (an example is commented out)
# ALSO, the slicing now changes since there is no depth dimension. I have commmented out areas where this has changed.

# kindly borrowed from Ben Moore Maley
# https://nbviewer.jupyter.org/urls/bitbucket.org/salishsea/analysis-ben/raw/2e76930808c2db860a25b56870859ab254f2c306/notebooks/OpenDrift/sample_opendrift_simulation.ipynb

# I changed the order of some code and added in some "del" statements to remove objects from memory. Otherwise I was getting memory errors.
# 2.5 months of data seems to be the max that my computer can handle.

import numpy as np
import xarray as xr
import os
from dateutil.parser import parse

paths = {
    'local': r'D:\Hakai\models\salishsea',
    'local_add': r'salishseacast'
}

filename = 'ssc_20140801_20141014.nc'
dates = ['2014 Aug 01 00:30', '2014 Oct 14 23:30']

def unstagger(u, v):
    """Unstagger velocities from u,v points to t points
    """
    
    u = np.add(u[..., :-1], u[..., 1:]) / 2
    v = np.add(v[..., :-1, :], v[..., 1:, :]) / 2
    
    return u[..., 1:, :], v[..., 1:]

def rotate(u, v):
    """Rotate velocities from model grid to lon/lat space (29 deg)
    """

    theta = 29 * np.pi / 180
    u = u * np.cos(theta) - v * np.sin(theta)
    v = u * np.sin(theta) + v * np.cos(theta)

    return u, v

# Load forcing data from ERDDAP
raw = []
for vel in ['u', 'v']:
    #with xr.open_dataset(os.path.join(paths['local'], paths['local_add'], f'ubcSSg3D{vel}.nc')) as data:
    with xr.open_dataset(os.path.join(paths['local'], paths['local_add'], filename)) as data:
        #raw.append(data[f'{vel}Velocity'][:, 0, ...].values) #use this one if you have a depth dimension
        raw.append(data[f'{vel}Velocity'][:, ...].values)

# Unstagger velocities to T points and rotate to lon/lat
u, v = rotate(*unstagger(*raw))
del raw

# Daterange for simulation
daterange = [parse(d) for d in dates]
with xr.open_dataset(os.path.join(paths['local'], paths['local_add'], filename)) as data:
#with xr.open_dataset(os.path.join(paths['local'], paths['local_add'], 'ubcSSg3Du.nc')) as data:
    time = data.time.sel(time=slice(*daterange))

# Forcing path
fn = 'SalishSea_1h_' + '_'.join(d.strftime('%Y%m%d') for d in daterange) + '_opendrift.nc'
forcing_NEMO = os.path.join(paths['local'], paths['local_add'], 'forcing', fn)

grid = xr.open_dataset(os.path.join(paths['local'], 'ubcSSnBathymetryV17-02.nc'))
mask = xr.open_dataset(os.path.join(paths['local'], 'ubcSSn3DMeshMaskV17-02.nc'))

# Reshape, remove landpoints, and save to local netCDF path
tmask = mask.tmask[0, 0, 1:, 1:].values.reshape(-1).astype(bool)
ds = xr.Dataset(
    {
        'longitude': ('flat', grid.longitude[1:, 1:].values.reshape(-1)[tmask]),
        'latitude': ('flat', grid.latitude[1:, 1:].values.reshape(-1)[tmask]),
        'u': (['time', 'flat'], u.reshape(time.size, -1)[:, tmask], {'standard_name': 'x_sea_water_velocity'}),
        'v': (['time', 'flat'], v.reshape(time.size, -1)[:, tmask], {'standard_name': 'y_sea_water_velocity'}),
    },
    coords={'time': time}
).to_netcdf(forcing_NEMO)

# JC
# NOTE: this results in longitude and latitude as variable and not dimensions. Arc crashes everytime I try to view this file, but opendrift is able to use it.
# If I need to view it in the future then I will need to export the coordinate out.
