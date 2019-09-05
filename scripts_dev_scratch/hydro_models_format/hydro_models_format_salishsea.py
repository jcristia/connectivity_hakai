# kindly borrowed from Ben Moore Maley
# https://nbviewer.jupyter.org/urls/bitbucket.org/salishsea/analysis-ben/raw/2e76930808c2db860a25b56870859ab254f2c306/notebooks/OpenDrift/sample_opendrift_simulation.ipynb

import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta
from dateutil.parser import parse

# Define paths, grid, and mask
paths = {
    'erddap': 'https://salishsea.eos.ubc.ca/erddap/griddap',
    'local': r'D:/Hakai/models/salishsea',
}
grid = xr.open_dataset(os.path.join(paths['local'], 'ubcSSnBathymetryV17-02.nc'))
mask = xr.open_dataset(os.path.join(paths['local'], 'ubcSSn3DMeshMaskV17-02.nc'))



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


# Daterange for simulation
daterange = [parse(d) for d in ['2016 Jul 09 00:30', '2016 Jul 29 00:30']]

# Forcing path
fn = 'SalishSea_1h_' + '_'.join(d.strftime('%Y%m%d') for d in daterange) + '_opendrift.nc'
forcing_NEMO = os.path.join(paths['local'], 'forcing', fn)


# Load forcing data from ERDDAP
raw = []
for vel in ['u', 'v']:
    with xr.open_dataset(os.path.join(paths['local'], f'ubcSSg3D{vel}GridFields1hV18-12.nc')) as data:
        time = data.time.sel(time=slice(*daterange))
        raw.append(data[f'{vel}Velocity'][:, 0, ...].values)

# Unstagger velocities to T points and rotate to lon/lat
u, v = rotate(*unstagger(*raw))

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