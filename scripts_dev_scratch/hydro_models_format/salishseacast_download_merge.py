# SOOOOOOOO, this seems to work for small datasets, but it has issues with larger ones, and its not clear what the issue is.
# I'm abandoning this for now. Just download manually and concat with NCO.
# DONE IS BETTER THAN PERFECT
# UPDATE: This works but you need a really good internet connection. It kept failing at home and the problem seemed to be the concatenation, but I guess it was just losing connection. This worked when running at UBC.
# This is by far the superior method than doing it more manually in the other script. The compression is better with this method.

# Purpose:
# download and merge salishseacast data
# there is a 2GB limit on downloads
# so need to download smaller chunks and merge

# refer to hydro_models_format_salishsea_download_ARCHIVE.py for old way I tried to do this
# those methods were too slow and did not work

# this code is from BMM (modified)
# https://nbviewer.jupyter.org/urls/bitbucket.org/salishsea/analysis-ben/raw/tip/notebooks/master_hindcast_extractor.ipynb
# I removed a lot of functionality from his original code.
# (e.g. masking, working with local files)

import xarray as xr
import os
from datetime import datetime, timedelta
from calendar import monthrange
import numpy as np
from tqdm import tqdm


##########
# helper functions
##########

def load_paths():
    paths = {
        'erddap': 'https://salishsea.eos.ubc.ca/erddap/griddap',
        'out': r'D:\Hakai\models\salishsea\salishseacast\ssc_',
    }
    return paths

def load_netCDF_keys(filesystem='errdap'):
    # NetCDF file keys master dict
    if filesystem == 'errdap':
        key_dict = {
            'temperature': 'g3DTracer',
            'salinity': 'g3DTracer',
            'nitrate': 'g3DBiology',
            'uVelocity': 'g3DuGrid',
            'vVelocity': 'g3DvGrid',
            'u_wind': 'aSurfaceAtmosphere',
            'v_wind': 'aSurfaceAtmosphere',
        }        
    return key_dict

def extract_variables(data_vars, ds, variables, key_dict, dates=[None], dim='time', indices={'gridX': slice(None), 'gridY': slice(None)}):

    # Define time index dict
    tindex = {dim: slice(*dates)}  # * unpacks list items, slice creates indices out of those dates.
    # note: don't confuse this 'dates' with dates that is global
    
    for var in variables:
        # Initialize data array
        if not var in data_vars:
            data_vars[var] = ds[key_dict[var]][var].isel(indices).sel(tindex).load()
            # start with key_dict because that is how we first created this slice
            # then var (e.g. 'vVelocity') to get the specific variable
            # isel in an xarray thing to select by position (i.e. index)
            # same with .sel, but it apparently lets you select by label (and it apprently lets you use nearest-neighbor lookup? Perhaps that is why we don't need to select the exact half-hour? Yes, since it is a slice, it will select date that falls within that range.)
        else: # Concatenate data arrays
            data_vars[var] = xr.concat([data_vars[var], ds[key_dict[var]][var].isel(indices).sel(tindex).load()], dim=dim,)        
    return data_vars


########
# Master function
########

def extract_hindcast(daterange, variables, res='1h', version='19-05', filesystem='errdap', indices={'x': slice(None), 'y': slice(None)}):
    
    # Prepare variable definitions
    years, months, days = [[getattr(date, key) for date in daterange] for key in ['year', 'month', 'day']]
    paths = load_paths()
    key_dict = load_netCDF_keys(filesystem=filesystem)
    ds, keys = {}, list(set([key_dict[var] for var in variables]))
    encoding = dict(zip(variables, np.repeat({'zlib': True}, len(variables))))  # this has something to do with compression when we save the file with xarray later
    prefix_out = os.path.join(paths['out'])
    
    # Initiate loading protocol based on filesystem
    if filesystem == 'errdap':
        #prefix_out = f'{prefix_out}{key_dict[variables[0]][1:]}_'
        for key in keys:
            ds[key] = xr.open_dataset(paths['erddap'] + f'/ubcSS{key}Fields{res}V{version}')
        attrs = ds[key_dict[variables[0]]].attrs  # these are global attributes
    else:
        raise ValueError(f'Unknown filesystem: {filesystem}')

    # Loop through years
    for year in range(years[0], years[1] + 1):
        
        # Initialize data_vars dict and parse months
        data_vars = {}
        monthday = [[1, 1], [12, 31]]
        monthspan = [1, 13]
        if year == years[0]: monthspan[0] = months[0]
        if year == years[1]: monthspan[1] = months[1] + 1
            
        # Extract data month by month
        for month in tqdm(range(*monthspan), desc=f'Loading {year}'):

            # Parse daterange
            day, monthdays = 1, monthrange(year, month)[1]
            if (year == years[0]) and (month == months[0]):
                day = days[0]
                monthdays = monthdays - day + 1
                monthday[0] = [month, day]
            if (year == years[1]) and (month == months[1]):
                monthdays = days[1]
                monthday[1] = [month, monthdays]
            startdate = datetime(year, month, day)

            # Load variables from ERDDAP using specified month range
            if filesystem == 'errdap':
                dates = [startdate, startdate + timedelta(monthdays)]
                data_vars = extract_variables(data_vars, ds, variables, key_dict, dates=dates, indices=indices)
            else:
                raise ValueError(f'Unknown filesystem: {filesystem}')

        # Save year's worth of data as netCDF file
        datestr = '_'.join(datetime(year, *md).strftime('%Y%m%d') for md in monthday)
        with xr.Dataset(data_vars=data_vars, attrs=attrs) as obj:
            obj.to_netcdf(prefix_out + datestr + '.nc', encoding=encoding)


########
# Perform the extraction
########

# Define indices and variables
indices = {'gridX': slice(0, 398), 'gridY': slice(0, 898), 'depth': 0}
variables = ['uVelocity', 'vVelocity']

dateranges = [
    (datetime(2017, 1, 1), datetime(2017, 3, 16)),
    (datetime(2017, 5, 1), datetime(2017, 7, 14)),
    (datetime(2017, 8, 1), datetime(2017, 10, 14)),
]

for daterange in dateranges:
   ds= extract_hindcast(daterange, variables, indices=indices)
