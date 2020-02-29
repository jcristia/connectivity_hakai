# download salish sea cast data and concatenate files
# there is a 2GB limit on downloads (even when accessing remotely with xarray). 2.5 months is about 2.4 GB, so I need to download smaller chunks

import os
import urllib.request
import sys

# save location
folder = r'D:\Hakai\models\salishsea\salishseacast_20170101_20170316'
# dates for filename
dates_f = [
    '20170101_20170115',
    '20170116_20170131',
    '20170201_20170215',
    '20170216_20170228',
    '20170301_20170316'
    ]
# dates for url
dates_u = [
    '(2017-01-01T00:30:00Z):1:(2017-01-15T23:30:00Z)',
    '(2017-01-16T00:30:00Z):1:(2017-01-31T23:30:00Z)',
    '(2017-02-01T00:30:00Z):1:(2017-02-15T23:30:00Z)',
    '(2017-02-16T00:30:00Z):1:(2017-02-28T23:30:00Z)',
    '(2017-03-01T00:30:00Z):1:(2017-03-16T23:30:00Z)'
    ]

# Strings: first postion: u/v, second position: date, third position: t
outname = 'ubcSS_V19-05_{}{}{}.nc'

velocity = ['u', 'v']

base_url = 'https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSg3D{}GridFields1hV19-05.nc?{}Velocity[{}][(0.5000003):1:(0.5000003)][(0.0):1:(897.0)][(0.0):1:(397.0)]'

for date_f, date_u in zip(dates_f, dates_u):
    for vel in velocity:
        url = base_url.format(vel, vel, date_u)
        outfile = os.path.join(folder, outname.format(vel, date_f, ''))
        print("downloading " + date_f + " dataset")
        urllib.request.urlretrieve(url, outfile)

# create list of files separated by u/v
ncfiles = []
for vel in velocity:
    vels = []
    for date in dates_f:
        outfile = os.path.join(folder, outname.format(vel, date, ''))
        vels.append(outfile)
    ncfiles.append(vels)

ncfiles_out = []
for vel in velocity:
    vels = []
    for date in dates_f:
        outfile = os.path.join(folder, outname.format(vel, date, 't'))
        vels.append(outfile)
    ncfiles_out.append(vels)

# NCO notes:
#For some reason the tool "ncrcat" doesn't ship with nco for windows. However, apparently it is exactly the same thing as "ncra". Therefore, just copy ncra.exe and rename to ncrcat.exe (in C:\nco).
#To concatenate, you need a 'record' dimension to join on. This means a dimension that can grow to unlimited size, which is usually Time.
#Therefore, I need to change time to a record dimension in each nc file.
#I can use 'ncecat', but apparently this is slower because it automatically creates a new time dimension and then you are left with an extra dimension.

# change time to a record dimension
for files, files_out in zip(ncfiles, ncfiles_out):
    for file, file_out in zip(files, files_out):
        command = '''ncks -O --mk_rec_dmn time {} {}'''.format(file, file_out)
        os.system(command)

# nco merge
for files, vel in zip(ncfiles_out, velocity):
    command = '''ncrcat {} {}'''.format(' '.join(files), os.path.join(folder, outname.format(vel, '', '')))
    os.system(command)

# test
import netCDF4 as nc
data = os.path.join(folder, outname.format('u','',''))
dataset = nc.Dataset(data, "r+")
variables = dataset.variables.keys()
for variable in variables:
    print (variable)
for dim in dataset.dimensions.values():
    print (dim)
for var in dataset.variables.values():
    print (var)

# delete original nc files
# NOTE: the nco processes have output that doesn't show up until you reset the terminal. This somteimes causes things to hang and you can't always delete these files. I tried a number of things to change this, but its not worth my time for this task.
# So just delete manually if this doesn't work.
[os.remove(item) for files in ncfiles for item in files]
[os.remove(item) for files in ncfiles_out for item in files]