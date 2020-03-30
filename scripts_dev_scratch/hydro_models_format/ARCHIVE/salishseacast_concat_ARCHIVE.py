# NOTE: use salishseacast_download_merge.py when possible. It is the superior method and results in smaller file sizes. However, you need a really good internet connection.

# Purpose:
# download salish sea cast data and concatenate files
# there is a 2GB limit on downloads (even when accessing remotely with xarray). 2.5 months is about 2.4 GB, so I need to download smaller chunks

# when downloading 2.5 months of data, I need to make sure I am accounting for days in the month. I want 75 days of data.
# WINTER
#[(2017-01-01T00:30:00Z):1:(2017-01-31T23:30:00Z)]
#[(2017-02-01T00:30:00Z):1:(2017-02-28T23:30:00Z)]
#[(2017-03-01T00:30:00Z):1:(2017-03-16T23:30:00Z)]
# SPRING
#[(2017-05-01T00:30:00Z):1:(2017-05-31T23:30:00Z)]
#[(2017-06-01T00:30:00Z):1:(2017-06-30T23:30:00Z)]
#[(2017-07-01T00:30:00Z):1:(2017-07-14T23:30:00Z)]
# SUMMER/FALL
#[(2017-08-01T00:30:00Z):1:(2017-08-31T23:30:00Z)]
#[(2017-09-01T00:30:00Z):1:(2017-09-30T23:30:00Z)]
#[(2017-10-01T00:30:00Z):1:(2017-10-14T23:30:00Z)]
# notice that for the very last time, I am going to the following day


import os
import sys

# save location
folder = r'D:\Hakai\models\salishsea\salishseacast_20170101_20170316'

# Strings: first postion: u/v, second position: date, third position: t
outname = 'ubcSSg3D{}'

velocity = ['u', 'v']

# create list of files separated by u/v
ncfiles = []
for vel in velocity:
    vels = []
    for file in os.listdir(folder):
        if file.startswith(outname.format(vel)):
            outfile = os.path.join(folder, file)
            vels.append(outfile)
    ncfiles.append(vels)

ncfiles_out = []
for vel in velocity:
    vels = []
    for file in os.listdir(folder):
        if file.startswith(outname.format(vel)):
            outfile = os.path.join(folder, 't' + file)
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
    command = '''ncrcat {} {}'''.format(' '.join(files), os.path.join(folder, outname.format(vel) + '.nc'))
    os.system(command)

# test
import netCDF4 as nc
data = os.path.join(folder, outname.format('u') + '.nc')
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
