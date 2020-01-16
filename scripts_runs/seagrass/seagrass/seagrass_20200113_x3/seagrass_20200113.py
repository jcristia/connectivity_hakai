
# full 60 day run with all particles

import sys
sys.path.append("/Linux/src/opendrift-master")
import numpy as np
from datetime import datetime
from datetime import timedelta
import ogr
import os

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_unstructured
from opendrift.readers import reader_basemap_landmask

# temporary fix for when running on anything other than mank01:
#sys.path.append("/Linux/src/opendrift-master")

#####################################################

# load readers outside of loop
file_salish_cluster = r'/home/jcristia/models/salishsea/SalishSea_1h_20160101_20160301_opendrift.nc'

reader_salish = reader_netCDF_CF_unstructured.Reader(file_salish_cluster)

reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-142.0, llcrnrlat=42.0,
                       urcrnrlon=-121.0, urcrnrlat=60.6,
                       resolution='f', projection='merc')

#####################################################

# multiple opendrift runs
# iterate over seagrass shapefiles

sg_path = r'/home/jcristia/runs/seagrass_20200113/seagrass_split'
sg_files = os.listdir(sg_path)
shapefiles = []
for file in sg_files:
    if file.endswith('.shp'):
        shapefiles.append(os.path.join(sg_path, file))

for shp in shapefiles:

    # get base number for output names
    base = os.path.splitext(os.path.basename(shp))[0]

    # get number of particles to seed
    shp = ogr.Open(shp)
    lyr = shp.GetLayer(0)
    for feature in lyr:
        particles = feature.GetField('particles')
        break

    # REMOVE THIS LATER:
    #particles = int(particles / 10)

    o = OceanDrift(loglevel=0)
    o.add_reader([reader_basemap, reader_salish])

    time_step = timedelta(hours=4)
    num_steps = 6
    for i in range(num_steps):
        o.seed_from_shapefile(shp, number=particles, time=reader_salish.start_time + i*time_step)

    # export starting coordinates to use in biology script
    # it is easier to do it here than deal with inconsistencies of the output nc file:
    # releasing on delay with diffusion creates issues with the starting positions
    np.save('outputs/lon_' + base + '.npy', o.elements_scheduled.lon)
    np.save('outputs/lat_' + base + '.npy', o.elements_scheduled.lat)

    o.set_config('drift:current_uncertainty_uniform', 1)
    o.set_config('general:coastline_action', 'stranding')
    o.set_config('drift:scheme', 'euler')

    o.run(end_time=reader_salish.end_time, time_step=30, time_step_output=1800,
          outfile='outputs/seagrass_' + base + '.nc', export_variables=["age_seconds", "land_binary_mask"])
    print(o)

    #o.plot(filename='D:\Hakai\script_runs\seagrass\seagrass_20191123\outputs\runX_' + base + '.png')

#####################################################