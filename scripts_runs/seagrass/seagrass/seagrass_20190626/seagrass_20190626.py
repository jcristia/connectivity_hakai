# seagrass for all of hakai oceanographic model
# islands less than 40,000 removed
# seagrass less than 2,000 removed
# holes removed

from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=0)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'/home/jcristia/models/cal03brcl_21_0003_EDITED.nc')

OceanDrift3D.fallback_values['ocean_vertical_diffusivity'] = 0.0
OceanDrift3D.fallback_values['vertical_advection'] = False

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-129.4, llcrnrlat=50.64,
                       urcrnrlon=-126.6, urcrnrlat=52.93,
                       resolution='f', projection='merc')
o.add_reader([reader_basemap, reader_hakai])

from datetime import datetime
from datetime import timedelta
time_step = timedelta(hours=1)
num_steps = 24
for i in range(num_steps):
    o.seed_from_shapefile(r'seagrass_less2000_hakai.shp', number=750000, time=reader_hakai.start_time + i*time_step)

o.elements_scheduled
# export starting coordinates to use in biology script
# it is easier to do it here than deal with inconsistencies of the output nc file:
# releasing on delay with diffusion creates issues with the starting positions
import numpy as np
np.save(r'outputs/lon_1.npy', o.elements_scheduled.lon)
np.save(r'outputs/lat_1.npy', o.elements_scheduled.lat)

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

o.run(end_time=reader_hakai.end_time, time_step=30, time_step_output=1800,
      outfile=r'outputs/seagrass_20190626_1.nc', export_variables=["age_seconds", "land_binary_mask"])
print o
