# release on grid
# with stranding
# start on the 1st day and release every hour for 24 hours. Run until end (20 days)

from opendrift.models.openoil import OpenOil
from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=20)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED.nc')
print reader_hakai

OceanDrift3D.fallback_values['ocean_vertical_diffusivity'] = 0.0
OceanDrift3D.fallback_values['vertical_advection'] = False

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-129.0, llcrnrlat=51.0,
                       urcrnrlon=-127.7, urcrnrlat=52.4,
                       resolution='f', projection='merc')
o.add_reader([reader_basemap, reader_hakai])

import numpy as np
lons = np.linspace(-128.5, -128.17, 10)
lats = np.linspace(51.38, 51.72, 10)
lons, lats = np.meshgrid(lons, lats)
lons = lons.ravel()
lats = lats.ravel()

from datetime import timedelta
time_step = timedelta(hours=1)
num_steps = 24
for i in range(num_steps+1):
    o.seed_elements(lons, lats, radius=0, number=100,
                    time=reader_hakai.start_time + i*time_step)

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

o.run(end_time=reader_hakai.end_time, time_step=30, time_step_output=3600,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\bhunt_collaboration\outputs\run1_grid_openocean_20190106.nc')
print o
o.plot(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\bhunt_collaboration\outputs\run1_grid_openocean_20190106.png')
o.animation(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\bhunt_collaboration\outputs\run1_grid_openocean_20190106.mp4', fps=10)

#####################