# test runs for building biology script
# calvert seagrass

from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=0)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED.nc')
#print reader_hakai

OceanDrift3D.fallback_values['ocean_vertical_diffusivity'] = 0.0
OceanDrift3D.fallback_values['vertical_advection'] = False

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-128.3, llcrnrlat=51.6,
                       urcrnrlon=-128.0, urcrnrlat=51.74,
                       resolution='f', projection='merc')
#reader_basemap.plot()
o.add_reader([reader_basemap, reader_hakai])

from datetime import datetime
from datetime import timedelta
time_step = timedelta(hours=1)
num_steps = 24
for i in range(num_steps):
    o.seed_from_shapefile(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\seagrass_working\seagrass_working_erase.shp', number=5000, time=reader_hakai.start_time + i*time_step)

o.elements_scheduled
o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

o.run(end_time=reader_hakai.end_time, time_step=30, time_step_output=1800,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190415_boatload_particles.nc', export_variables=["age_seconds", "land_binary_mask"])
print o
#o.plot()
#o.animation()

o.plot(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190415.png')
o.animation(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190415.mp4', fps=10)

#####################