# track between rocky intertidal for hakai
# islands less than 40,000 removed


from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=20)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED.nc')
print reader_hakai

OceanDrift3D.fallback_values['ocean_vertical_diffusivity'] = 0.0
OceanDrift3D.fallback_values['vertical_advection'] = False

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-129.4, llcrnrlat=50.64,
                       urcrnrlon=-126.6, urcrnrlat=52.93,
                       resolution='f', projection='merc')
#reader_basemap.plot()
o.add_reader([reader_basemap, reader_hakai])

from datetime import datetime
from datetime import timedelta
time_step = timedelta(hours=1)
num_steps = 24
for i in range(num_steps):
    o.seed_from_shapefile(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs\hakai_margot\spatial\hakai_margot_reproject\hakai_polys_1rockyintertidal.shp', number=2000, time=reader_hakai.start_time + i*time_step)

o.elements_scheduled
o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

o.run(end_time=reader_hakai.end_time, time_step=30, time_step_output=1800,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs\hakai_margot\hakai_margot\1rockyintertidal\betweensites\output\rockyint_20190501_1.nc', export_variables=["age_seconds", "land_binary_mask"])
print o
#o.plot()
#o.animation()

#o.plot(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190415.png')
#o.animation(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190415.mp4', fps=10)

#####################
