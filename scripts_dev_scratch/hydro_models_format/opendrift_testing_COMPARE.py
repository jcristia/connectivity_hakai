filename_nemo = r'D:\Hakai\NEP36-N26_5d_20140524_20161228_grid_UV_20160712-20160721_COMBINED_VCROPPED.nc'

filename_hakai = r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED.nc'

from opendrift.models.oceandrift import OceanDrift
o = OceanDrift(loglevel=0)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(filename_hakai)
print (reader_hakai)

from opendrift.readers import reader_NEMO_pacific_JC
reader_nemo_pac = reader_NEMO_pacific_JC.Reader(filename_nemo)
print(reader_nemo_pac)

OceanDrift.fallback_values['ocean_vertical_diffusivity'] = 0.0
OceanDrift.fallback_values['vertical_advection'] = False

from opendrift.readers import reader_basemap_landmask

# test with just area around Hakai for now
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-131.4, llcrnrlat=50.64,
                       urcrnrlon=-126.6, urcrnrlat=52.93,
                       resolution='f', projection='merc')
#reader_basemap.plot()
o.add_reader([reader_basemap, reader_hakai, reader_nemo_pac])

o.seed_elements(lon=-129.18, lat=51.43, number=100, time=reader_nemo_pac.start_time)
o.elements_scheduled

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

#o.list_configspec()

from datetime import datetime
from datetime import timedelta
o.run(end_time=reader_nemo_pac.end_time, time_step=30, time_step_output=1800, outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\hydro_models_format\output_1.nc', export_variables=["age_seconds", "land_binary_mask"])

print(o)
o.plot()
o.animation()

#####################
