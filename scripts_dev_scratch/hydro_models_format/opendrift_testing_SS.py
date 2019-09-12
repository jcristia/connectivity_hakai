filename = r'D:\Hakai\models\salishsea\forcing\SalishSea_1h_20160709_20160729_opendrift.nc'

from opendrift.models.oceandrift import OceanDrift
o = OceanDrift(loglevel=0)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_salishsea = reader_netCDF_CF_unstructured.Reader(filename)
print (reader_salishsea)

from opendrift.readers import reader_basemap_landmask

# test with just area around Hakai for now
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-142.0, llcrnrlat=42.0,
                       urcrnrlon=-121.0, urcrnrlat=60.6,
                       resolution='f', projection='merc')
o.add_reader([reader_basemap, reader_salishsea])

from datetime import datetime
from datetime import timedelta

o.seed_elements(lon=-123.27, lat=49.01, number=100, time=reader_salishsea.start_time+timedelta(days=19.5))
o.elements_scheduled

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

#o.list_configspec()


o.run(end_time=reader_salishsea.end_time, time_step=30, time_step_output=1800, outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\hydro_models_format\output_1.nc', export_variables=["age_seconds", "land_binary_mask"])

print(o)
o.plot()
o.animation()

#####################
