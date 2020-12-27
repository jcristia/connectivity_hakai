
filename = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\models\nep_nemo\Li_sample\combined\NEP36-OPM221_1h_20070101_20111205_grid_2D_20101231-20110320_UVALL.nc'

from opendrift.models.oceandrift import OceanDrift
o = OceanDrift(loglevel=20)

# test new reader
from opendrift.readers import reader_NEMO_pacific_JC
reader_nemo_pac = reader_NEMO_pacific_JC.Reader(filename)
print(reader_nemo_pac)

from opendrift.readers import reader_global_landmask
# test with just area around Hakai for now
reader_landmask = reader_global_landmask.Reader(extent=[-129, -126, 50, 52])
#reader_landmask.plot()
o.add_reader([reader_landmask, reader_nemo_pac])

o.seed_elements(lon=-128.5, lat=51.56, number=10, time=reader_nemo_pac.start_time)
o.elements_scheduled

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

#o.list_configspec()

from datetime import datetime
from datetime import timedelta
o.run(end_time=reader_nemo_pac.start_time + timedelta(days=5), time_step=60, time_step_output=3600, outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\hydro_models_format\output_1.nc', export_variables=["age_seconds", "land_binary_mask"])

print(o)
o.plot()
o.animation()

#####################
