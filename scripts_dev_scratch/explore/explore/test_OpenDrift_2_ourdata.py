# following:
# https://github.com/OpenDrift/opendrift/wiki/How-to-run-a-trajectory-simulation
# in this one I try to use our sample data

from opendrift.models.openoil import OpenOil
o = OpenOil(loglevel=0)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\sample_20180924_JC_edits\calvert03_0001.nc')
print reader_hakai
reader_hakai.plot()

print OpenOil.required_variables
print OpenOil.fallback_values

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-130, llcrnrlat=50,
                       urcrnrlon=-126, urcrnrlat=54,
                       resolution='h', projection='merc')

o.add_reader([reader_basemap, reader_hakai])

from datetime import timedelta
o.seed_elements(lon=-128.5, lat=51.5, number=100, radius=1000,
                    time=[reader_hakai.start_time, reader_hakai.start_time+timedelta(hours=5)])
o.elements_scheduled

o.list_configspec()
scheme = o.get_config('drift:scheme')
print scheme

o.run(end_time=reader_hakai.end_time, time_step=1800,
          time_step_output=3600, outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\explore\please.nc')

print o

o.plot()
o.plot(background=['x_sea_water_velocity', 'y_sea_water_velocity'])
o.animation()