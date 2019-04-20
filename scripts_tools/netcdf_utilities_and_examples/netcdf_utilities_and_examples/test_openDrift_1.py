# following the walk-through:
# https://github.com/OpenDrift/opendrift/wiki/How-to-run-a-trajectory-simulation

from opendrift.models.openoil import OpenOil
o = OpenOil(loglevel=0)

from opendrift.readers import reader_netCDF_CF_generic
reader_norkyst = reader_netCDF_CF_generic.Reader(r'C:\Python27\ArcGISx6410.3\Lib\site-packages\opendrift-master\tests\test_data\16Nov2015_NorKyst_z_surface\norkyst800_subset_16Nov2015.nc')
print reader_norkyst
reader_norkyst.plot()

reader_arctic20 = reader_netCDF_CF_generic.Reader(r'C:\Python27\ArcGISx6410.3\Lib\site-packages\opendrift-master\tests\test_data\2Feb2016_Nordic_sigma_3d\Arctic20_1to5Feb_2016.nc')
print reader_arctic20
reader_arctic20.plot()


print OpenOil.required_variables
print OpenOil.fallback_values

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=2, llcrnrlat=59,
                       urcrnrlon=8, urcrnrlat=63,
                       resolution='h', projection='merc')

o.add_reader([reader_basemap, reader_norkyst, reader_arctic20])

from datetime import timedelta
o.seed_elements(lon=4, lat=60, number=100, radius=1000,
                    time=[reader_norkyst.start_time, reader_norkyst.start_time+timedelta(hours=5)])
o.elements_scheduled

o.list_configspec()
scheme = o.get_config('drift:scheme')
print scheme

o.run(end_time=reader_norkyst.end_time, time_step=3600,
          time_step_output=3600, outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\explore\openoil_output.nc', export_variables=['density', 'water_content'])

print o

o.plot(linecolor='z')
o.plot(background=['x_sea_water_velocity', 'y_sea_water_velocity'])
o.animation()

