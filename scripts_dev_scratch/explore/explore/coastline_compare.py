# run with seagrass data
# test behavior on coastline

from opendrift.models.openoil import OpenOil # for some reason OceanDrift3D doesn't set the correct environments for displaying figures. It only works if you import OpenOil
from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=0)

from opendrift.readers import reader_netCDF_CF_unstructured
#reader_hakai = reader_netCDF_CF_unstructured.Reader(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\fvcom_results\sample_20180928_edited\calvert03_12hours_tides_winds.nc')
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\fvcom_results\calvert03_tides_verified_EDITED.nc')
print reader_hakai
reader_hakai.plot()

# turn off 3D advection for now
print OceanDrift3D.required_variables
print OceanDrift3D.fallback_values
OceanDrift3D.fallback_values['ocean_vertical_diffusivity'] = 0.0
OceanDrift3D.fallback_values['vertical_advection'] = False
print OceanDrift3D.fallback_values

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-129.0, llcrnrlat=51.0,
                       urcrnrlon=-127.7, urcrnrlat=52.4,
                       resolution='f', projection='merc')

o.add_reader([reader_basemap, reader_hakai])

# seed from seagrass shapefile
o.seed_from_shapefile(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\seagrass_test_coastline\scratch.shp', number=100, time=reader_hakai.start_time)
o.elements_scheduled

# compare behaviour on coastline
o.list_configspec()
# set diffusion
#o.set_config('drift:current_uncertainty', 1)
o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')

o.run(end_time=reader_hakai.end_time, time_step=30,
          time_step_output=3600, outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\explore\coastline_compare.nc')
print o
o.plot()
o.animation()

#####################