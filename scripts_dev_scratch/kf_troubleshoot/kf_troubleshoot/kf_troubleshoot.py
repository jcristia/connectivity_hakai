from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=20)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED_SHORTENED1.nc')

OceanDrift3D.fallback_values['ocean_vertical_diffusivity'] = 0.0

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-129.4, llcrnrlat=50.64,
                       urcrnrlon=-126.6, urcrnrlat=52.93,
                       resolution='f', projection='merc')
o.add_reader([reader_basemap, reader_hakai])

o.seed_elements(lon=-128.158, lat=51.638, number=100, time=reader_hakai.end_time)

o.elements_scheduled

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

o.run(end_time=reader_hakai.start_time, time_step=-30, time_step_output=1800,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\kf_troubleshoot\output_1.nc', export_variables=["age_seconds", "land_binary_mask"])
print o

o.plot()
o.animation()

#####################

#import netCDF4 as nc
#import numpy as np
#nc_output = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\kf_troubleshoot\output_1.nc'
#dataset = nc.Dataset(nc_output, "r+")
#lon = dataset.variables["lon"]
#lat = dataset.variables["lat"]
#traj = dataset.variables["trajectory"]
#status = dataset.variables["status"]
#timestep = dataset.variables["time"]

