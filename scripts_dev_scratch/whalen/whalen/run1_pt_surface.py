# release from 5 points on surface
# try with and without stranding
# start on the 1st day and release every hour for 12 hours
# for the stranding - run until all are stranded or until end of reader_hakai
# for the "previous" run for just 1 day

from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=20)

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\fvcom_results\cal03brcl_21_0003_EDITED.nc')
print reader_hakai

OceanDrift3D.fallback_values['ocean_vertical_diffusivity'] = 0.0
OceanDrift3D.fallback_values['vertical_advection'] = False

from opendrift.readers import reader_basemap_landmask
#reader_basemap = reader_basemap_landmask.Reader(
#                       llcrnrlon=-129.0, llcrnrlat=51.5,
#                       urcrnrlon=-127.7, urcrnrlat=52.25,
#                       resolution='f', projection='merc')
# zoomed in test
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-128.2, llcrnrlat=51.628,
                       urcrnrlon=-128.04, urcrnrlat=51.79,
                       resolution='f', projection='merc')
o.add_reader([reader_basemap, reader_hakai])

lons = [-128.155,-128.149,-128.135,-128.098,-128.129]
lats = [51.640,51.657,51.666,51.664,51.654]

from datetime import timedelta
time_step = timedelta(hours=1)
num_steps = 12
for i in range(num_steps):
    o.seed_elements(lons, lats, radius=0, number=500,
                    time=reader_hakai.start_time + i*time_step)

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'previous')
o.set_config('drift:scheme', 'euler')

o.run(end_time=reader_hakai.start_time + timedelta(days=1), time_step=30, time_step_output=1800,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\whalen\outputs\run1_pt_surface_20190327_nostranding.nc', export_variables=["age_seconds", "land_binary_mask"])
print o
o.plot()
o.animation()
o.plot(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\whalen\outputs\run1_pt_surface_20190327_nostranding.png')
o.animation(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\whalen\outputs\run1_pt_surface_20190327_nostranding.mp4', fps=7)

#####################

#o.io_import_file(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\whalen\outputs\run1_pt_surface_20190327_nostranding.nc')