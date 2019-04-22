# test if releasing from a polygon that directly lines up with coastline causes issues
# after running: there does not seem to be a problem. Opendrift seeds them on or just off the coastline, but does not throw an error.
# However, in one example, 2 out of 10,000 particles are being seeded way outside the poly. Not sure why this happens.
# Howerver, once I reduce it to 5,000 this doesn't happen.
# I tried with a different poly, and with 10,000 it had just 1 particle just barely outside the poly.
# So, there must be something with the geometry, number of particles, and equation used that makes this happen. Also, Knut-Frode mentioned that the initial positions are actually after 1 time step. I still don't understand how that could be the case, but perhaps that is also the issue.
# Therefore, I don't think I need to worry about this. The release is happening close enought to the polygon (in the grand scheme of things, I am working on a large scale so it shouldn't matter), and it is only a couple out of thousands.

# I also tested what happens when you seed from a polygon that overlaps an island (since you can't seed from a polygon with a hole in it.
# It correctly moves the points to the ocean, which is great. So now I don't need to worry about manually breaking polygons apart to account for islands.

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

# C:\Users\jcristia\Documents\ArcGIS\seagrass_1patch_erase.shp
# C:\Users\jcristia\Documents\ArcGIS\seagrass_1patch_test2_erase.shp
# C:\Users\jcristia\Documents\ArcGIS\seagrass_1patch_island.shp
o.seed_from_shapefile(r'C:\Users\jcristia\Documents\ArcGIS\seagrass_1patch_test2_erase.shp', number=5000, time=reader_hakai.start_time)

o.elements_scheduled
#o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'stranding')
o.set_config('drift:scheme', 'euler')

#end_time=reader_hakai.end_time
o.run(steps=1, time_step=30, time_step_output=1800,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\explore\test_seagrass_coastline.nc', export_variables=["age_seconds", "land_binary_mask"])
print o
o.plot()
#o.animation()

#o.plot(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190415.png')
#o.animation(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190415.mp4', fps=10)

#####################