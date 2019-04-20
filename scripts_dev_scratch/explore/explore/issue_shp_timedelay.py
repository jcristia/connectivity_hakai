# this was used to demonstrate to KnutFrode the issue of seeding from a shapefile on a time delay with diffusion

#from opendrift.models.openoil import OpenOil
from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=0)

from opendrift.readers import reader_basemap_landmask
reader_basemap = reader_basemap_landmask.Reader(
                       llcrnrlon=-128.13, llcrnrlat=51.65,
                       urcrnrlon=-128.11, urcrnrlat=51.663,
                       resolution='f', projection='merc')
o.add_reader([reader_basemap])

from datetime import datetime
from datetime import timedelta
o.seed_from_shapefile(r'C:\Users\jcristia\Documents\ArcGIS\seagrass_patch.shp', number=120, time=[datetime.now(), datetime.now()+timedelta(seconds=3600)])
#o.seed_from_shapefile(r'C:\Users\jcristia\Documents\ArcGIS\seagrass_patch.shp', number=120, time=datetime.now())

o.elements_scheduled

o.set_config('drift:current_uncertainty_uniform', 1)
o.set_config('general:coastline_action', 'none')

o.run(steps=120, time_step=30, time_step_output=3600,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\explore\issue_shp_timedelay.nc')
o.plot()