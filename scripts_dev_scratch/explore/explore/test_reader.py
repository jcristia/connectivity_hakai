# individual bits and pieces for testing how to read in my own basemap data


#from opendrift.readers import reader_basemap_hakai
# local test version:
import reader_basemap_hakai
from opendrift.models.openoil import OpenOil
o = OpenOil(loglevel=0)
shp = r'C:\Users\jcristia\Documents\ArcGIS\hakai_2islands.shp'
reader_basemap = reader_basemap_hakai.Reader(s=shp,
                       llcrnrlon=-129.0, llcrnrlat=51.0,
                       urcrnrlon=-127.7, urcrnrlat=52.4,
                       projection='merc')
print reader_basemap

#####################################################################

from mpl_toolkits.basemap import Basemap
llcrnrlon=-128.27
llcrnrlat=51.58
urcrnrlon=-127.91
urcrnrlat=51.76
resolution='f'
projection='merc'
map = Basemap(llcrnrlon, llcrnrlat,
                    urcrnrlon, urcrnrlat, area_thresh=0,
                    resolution=resolution, projection=projection)

map.landpolygons[1].boundary
# so if I can just get a series of arrays like this then I can hardcode everything else


#####################################################################

import cartopy.crs as ccrs
from cartopy.io import shapereader
import shapely

shp = r'C:\Users\jcristia\Documents\ArcGIS\hakai_2islands.shp'

# actually, I think I can just use ogr for this, extract boundaries then put them into an array.
# I just need to figure out how to deal with the projection. As long as I can define it with proj4 then I should be ok

coast = shapereader.Reader(shp)
# so apparently cartopy doesn't 


######################################################################

import ogr
import osr
import numpy as np
from matplotlib.patches import Polygon
shp = r'C:\Users\jcristia\Documents\ArcGIS\hakai_2islands.shp'
targetSRS = osr.SpatialReference()
targetSRS.ImportFromEPSG(4326)
s = ogr.Open(shp)

polygons = []
for layer in s:
    coordTrans = osr.CoordinateTransformation(layer.GetSpatialRef(),targetSRS)
    featurenum = range(1, layer.GetFeatureCount() + 1)
    
    for f in featurenum:
        feature = layer.GetFeature(f - 1)
        if feature is None:
            continue
        geom = feature.GetGeometryRef()

        try:
            geom.Transform(coordTrans)
        except:
            pass
        b = geom.GetBoundary()
        points = b.GetPoints()
        lons = [p[0] for p in points]
        lats = [p[1] for p in points]

        lons = np.asarray(lons)
        lats = np.asarray(lats)
        if len(lons) < 3:
            logging.info('At least three points needed to make a polygon')
            continue
        if len(lons) != len(lats):
            raise ValueError('lon and lat arrays must have same length.')
        poly = Polygon(list(zip(lons, lats)), closed=True)
        polygons.append(poly)

# check
polygons[1].get_xy()


coastsegs = [[]]
for poly in polygons:
    for x,y in poly.get_xy():
        coastsegs.append(list(zip(x,y)), closed = True)


# how did I deal with holes and multipart?



# save for later
#proj = pyproj.Proj('+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f '
#                    '+lon_0=%f +R=6370997.0 +units=m +ellps=WGS84'
#                    % (lats.min(), lats.max(),
#                        (lats.min()+lats.max())/2,
#                        (lons.min()+lons.max())/2))




###############################################################
# Test reader
###############################################################



from opendrift.models.openoil import OpenOil
from opendrift.models.oceandrift3D import OceanDrift3D
o = OceanDrift3D(loglevel=0)

from opendrift.readers import reader_basemap_landmask
#shp = r'C:\Users\jcristia\Documents\ArcGIS\hakai_2islands.shp'
reader_basemap = reader_basemap_landmask.Reader(llcrnrlon=-129.0, llcrnrlat=51.0,
                       urcrnrlon=-127.7, urcrnrlat=52.4,
                       projection='merc', resolution='f')
o.add_reader([reader_basemap])
from datetime import datetime
from datetime import timedelta
#o.seed_from_shapefile(r'C:\Users\jcristia\Documents\ArcGIS\seagrass_patch.shp', number=120, time=[datetime.now(), datetime.now()+timedelta(seconds=3600)])
o.seed_from_shapefile(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\seagrass\seagrass_patch.shp', number=120, time=datetime.now())

o.set_config('drift:current_uncertainty_uniform', 1)

o.run(steps=120, time_step=30, time_step_output=3600,
      outfile=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\explore\test_hakai_reader.nc')
o.plot()



##################################################################



coastsegs, coastpolygontypes = _readboundarydata('gshhs',as_polygons=True)
# reformat for use in matplotlib.patches.Polygon.
coastpolygons = []
for seg in coastsegs:
    x, y = list(zip(*seg))
    coastpolygons.append((x,y))
# replace coastsegs with line segments (instead of polygons)
coastsegs2, types = _readboundarydata('gshhs',as_polygons=False)
# create geos Polygon structures for land areas.
# currently only used in is_land method.
landpolygons=[]
lakepolygons=[]
if len(coastpolygons) > 0:
    x, y = list(zip(*coastpolygons))
    for x,y,typ in zip(x,y,coastpolygontypes):
        b = np.asarray([x,y]).T
        if typ == 1: landpolygons.append(_geoslib.Polygon(b))
        if typ == 2: lakepolygons.append(_geoslib.Polygon(b))