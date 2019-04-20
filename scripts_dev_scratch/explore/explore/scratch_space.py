# I have been using this script to test the seed_from_shapefile scripting
# I'm trying to understand why things get screwed up with seeding over delayed time


shapefile = r'C:\Users\jcristia\Documents\ArcGIS\seagrass_scratch2.shp'
layername = None
featurenum = None
number = 100
import numpy as np

try:
    import ogr
    import osr
except Exception as e:
    logging.warning(e)
    raise ValueError('OGR library is needed to read shapefiles.')

#if 'timeformat' in kwargs:
#    # Recondstructing time from filename, where 'timeformat'
#    # is forwarded to datetime.strptime()
#    kwargs['time'] = datetime.strptime(os.path.basename(shapefile),
#                                        kwargs['timeformat'])
#    del kwargs['timeformat']

targetSRS = osr.SpatialReference()
targetSRS.ImportFromEPSG(4326)
try:
    s = ogr.Open(shapefile)
except:
    s = shapefile

for layer in s:
    if layername is not None and layer.GetName() != layername:
        print 'Skipping layer: ' + layer.GetName()
        continue
    else:
        print 'Seeding for layer: %s (%s features)' %(layer.GetDescription(), layer.GetFeatureCount())

    coordTrans = osr.CoordinateTransformation(layer.GetSpatialRef(),
                                                targetSRS)

    if featurenum is None:
        featurenum = range(1, layer.GetFeatureCount() + 1)
    else:
        featurenum = np.atleast_1d(featurenum)
    if max(featurenum) > layer.GetFeatureCount():
        raise ValueError('Only %s features in layer.' %
                            layer.GetFeatureCount())

    # Loop first through all features to determine total area
    total_area = 0
    layer.ResetReading()
    for i, f in enumerate(featurenum):
        feature = layer.GetFeature(f - 1)  # Note 1-indexing, not 0
        if feature is not None:
            total_area += feature.GetGeometryRef().GetArea()
    layer.ResetReading()  # Rewind to first layer
    print 'Total area of all polygons: %s m2' %(total_area)

    num_seeded = 0
    for i, f in enumerate(featurenum):
        feature = layer.GetFeature(f - 1)
        if feature is None:
            continue
        geom = feature.GetGeometryRef()
        num_elements = np.int(number*geom.GetArea()/total_area)
        if f == featurenum[-1]:
            # For the last feature we seed the remaining number,
            # avoiding difference due to rounding:
            num_elements = number - num_seeded
        print '\tSeeding %s elements within polygon number %s' %(num_elements, featurenum[i])
        try:
            geom.Transform(coordTrans)
        except:
            pass
        b = geom.GetBoundary()
        if b is not None:
            points = b.GetPoints()
            lons = [p[0] for p in points]
            lats = [p[1] for p in points]
        else:
            # Alternative if OGR is not built with GEOS support
            r = geom.GetGeometryRef(0)
            lons = [r.GetX(j) for j in range(r.GetPointCount())]
            lats = [r.GetY(j) for j in range(r.GetPointCount())]

        #self.seed_within_polygon(lons, lats, num_elements, **kwargs)
        num_seeded += num_elements

######################################################

from matplotlib.patches import Polygon
import pyproj
from matplotlib.path import Path
have_nx = False

#if number == 0:
#    return

lons = np.asarray(lons)
lats = np.asarray(lats)
#if len(lons) < 3:
#    logging.info('At least three points needed to make a polygon')
#    return
#if len(lons) != len(lats):
#    raise ValueError('lon and lat arrays must have same length.')
poly = Polygon(list(zip(lons, lats)), closed=True)
# Place N points within the polygons
proj = pyproj.Proj('+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f '
                    '+lon_0=%f +R=6370997.0 +units=m +ellps=WGS84'
                    % (lats.min(), lats.max(),
                        (lats.min()+lats.max())/2,
                        (lons.min()+lons.max())/2))
lonlat = poly.get_xy()
lon = lonlat[:, 0]
lat = lonlat[:, 1]
# I wonder if we could just create a poly from the xy and reproject and then get the true area. I have a feeling that with weirdly shaped polygons that the area calculation is not accurate.
x, y = proj(lon, lat)
area = 0.0
for i in range(-1, len(x)-1):
    area += x[i] * (y[i+1] - y[i-1])
area = abs(area) / 2

# Make points, evenly distributed
deltax = np.sqrt(area/number)
lonpoints = np.array([])
latpoints = np.array([])
lonlat = poly.get_xy()
lon = lonlat[:, 0]
lat = lonlat[:, 1]
x, y = proj(lon, lat)
xvec = np.linspace(x.min() + deltax/2, x.max() - deltax/2,
                    int((x.max()-x.min())/deltax))
yvec = np.linspace(y.min() + deltax/2, y.max() - deltax/2,
                    int((y.max()-y.min())/deltax))
x, y = np.meshgrid(xvec, yvec)
lon, lat = proj(x, y, inverse=True)
lon = lon.ravel()
lat = lat.ravel()
points = np.c_[lon, lat]
if have_nx:
    ind = nx.points_inside_poly(points, poly.xy)
else:
    ind = Path(poly.xy).contains_points(points)
if not any(ind):  # No elements are inside, we seed on border
    lonpoints = np.append(lonpoints, lons[0:number])
    latpoints = np.append(latpoints, lats[0:number])
else:
    lonpoints = np.append(lonpoints, lon[ind])
    latpoints = np.append(latpoints, lat[ind])
if len(ind) == 0:
    logging.info('Small or irregular polygon, using center point.')
    lonpoints = np.atleast_1d(np.mean(lons))
    latpoints = np.atleast_1d(np.mean(lats))
# Truncate if too many
# NB: should also repeat some points, if too few
lonpoints = lonpoints[0:number]
latpoints = latpoints[0:number]
if len(lonpoints) < number:
    # If number of positions is smaller than requested,
    # we duplicate the first ones
    missing = number - len(lonpoints)
    lonpoints = np.append(lonpoints, lonpoints[0:missing])
    latpoints = np.append(latpoints, latpoints[0:missing])

# Finally seed at calculated positions
#self.seed_elements(lonpoints, latpoints, **kwargs)

