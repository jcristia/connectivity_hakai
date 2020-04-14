# for one community average simulation:
# create convex hull polygons and get point count

import os
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon, Point
import shapely.affinity
from math import atan2, degrees


project = 'seagrass_20200327_SS201408'
shp_pts = 'shp_merged\patch_centroids_metrics_commavg.shp'
out_poly = 'shp_merged\patch_clusters_convexhull.shp'

root = r'D:\Hakai\script_runs\seagrass'
path = os.path.join(root, project, shp_pts)
out_shp = os.path.join(root, project, out_poly)


df = gp.read_file(path)

# create convex hull polys
clusters = df.groupby('comidns')
polys_all = []
for name, cluster in clusters:
    point_count = len(cluster)
    if point_count > 2:
        poly = Polygon([[p.x, p.y] for p in cluster.geometry.values])
        convex_hull = poly.convex_hull
        polys_all.append([name, point_count, convex_hull, convex_hull.area])
    if point_count ==  2:  # for clusters with only 2 points, create a narrow ellipse
        # coordinates of midpoint
        point1 = cluster.iloc[0].geometry
        point2 = cluster.iloc[1].geometry
        mid_x = (point1.x + point2.x)/2
        mid_y = (point1.y + point2.y)/2        
        dist = point1.distance(point2)
        angle = degrees(atan2(point2.y - point1.y, point2.x - point1.x))

        # create ellipse
        # 1st elem = center point (x,y) coordinates
        # 2nd elem = the two semi-axis values (along x, along y)
        # 3rd elem = angle in degrees between x-axis of the Cartesian base
        #            and the corresponding semi-axis
        ellipse = ((mid_x, mid_y),(dist, 100),angle)
        # create a circle of radius 1 around center point:
        circ = shapely.geometry.Point(ellipse[0]).buffer(1)
        # create the ellipse along x and y:
        ell  = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
        # rotate the ellipse (clockwise, x axis pointing right):
        ellr = shapely.affinity.rotate(ell,ellipse[2])
        # If one need to rotate it clockwise along an upward pointing x axis:
        #elrv = shapely.affinity.rotate(ell,90-ellipse[2])
        # According to the man, a positive value means a anti-clockwise angle,
        # and a negative one a clockwise angle.
        polys_all.append([name, point_count, ellr, ellr.area])


gdf = gp.GeoDataFrame(polys_all, columns=['comid', 'pt_count', 'geometry', 'area'])
gdf.crs = df.crs
gdf.to_file(filename=out_shp, driver='ESRI Shapefile')