# extract points and color code based on origin point
# still some manual work in this (see notes below)
# I would consider this a short term soluation for now

import netCDF4 as nc
import numpy as np
from shapely.geometry import shape, Point, Polygon
import pandas as pd
import geopandas

filename = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\whalen\outputs\run1_pt_surface_20190327_nostranding.nc'
dataset = nc.Dataset(filename, "r+")

traj = dataset.variables["trajectory"]
lon = dataset.variables["lon"]
lat = dataset.variables["lat"]

# NOTE: I REALIZE NOW that with the roundabout way I am doing things, I don't need the coordinates in the first dataframe.
# get starting lat/lon for each particle
#lons = []
#lats = []
#for i in range(len(traj)):
#    for j in lon[i]:
#        if not np.ma.is_masked(j):
#            lons.append(j)
#            break
#    for j in lat[i]:
#        if not np.ma.is_masked(j):
#            lats.append(j)
#            break

# make into pandas df and attach trajectory ID
points = pd.DataFrame()
points['traj_id'] = list(traj)
#df['Coordinates'] = list(zip(lons, lats))
#df['Coordinates'] = df['Coordinates'].apply(Point)
#points = geopandas.GeoDataFrame(df, geometry='Coordinates')
#points.crs = {'init' :'epsg:4326'}

# note: this still requires manual work: assigning an order_id in the shapefile to the order that they get listed here. However, OpenDrift did seem to keep them in order.
num_points = 5
num_part_per_ts = 500
part_per_pt = num_part_per_ts / num_points
num_releases = 12
total_part = num_points * part_per_pt * num_releases
points['orig_pt'] = 0
orig_id = 1
for i in range(0,total_part,100):
    points.iloc[i:i+part_per_pt,1] = orig_id
    if orig_id < 5:
        orig_id += 1
    else:
        orig_id = 1

# get destination positions
lons_dest = []
lats_dest = []
for i in range(len(traj)):
    for j in range(len(lon[i])-1,-1,-1):
        if not np.ma.is_masked(lon[i][j]):
            lons_dest.append(lon[i][j])
            break
    for j in range(len(lat[i])-1,-1,-1):
        if not np.ma.is_masked(lat[i][j]):
            lats_dest.append(lat[i][j])
            break

df = pd.DataFrame()
df['Coordinates'] = list(zip(lons_dest, lats_dest))
df['Coordinates'] = df['Coordinates'].apply(Point)
df['traj_id'] = list(traj)
points_dest = geopandas.GeoDataFrame(df, geometry='Coordinates')
points_dest.crs = {'init' :'epsg:4326'}

# join origin to destination
points_dest_join = points_dest.merge(points, on='traj_id')

points_dest_join.to_file(filename=r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\whalen\spatial\whalen_0m_dest.shp', driver='ESRI Shapefile')