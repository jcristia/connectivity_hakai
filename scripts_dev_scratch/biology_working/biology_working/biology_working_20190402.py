# 

import netCDF4 as nc
import numpy as np
from shapely.geometry import shape, Point, Polygon, mapping
from shapely.ops import nearest_points
import fiona
import pandas as pd
import geopandas
import time
import math
#from osgeo import ogr

filename = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190313.nc'
dataset = nc.Dataset(filename, "r+")

###################
# exploration helpers
###################

#variables = dataset.variables.keys()
#for variable in variables:
#    print variable

#for dim in dataset.dimensions.values():
#    print(dim)
#for var in dataset.variables.values():
#    print(var)

#times = dataset.variables["time"][:]
#times.shape

#status = dataset.variables["status"]
#print len(status)
#status[0][0]
#status[0][0:5]

#times = dataset.variables["time"]
#print nc.Variable.ncattrs(times)
#print nc.Variable.getncattr(times, "units")

###################
# attach poly uID to particle
###################

# get starting lat/lon for each particle
# check which poly it is in

# starting positions
# particles start at different times, so check if it is masked, stop once we find our first value
lon = dataset.variables["lon"]
lons = []
for i in range(len(lon)):
    for j in lon[i]:
        if not np.ma.is_masked(j):
            lons.append(j)
            break
lat = dataset.variables["lat"]
lats = []
for i in range(len(lat)):
    for j in lat[i]:
        if not np.ma.is_masked(j):
            lats.append(j)
            break

# fiona/shapely approach
# points are in lat/lon, need in meters
#from pyproj import Proj, transform
#inproj = Proj("+proj=longlat +ellps=WGS84")
#outproj = Proj("+proj=aea +lat_1=50 +lat_2=58.5 +lat_0=45 +lon_0=-126 +x_0=1000000 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs")
#ptplyid = []
#shp = fiona.open(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\seagrass_working\seagrass_working_erase.shp')
#start = time.time()
#for i in range(len(lons)):
#    lonx,laty = transform(inproj, outproj, lons[i],lats[i])
#    p = Point(lonx,laty)
#    for poly in shp:
#        f = shape(poly["geometry"])
#        if p.within(f):
#            ptplyid.append(poly["properties"]["uID"])
#            break
#end = time.time()
#print(end - start)
# fiona and shapely - 42.6029999256

# geopandas approach
start = time.time()
traj = dataset.variables["trajectory"]
poly  = geopandas.GeoDataFrame.from_file(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\seagrass_working\seagrass_working_erase.shp')
poly.crs = {'init' :'epsg:3005'}
df = pd.DataFrame()
df['o_coords'] = list(zip(lons, lats))
df['o_coords'] = df['o_coords'].apply(Point)
df['traj_id'] = list(traj)
points = geopandas.GeoDataFrame(df, geometry='o_coords')
points.crs = {'init' :'epsg:4326'}
points = points.to_crs({'init' :'epsg:3005'})
#print(gdf.head()) 
origin_ids = geopandas.tools.sjoin(points, poly, how='left')
origin = pd.DataFrame(data=origin_ids)
origin = origin[['o_coords','traj_id', 'uID']].copy()

# deal with points that didn't fall within polygons
shp = fiona.open(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\seagrass_working\seagrass_working_erase.shp')
for row in origin.itertuples(index=True):
    if math.isnan(row[3]):
        #print row
        index = row[0]
        before = origin["uID"][index-1]
        after = origin["uID"][index+1]
        # most points should fall into the first if statement
        # the remaining statements are to catch when multiple nan values in a row
        # or if the nan value is the first or last point in that polygon
        if before == after:
            origin['uID'][index] = before
        elif math.isnan(before) or math.isnan(after): # the uID before or after is also nan
            i = 1
            eureka = True
            while eureka:
                if not math.isnan(origin["uID"][index-i]):
                    before = origin["uID"][index-i]
                    eureka = False
                i += 1
            i = 1
            eureka = True
            while eureka:
                if not math.isnan(origin["uID"][index+i]):
                    after = origin["uID"][index+i]
                    eureka = False
                i += 1
            point = row[1]
            for poly in shp:
                if poly['properties']['uID'] == before:
                    f_before = shape(poly["geometry"])
                if poly['properties']['uID'] == after:
                    f_after = shape(poly["geometry"])
            distance_b = point.distance(f_before)
            distance_a = point.distance(f_after)
            if distance_b < distance_a:
                origin['uID'][index] = before
            else:
                origin['uID'][index] = after
        else: # if the uID before and after are different
            point = row[1]
            for poly in shp:
                if poly['properties']['uID'] == before:
                    f_before = shape(poly["geometry"])
                if poly['properties']['uID'] == after:
                    f_after = shape(poly["geometry"])
            distance_b = point.distance(f_before)
            distance_a = point.distance(f_after)
            if distance_b < distance_a:
                origin['uID'][index] = before
            else:
                origin['uID'][index] = after
origin.uID = origin.uID.astype('int64')
end = time.time()
print (end - start)




###################
# settle when over a seagrass patch
###################
start = time.time()
poly  = geopandas.GeoDataFrame.from_file(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\seagrass_working\seagrass_working_erase.shp')
poly.crs = {'init' :'epsg:3005'}
dest_df = pd.DataFrame(columns=['d_coords','traj_id','dest_id','time_int'])
timestep = dataset.variables["time"]
status = dataset.variables["status"]
lon = dataset.variables["lon"]
lat = dataset.variables["lat"]
trajs = dataset.variables["trajectory"]
pd_i = 0

for i in range(1,len(timestep)): # don't need first timestep since nothing will have moved
    t = trajs[np.where((status[:,[i]] == 0) | ((status[:,[i]] == 1) & (status[:,[i-1]] == 0)))[0]]
    for p in dest_df["traj_id"]:
        if p in t:
            index = np.argwhere(t==p)
            t = np.delete(t, index)
    lons = []
    lats = []
    for par in t:
        index = par - 1  # t is the actual trajectory ID, the index for that value is 1 less
        lons.append(lon[index][i])
        lats.append(lat[index][i])
    # now do the spatial join
    df = pd.DataFrame()
    df['d_coords'] = list(zip(lons, lats))
    df['d_coords'] = df['d_coords'].apply(Point)
    df['par_id'] = list(t)
    points = geopandas.GeoDataFrame(df, geometry='d_coords')
    points.crs = {'init' :'epsg:4326'}
    points = points.to_crs({'init' :'epsg:3005'})
    pointInPolys = geopandas.tools.sjoin(points, poly, how='inner')
    for row in pointInPolys.itertuples(index=False):
        dest_df.loc[pd_i] = [row[0],row[1],row[6],i]
        pd_i += 1
dest_df.traj_id = dest_df.traj_id.astype('int64')
dest_df.dest_id = dest_df.dest_id.astype('int64')
dest_df.time_int = dest_df.time_int.astype('int64')
end = time.time()
print (end - start)

# join the two tables. The resulting data frame are the particles that settled in another patch
# to get all particles together add:  how='outer'
origin_dest = dest_df.merge(origin, on='traj_id', how='inner')

# check which ones settled on another patch
for row in origin_dest.itertuples(index=False):
    if row[2] != row[5]:
        print row

# TO DO:
# MOVE TO NEW FILE, PUT INTO FUNCTIONS and NOTES THROUGHOUT - do this sooner than later, clean up
# for mortality - create copy of netcdf at beginning with today's date
# implement mortality before settlement. Perhaps best way is to change status to stranded in the array and not in the netcdf.
    # for each time step (cumulative day) - deactivate a certain percentage of particles
    # see Connolly and Baird 2010 for a Weibull distribution (as referenced in Seascape Ecology book)
    # should I do this by seagrass patch?
# DO SOME THOROUGH VALIDATION and QUESTIONING
# is this the best way to implement settlement? Are they all just going to settle on their home patch?
    # yes it looks that way. perhaps I need to implement precompetency to handle this
# THIS IS WHERE I SHOULD OUTPUT THE RESULTS (use Whalen script) and consider what is going on





#TESTING
# this is "take of all of the particles and return the lon of the first time step for each"
lon[:,[0]]
# now get ones that are active
x = (lon[:,[0]])[np.where(status[:,[0]] == 0)]
len(x)
# wow this works. So this is taking the longitudes where the status is active
# np.where returns the indexes of those true locations
# now check ones that were JUST stranded
x = (lon[:,[1]])[np.where((status[:,[1]] == 1) & (status[:,[0]] == 0))]
len(x)
# now put those together
x = (lon[:,[1]])[np.where((status[:,[1]] == 0) | ((status[:,[1]] == 1) & (status[:,[1-1]] == 0)))]
len(x)
# we will also need to do the same select from the "trajectories" so that we can maintain the id to insert into the dictionary later
t = trajs[np.where((status[:,[1]] == 0) | ((status[:,[1]] == 1) & (status[:,[1-1]] == 0)))[0]] # the extra [0] this is because the np.where returns a 2D array but trajs is 1D
len(t)
# wait... why do I even need to do lat/lon until the last step if I have the traj numbers
x = (lon[:,[1]])[np.where(t)] # this doesn't work
# once I have my list of removed values (from the pd df below), I need to figure out how to use those traj ids to reference the lat/lon
# ok I think I got this:
# go way back to when I put together my list of lons
# go through each value in t
# then use the value as the index to pull the lon and lat

# check if already in array and remove
# I don't think this is the best option because numpy arrays are copied each time they are appended to and you can't really create a blank one without allocating space
#desties = np.array([[1,8,13134235,32],[2,7,2342324,41]])
#desties_i = desties[:,0]
#for tr in desties_i:
#    if tr in t:
#        index = np.argwhere(t==tr)
#        t = np.delete(t, index)

# try now with dictionary
d = {}
start = time.time()
d[1] = {}
d[1] = {
    'dest_id' : 8,
    'time' : 24525,
    'time_interval' : 23
    }
for p in d:
    if p in t:
        index = np.argwhere(t==p)
        t = np.delete(t, index)
end = time.time()
print (end - start)

# now try with pandas df
dest_df = pd.DataFrame(columns=['traj','dest_id','time','time_interval'])
start = time.time()
dest_df.loc[0] = [1,53456,13,5]
dest_df.loc[1] = [2,53456,13,5]
for pa in dest_df["traj"]:
    if pa in t:
        print pa
        index = np.argwhere(t==pa)
        print index
        t = np.delete(t, index)
end = time.time()
print (end - start)
# pandas slightly faster
# will also be easier to work with in the end