# 

import netCDF4 as nc
import numpy as np
from shapely.geometry import shape, Point, Polygon
from shapely.ops import nearest_points
import fiona
import pandas as pd
import geopandas
import math
#import time


###################
# to set
###################

nc_output = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\output\run1_20190313.nc'

seagrass = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\seagrass_working\seagrass_working_erase.shp'
seagrass_buff = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts\biology_working\seagrass_working\seagrass_working_erase_buff20.shp' # buffered by 20m just for checking settlement. This is to account for that seagrass polys have slivers between coastline
seagrass_crs = {'init' :'epsg:3005'}

# if I am using 'stranding' in opendrift, then I likely need at least a small precompetency period because everything just ends up settling at home otherwise
# be careful setting this. It depends on the time_step and time_step_output you used in the run
# it is in units of the timestep output. If time step output is 30 minutes, then precomp of 2 is 1 hour
precomp = 4

# get these values from the simulation script
time_step_output = 0.5 # in hours
particles_per_release = 500
interval_of_release = 1 # in hours (as it's set up now, interval can't be less than time step output) (if no delayed release then just put same value as time_step_output)
num_of_releases = 24 # if no delayed release then just put 1


# should I make these global?
dataset = nc.Dataset(nc_output, "r+")
lon = dataset.variables["lon"]
lat = dataset.variables["lat"]
traj = dataset.variables["trajectory"]
status = dataset.variables["status"]
timestep = dataset.variables["time"]


###################
# attach poly uID to particle
###################

def get_particle_originPoly(seagrass, lon, lat, traj, seagrass_crs):

    # get starting lat/lon for each particle
    # particles start at different times, so check if it is masked, stop once we find our first value
    lons = []
    for i in range(len(lon)):
        for j in lon[i]:
            if not np.ma.is_masked(j):
                lons.append(j)
                break
    lats = []
    for i in range(len(lat)):
        for j in lat[i]:
            if not np.ma.is_masked(j):
                lats.append(j)
                break
    
    # check which polygon it seeds in
    poly  = geopandas.GeoDataFrame.from_file(seagrass)
    poly.crs = seagrass_crs
    df = pd.DataFrame()
    df['o_coords'] = list(zip(lons, lats))
    df['o_coords'] = df['o_coords'].apply(Point)
    df['traj_id'] = list(traj)
    points = geopandas.GeoDataFrame(df, geometry='o_coords')
    points.crs = {'init' :'epsg:4326'}
    points = points.to_crs(seagrass_crs)
    origin_ids = geopandas.tools.sjoin(points, poly, how='left')
    origin = pd.DataFrame(data=origin_ids)
    origin = origin[['o_coords','traj_id', 'uID']].copy()
    
    # deal with points that didn't fall within polygons
    # (some of the points were seeded just outside the polygons. I'm guessing there is some precision issues when they were seeded or with projecting. I think the safest option is to check them instead of changing opendrift code)
    shp = fiona.open(seagrass)
    for row in origin.itertuples(index=True):
        if math.isnan(row[3]):
            #print row
            index = row[0]
            before = origin["uID"][index-1]
            after = origin["uID"][index+1]
            # most points should fall into the first if statement (if the before and after uIDs are the same)
            # the remaining statements are to catch when there are multiple nan values in a row
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
    
    # I tried to make certain columns integers upon creation, but it didn't work
    origin.uID = origin.uID.astype('int64')

    return origin


###################
# calculate precompetency and release intervals
###################

def calc_precomp(precomp, time_step_output, particles_per_release, interval_of_release, num_of_releases, traj):

    timesteps_with_release = []
    precomp_end_timestep = []
    for release in range(num_of_releases):
        ts = (float(interval_of_release) / float(time_step_output)) * release
        timesteps_with_release.append(int(ts))

    precomp_end_timestep = []
    for release in timesteps_with_release:
        ts_e = release + precomp
        precomp_end_timestep.append(ts_e)

    precomp_range = []
    for p in precomp_end_timestep:
        precomp_range.append([p-precomp, p])

    particle_range = []
    if num_of_releases == 1:
        particle_range = [[1, len(traj) + 1]]
    else:
        for release in range(1,num_of_releases+1):
            p_range = [1 + ((release-1) * particles_per_release),(release * particles_per_release) +1]
            particle_range.append(p_range)

    return timesteps_with_release, precomp_end_timestep, precomp_range, particle_range


###################
# settle when over a seagrass patch
# account for precompetency period
###################

def settlement(origin, seagrass_buff, timestep, status, lon, lat, traj, seagrass_crs, precomp, precomp_range, particle_range):

    poly  = geopandas.GeoDataFrame.from_file(seagrass_buff)
    poly.crs = seagrass_crs
    dest_df = pd.DataFrame(columns=['d_coords','traj_id','dest_id','time_int'])
    pd_i = 0

    for i in range(1,len(timestep)):
        # get traj ids for particles that are active or where they were active on the previous step (just stranded)
        t_strand = traj[np.where((status[:,[i]] == 1) & (status[:,[i-1]] == 0))[0]]
        t_active = traj[np.where(status[:,[i]] == 0)[0]]
        
        # if we already settled it on a previous interation of the for loop then remove it from the list so we don't check it again
        for p in dest_df["traj_id"]:
            if p in t_strand:
                index = np.argwhere(t_strand==p)
                t_strand = np.delete(t_strand, index)
            if p in t_active:
                index = np.argwhere(t_active==p)
                t_active = np.delete(t_active, index)

        # this is where mortality should go
        # I should create a mortality list that includes all particles that are dead
        # then I should select all particles that are not on that list
        # remove ones that are already successfully settled
        # then I can apply mortality
        # add this selection to the mortality list
        # remove these particles from t_strand and t_active

        if precomp > 0: # remove from p_active ones that are in their precomp period
            for period in precomp_range:
                if i in range(period[0],period[1]):
                    period_index = precomp_range.index(period)
                    # get particles that are still in their precomp period
                    p_in_precomp = range(particle_range[period_index][0],particle_range[period_index][1])
                    for y in p_in_precomp:
                        if y in t_active:
                            index = np.argwhere(t_active==y)
                            t_active = np.delete(t_active, index)

        t = np.concatenate((t_strand,t_active))

        lons = []
        lats = []
        for par in t:
            index = par - 1  # t is the actual trajectory ID, the index for that value is 1 less
            lons.append(lon[index][i])
            lats.append(lat[index][i])
        df = pd.DataFrame()
        df['d_coords'] = list(zip(lons, lats))
        df['d_coords'] = df['d_coords'].apply(Point)
        df['par_id'] = list(t)
        points = geopandas.GeoDataFrame(df, geometry='d_coords')
        points.crs = {'init' :'epsg:4326'}
        points = points.to_crs(seagrass_crs)
        pointInPolys = geopandas.tools.sjoin(points, poly, how='inner')
        for row in pointInPolys.itertuples(index=False):
            dest_df.loc[pd_i] = [row[0],row[1],row[6],i]
            pd_i += 1
    dest_df.traj_id = dest_df.traj_id.astype('int64')
    dest_df.dest_id = dest_df.dest_id.astype('int64')
    dest_df.time_int = dest_df.time_int.astype('int64')
    
    # join the two tables
    # The resulting data frame is the particles that settled in another patch
    # to get all particles including the ones that did not settle change to:  how='outer'
    origin_dest = dest_df.merge(origin, on='traj_id', how='inner')

    return origin_dest



###################
# run
###################

origin = get_particle_originPoly(seagrass, lon, lat, traj, seagrass_crs)

if precomp == 0:
    timesteps_with_release = None
    precomp_end_timestep = None
    precomp_range = None
    particle_range = None
else:
    timesteps_with_release, precomp_end_timestep, precomp_range, particle_range = calc_precomp(precomp, time_step_output, particles_per_release, interval_of_release, num_of_releases, traj)

timesteps_with_release
precomp_end_timestep
precomp_range
particle_range



origin_dest = settlement(origin, seagrass_buff, timestep, status, lon, lat, traj, seagrass_crs, precomp, precomp_range, particle_range)

# quick check of which ones settled on a patch that is not their origin
for row in origin_dest.itertuples(index=False):
    if row[2] != row[5]:
        print row





# TO DO:
# for mortality - create copy of netcdf at beginning with today's date
# implement mortality before settlement. Perhaps best way is to change status to stranded in the array and not in the netcdf.
    # for each time step (cumulative day) - deactivate a certain percentage of particles
    # see Connolly and Baird 2010 for a Weibull distribution (as referenced in Seascape Ecology book)
    # also go back to that spreadsheet where I figured out how it calculated
    # should I do this by seagrass patch?
# DO SOME THOROUGH VALIDATION and QUESTIONING
# THIS IS WHERE I SHOULD OUTPUT THE RESULTS (use Whalen script) and consider what is going on