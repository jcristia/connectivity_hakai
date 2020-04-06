#ISSUE:
#	* I noticed in the resulting connectivity lines that there was a very strong connection between meadows 361 and 360. This would require I very unlikely travel of a lot of particles through multiple channels. Also, it said the connection was made on the 1st time step.
#	* I looked at the destination points and everything around 361 said it came from 360. In fact, in the destination points shapefile there were no particles that came from 361.
#	* There were exactly 84 points in this area, indicating that it is a meadow that only seeds 1 particle per release.
#	* I looked at the starting points from the npy files and it showed that it is a meadow where the particle gets seeded just outside the shapefile.
#	* Therefore, in my biology code, its origin ID first gets coded as NA and then it goes through the series of steps I laid out for finding its origin patch. These steps rely on other particles from that patch being seeded with the same meadow, and then it just matches it to those. It does account for if it is the last particle for the patch, in which case the particle before and after it would be from different patches. In this case it uses distance between those two patches and assigns it to the closer patch.
#	* I realize now I wasn't accounting for the situation where only 1 particle was seeded from that patch and it was assigned NA (or a couple particles that were all NA). In this case, the particles with correct assignments before and after this particle are from completely different patches. When he goes to compare them it will assign this particle to the closest of those patches. That is how I am getting these strong connections between patches that are not likely to be connected.
#	* I went and checked all the destination points to find any uIDs that had 0 records. I thought that if it is just this one patch then I would maybe just ignore it and delete that connection. However, there are 11 patches with this issue. I checked the one with the largest area and it still only seeds 2 particles, of which both are in fact outside of their polygon. So that confirms the issue.
#SOLUTION:
#	* I really don't want to rerun all the biology scripts and conefor. I thought that I could just change the point assignment, but that would still be writing code and I know I wouldn't feel great about that. Either way, I need to write the fix in the biology code for any future runs, so I will just rerun everything. I think I can get it all done in less than a week.
#	* In biology.py, in get_particle_origin_poly function, for any NA particle
#		* if before and after are the same then assign it
#		* in elif, once I get the before and after meadows, check if they are exactly 1 apart, and if so, continue as is. If they are not, then get the number(s) of the meadows in between and get the distances, then assign it whichever one is lowest. (plan for the scenario where there are multiple 1 particle meadows in a row)
#		* in else, will also need to do the same thing as in the elif

# THIS SCRIPT WAS JUST TO INVESTIGATE THE ABOVE ISSUE

import os
import pandas as pd
import geopandas as gp
import numpy as np
from shapely.geometry import shape, Point, LineString, Polygon


dir = 'seagrass_20200228_SS201701'

root = r'D:\Hakai\script_runs\seagrass'
pts = 'dest_biology_pts_sg{}.shp'
subfolders = 9
subfolder = 'seagrass_{}'

seagrass_crs = {'init' :'epsg:3005'}

dest_paths = []
for sub in range(1, subfolders + 1):
    dest_pt = os.path.join(root, dir, subfolder.format(sub), 'outputs\\shp', pts.format(sub))
    dest_paths.append(dest_pt)

df = pd.DataFrame()
for dest_pt in dest_paths:
    gdf = gp.read_file(dest_pt)
    gdf = gdf.drop(columns=['geometry','date_start'])
    df = df.append(gdf)

# THIS IS THE LIST OF MEADOWS WITH THE ISSUE
i=1
while i <= 970:
    df_test = df[df.uID == i]
    if len(df_test) == 0:
        print(i)
    i+=1


# LOOK AT STARTING POINTS FOR MEADOW 103
# bring in starting points of all pts from npy files
latnpy = r'D:\Hakai\script_runs\seagrass\seagrass_20200228_SS201701\seagrass_1\outputs\lat_sg1.npy'
lonnpy = r'D:\Hakai\script_runs\seagrass\seagrass_20200228_SS201701\seagrass_1\outputs\lon_sg1.npy'

lons = np.load(lonnpy)
lats = np.load(latnpy)

df = pd.DataFrame()
df['o_coords'] = list(zip(lons, lats))
df['o_coords'] = df['o_coords'].apply(Point)
df['traj_id'] = list(range(1,330624+1))
points = gp.GeoDataFrame(df, geometry='o_coords')
points.crs = {'init' :'epsg:4326'}
points = points.to_crs(seagrass_crs)
outshp = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs\seagrass\seagrass\seagrass_20200403_ISSUE'
points.to_file(filename=outshp, driver='ESRI Shapefile')

# LOOK AT POINTS IN ARCGIS AND YOU CAN SEE THE STARTING POINTS FOR MEADOW 103