# Format outputs from biology_opendrift for input to connectivity metrics script.
# For runs where seagrass datasets were split up, there will be multiple output shapefiles that need to be merged together.

import os
import arcpy

# folder where individual shapefiles from biology_opendrift.py output are stored
shp_folder = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\hydro_models_format\outputs\shp'

# folder for merged shapefiles
merged_folder = os.path.join(shp_folder, 'shp_merged')
try:
    os.mkdir(merged_folder)
except:
    print("directory already exists")

# root names of input/output shapefiles
conn = 'connectivity'
pts = 'dest_biology_pts'
centroids = 'patch_centroids' # just need to copy this

# copy centroids in
arcpy.Copy_management(os.path.join(shp_folder, centroids + '.shp'), os.path.join(merged_folder, centroids + '.shp'))

# merge
sg_files = os.listdir(shp_folder)
conn_list = []
pts_list = []
for file in sg_files:
    if file.startswith(conn) and file.endswith('.shp'):
        conn_list.append(os.path.join(shp_folder, file))
    elif file.startswith(pts) and file.endswith('.shp'):
        pts_list.append(os.path.join(shp_folder, file))

arcpy.Merge_management(conn_list, os.path.join(merged_folder,conn + '.shp'))
arcpy.Merge_management(pts_list, os.path.join(merged_folder, pts + '.shp'))