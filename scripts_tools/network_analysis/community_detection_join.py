# run this script after running community_detection.py

# environment: arcpy10-7
import arcpy
csv = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\df_c.csv'
gdb = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output\seagrass_20190629\metrics.gdb'
patch_shp = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output\seagrass_20190629\patch_centroids.shp'
arcpy.env.workspace = gdb
arcpy.env.overwriteOutput = True

arcpy.TableToTable_conversion(csv, gdb,'communities_table')
# doing a join requires a feature layer
# create join
# create a new feature from joined table
arcpy.MakeFeatureLayer_management (patch_shp, 'patches')
arcpy.AddJoin_management('patches', "uID", 'communities_table', "patch_id")
arcpy.CopyFeatures_management('patches', 'communities')

arcpy.Delete_management('communities_table')
# arcpy.Delete_management(csv)
arcpy.DeleteField_management('communities', drop_field=['communities_table_OBJECTID','communities_table_patch_id'])