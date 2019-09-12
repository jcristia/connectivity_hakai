# delete seagrass meadows

import arcpy
import os

arcpy.env.overwriteOutput = True
sg_folder = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\seagrass\seagrass_all'
sg_main_gdb = 'seagrass.gdb'
sg_fc = 'seagrass_all_6NEAR'
sg_fc_next = 'seagrass_all_7MARKSMALL'
arcpy.env.workspace = os.path.join(sg_folder)

if not arcpy.Exists(os.path.join(sg_main_gdb, sg_fc_next)):
    arcpy.FeatureClassToFeatureClass_conversion(os.path.join(sg_main_gdb, sg_fc), sg_main_gdb, sg_fc_next)

arcpy.env.workspace = os.path.join(sg_folder, sg_main_gdb)

# create field if it doesn't exist
fields = arcpy.ListFields(sg_fc_next)
fnames = []
for field in fields:
    fnames.append(field.name)
if "delete_patch" not in fnames:
    arcpy.AddField_management(sg_fc_next, 'delete_patch', 'TEXT')

# if less than 500m2 and near distance less than 1000m, mark for deletion
with arcpy.da.UpdateCursor(sg_fc_next, ['Shape_Area', 'NEAR_DIST', 'delete_patch']) as cursor:
    for row in cursor:
        if (row[0] < 1000 and row[1] < 1000):
            row[2] = "y"
        cursor.updateRow(row)


# issue now is that if I just run the next iteration, some of the near features are also ones that were already slated for deletion on the previous step. I should remvove these first then run near again.
# copy to new layer withouth 'y' features
if not arcpy.Exists('seagrass_all_8DELSMALL'):
    delimitedField = arcpy.AddFieldDelimiters(arcpy.env.workspace, 'delete_patch')
    expression = delimitedField + " NOT IN ('y')"
    arcpy.FeatureClassToFeatureClass_conversion(sg_fc_next, os.path.join(sg_folder,sg_main_gdb), 'seagrass_all_8DELSMALL', expression)
# delete near fields
arcpy.DeleteField_management('seagrass_all_8DELSMALL', ['ORIG_FID','NEAR_FID','NEAR_DIST'])
# run near again
arcpy.Near_analysis('seagrass_all_8DELSMALL','seagrass_all_8DELSMALL')


# if less than 30,000m2 and near distance less than 1000m, and if near feature is greater than 30,000m2, then mark for deletion
with arcpy.da.UpdateCursor('seagrass_all_8DELSMALL', ['NEAR_FID','Shape_Area', 'NEAR_DIST', 'delete_patch']) as cursor:
    for row in cursor:
        if (row[1] < 30000 and row[2] < 1000):
            where = """OBJECTID = {}""".format(row[0])
            with arcpy.da.SearchCursor('seagrass_all_8DELSMALL', ['OBJECTID', 'Shape_Area'], where) as subcursor:
                for subrow in subcursor:
                    if subrow[1] >= 10000:
                        row[3] = 'y'
                        cursor.updateRow(row)

# copy to new layer withouth 'y' features
if not arcpy.Exists('seagrass_all_9DELMED'):
    delimitedField = arcpy.AddFieldDelimiters(arcpy.env.workspace, 'delete_patch')
    expression = delimitedField + " NOT IN ('y')"
    arcpy.FeatureClassToFeatureClass_conversion('seagrass_all_8DELSMALL', os.path.join(sg_folder,sg_main_gdb), 'seagrass_all_9DELMED', expression)
# delete near fields
arcpy.DeleteField_management('seagrass_all_9DELMED', ['NEAR_FID','NEAR_DIST'])
# run near again
arcpy.Near_analysis('seagrass_all_9DELMED','seagrass_all_9DELMED')

#if near feature less than 30,000m2 AND near feature does not have ....


#hmmm perhaps relax this further: go back to first loop. If less than 2,000 and less than 1000 near, delete. For second loop, second part, if near features is >= 10000, then delete.

