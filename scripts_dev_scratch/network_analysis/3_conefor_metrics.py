# use conefor directed command line to calculate dPC

# NOTE: do not run this script all at once. Read instructions throughout.

# The directed version of conefor only has a windows version right now so I can't run it on the cluster. I could set it up to run it in the computer lab though so that it doesn't take up my computer for 5 days.

import os
import pandas as pd
import geopandas as gp

# inputs / outputs
root = r'D:\Hakai\script_runs\seagrass\seagrass_20200228_SS201701'
seagrass_cent = r'shp_merged\patch_centroids.shp'
out_folder = r'conefor'
conn_shp = r'shp_merged\connectivity_average.shp'
nodefile = r'nodes_conefor.txt'
connections_file = r'conns_conefor.txt'
out_shp = r'conefor_metrics.shp'




#######################
# set up file paths, create folder
seagrass_cent = os.path.join(root, seagrass_cent)
out_folder = os.path.join(root, out_folder)
conn_shp = os.path.join(root, conn_shp)
nodefile = os.path.join(root, out_folder, nodefile)
connections_file = os.path.join(root, out_folder, connections_file)
if not os.path.exists(out_folder):
    os.mkdir(out_folder)


#######################
# convert shapefiles to correct conefor table format

df_nodes = gp.read_file(seagrass_cent)
df_nodes.to_csv(nodefile, sep='\t', columns=['uID', 'area'], header=False, index=False)

df_conn = gp.read_file(conn_shp)
df_conn.to_csv(connections_file, sep='\t', columns=['from_id', 'to_id', 'prob_avg'], header=False, index=False)


#######################
# run Conefor PC calculations

# NOTE: if you send the command using os.system(command), then nothing prints to the screen.
# Therefore, it is best to do this manually. Print out and copy and the command and run it
# in the command line separately.

# the Conefor output texts files are saved in whichever directory I am in on the command line. Therefore, print out_folder and cd directory to it before running the above command.
# if using the windows command line, first simply type 'd:' to change to that drive (don't need to use cd)
print('''cd {}'''.format(out_folder.replace('\\', '/')))
command = '''conefor.exe -nodeFile {} -conFile {} -t prob -PC -BCPC -pcHeur 0.0'''.format(nodefile, connections_file)
print(command)
# Can change pcHeur to a minimum threshold connecton to consider
# If in the future I want to batch process mutliple text files, refer back to the conefore examples. There is a way to easily do this.

#os.system(command) # this works but doesn't print anything to screen
#import subprocess
#out = subprocess.check_output(command)
#print(out)
# NOTE: the above code works, however it does not print the output continuously. Since the process takes a really long time, it may be nice to have this printed to screen.
# Therefore, I should just accept that I will do this step manually from the command line.


########################
# put output back into shapefiles

node_importances = os.path.join(out_folder, "node_importances.txt")
df_node_imp = pd.read_csv(node_importances, sep='\t')
df_node_imp = df_node_imp.drop(['Unnamed: 11'], axis=1)
# get geometry and attributes from original shapefile
gdf = gp.read_file(seagrass_cent)
# join to get geomtery
gdf_imp = gdf.merge(df_node_imp, left_on='uID', right_on='Node')
gdf_imp.to_file(filename=os.path.join(out_folder, out_shp), driver='ESRI Shapefile')
