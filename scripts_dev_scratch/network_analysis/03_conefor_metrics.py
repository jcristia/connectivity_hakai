# use conefor directed command line to calculate dPC

# NOTE: do not run this script all at once. Read instructions throughout.

# The directed version of conefor only has a windows version right now so I can't run it on the cluster. I could set it up to run it in the computer lab though so that it doesn't take up my computer for 5 days.

import os
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
from math import log10, floor

# input paths
root = r'D:\Hakai\script_runs\seagrass\seagrass_20200327_SS201408'
conn_shp = r'shp_merged\connectivity_pld60.shp'
seagrass_cent = r'shp_merged\patch_centroids.shp'
# output paths
main_out_folder = r'conefor'
ind_out_folder = r'conefor_{}'
nodefile = r'nodes_conefor.txt'
connections_file = r'conns_conefor.txt'
out_shp = r'conefor_metrics.shp'

# DECISION: my default is to log transform and normalize, so keep these values set as True
# Refer to "CONEFOR metrics and log transforming areas" note in Evernote for justification
# Main points:
# spans 8 orders of magnitude.
# dispersal probs span 5 orders of magnitude.
# Area and intraconnectivity is not my main focus.
# Also PC is based on the habitat availability concept. This is assuming that if everything is connected then you only need to focus on area. However, some of my dispersal probabilities are so low, that even though I'm showing that everything is connected, there is a lot of uncertainty, so I don't want to just say I should conserve the biggest patches because everything is connected. Also, some of my biggest patches don't have a lot of connections going in and out.
# Also log transform because we are thinking of multigenerational movement and not an animal that can choose to stay in a nice patch.
# Also these animals have a PD phase and movement is necessary.
# We are dealing with tiny tiny invertebrates, not something large like elk. A small patch is still adequate habitat, and bigger size is probably only better up to a certain point.
normalize = True
log_transform = True # only set true if normalize also true
# NOTE: once I join the resulting node importances back to the node shapefile, the transformed numbers don't carry forward, so it's hard to remember that I did this.

#######################
# set up file paths, create folder

# main conefor folder
out_folder = os.path.join(root, main_out_folder)
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# individual run conefor folder
ind_out_folder = ind_out_folder.format(os.path.splitext(os.path.basename(conn_shp))[0])
out_folder = os.path.join(root, main_out_folder, ind_out_folder)
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

conn_shp = os.path.join(root, conn_shp)
seagrass_cent = os.path.join(root, seagrass_cent)
nodefile = os.path.join(root, out_folder, nodefile)
connections_file = os.path.join(root, out_folder, connections_file)


#######################
# convert shapefiles to correct conefor table format

df_nodes = gp.read_file(seagrass_cent)
if not normalize: # use area as is
    df_nodes.to_csv(nodefile, sep='\t', columns=['uID', 'area'], header=False, index=False)
elif not log_transform: # normalize but don't transform
    max_area = df_nodes['area'].max()
    min_area = df_nodes['area'].min()
    min_area = 0 # normalize, but use 0 as min
    df_nodes['area_norm'] = (df_nodes['area']-min_area) / (max_area - min_area)
    df_nodes = df_nodes.drop(columns=['area'])
    df_nodes.to_csv(nodefile, sep='\t', columns=['uID', 'area_norm'], header=False, index=False)
else: # transform and normalize
    df_nodes['area_log'] = np.log10(df_nodes['area'])
    max_area = df_nodes['area_log'].max()
    min_area = df_nodes['area_log'].min()
    min_area = 0 # normalize, but use 0 as min
    df_nodes['area_norm'] = (df_nodes['area_log']-min_area) / (max_area - min_area)
    df_nodes = df_nodes.drop(columns=['area', 'area_log'])
    df_nodes.to_csv(nodefile, sep='\t', columns=['uID', 'area_norm'], header=False, index=False)

# plot (testing purposes)
#fig, ax = plt.subplots()
##ax.set(xscale='log')
##min = df_nodes.area_norm.min()
##min_round = 10**(int(floor(log10(abs(min)))))
##bins = np.logspace(np.log10(min_round),np.log10(1.0), 50)
#ax = df_nodes['area_norm'].plot.hist()
#plt.show()

df_conn = gp.read_file(conn_shp)
if 'prob_avg' in df_conn.columns:
    df_conn.to_csv(connections_file, sep='\t', columns=['from_id', 'to_id', 'prob_avg'], header=False, index=False)
elif 'prob' in df_conn.columns:
    df_conn.to_csv(connections_file, sep='\t', columns=['from_id', 'to_id', 'prob'], header=False, index=False)


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
# add in interconnectivity attribute, which is flux + connector
df_node_imp['dPCinter'] = df_node_imp['dPCflux'] + df_node_imp['dPCconnector']
# get geometry and attributes from original shapefile
gdf = gp.read_file(seagrass_cent)
# join to get geomtery
gdf_imp = gdf.merge(df_node_imp, left_on='uID', right_on='Node')
gdf_imp.to_file(filename=os.path.join(out_folder, out_shp), driver='ESRI Shapefile')
