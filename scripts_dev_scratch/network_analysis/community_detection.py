# community detection using leidenalg, which is an improved version of louvain
# after running this, you need to run the community_detection_join.py script

# plotting in igraph doesn't work in visual studio
# will work from command line though

# packages and install instructions for leidenalg and igraph:
# https://anaconda.org/vtraag/repo

# environment: community_detection

import igraph as ig
import leidenalg as la
import pandas as pd
import geopandas as gp
import fiona
import numpy as np

# example from the leidenalg documentation:
#G = ig.Graph.Famous('Zachary')
#partition = la.find_partition(G, la.ModularityVertexPartition)
#ig.plot(partition)
#partition.membership[10]
#for p in partition.membership:
#    print(p)

shp_conn = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output\seagrass_20190629\connectivity.shp'
shp_pts = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output\seagrass_20190629\patch_centroids.shp'
df = gp.read_file(shp_conn)
df = df.drop(columns=['geometry','time_int','date_start','quantity','totalori'])
df_pts = gp.read_file(shp_pts)

# so it looks like leidenalg is going to renumber from_id based on the number of vertices with connections. Therefore, if my list is not complete (a meadow with no connections and no self-connections, so it isn't in the connectivity shapefile), then it screws up the numbering in leidenalg.
# SO... in pandas I will need to fix this to include every connection, but just make it a 0 probability.
# ACTUALLY, the indexing it uses is different than the vertice names regardless, so the above doesn't matter as much, but I will leave this code in anyway to make it complete
# https://stackoverflow.com/questions/45726522/whats-the-best-way-to-tell-the-missing-row-in-pandas-dataframe 
seq = pd.DataFrame(np.arange(df_pts.iloc[0].uID, df_pts.iloc[-1].uID))
seq_miss_from = seq[~seq[0].isin(df.from_id)]
# now I should also check to_id, since a patch may only be present there
seq_miss_to = seq[~seq[0].isin(df.to_id)]
# I would then take the values that are in both lists
seq_missing = pd.merge(seq_miss_from, seq_miss_to, on=[0], how='inner')
# append in the missing IDs
for miss in seq_missing[0]:
    df = df.append({'from_id': miss, 'to_id': miss, 'prob': 0}, ignore_index=True)
df = df.sort_values(by=['from_id'])

# igraph doesn't have a way to read a pandas df directly
tuples = []
for x in df.values:
    t = tuple([int(x[0]), int(x[1]), x[2]])
    tuples.append(t)
#tuples = [tuple(x) for x in df.values]
G = ig.Graph.TupleList(tuples, directed = True, edge_attrs = ['prob'])
#G.edge_attributes()
#G.is_weighted()
#G.es['prob']
G.es['weight'] = G.es['prob']
G.is_weighted()

# using igraphs clustering algorithm. Results are similar to leidenalg
#test1 = ig.Graph.community_infomap(G,edge_weights='weight')
#test1.modularity

partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight', seed=11)
# seed allows you to set the start of the random number generator. Setting this ensures you get the same clusters when you rerun it (I think).
partition.summary()
partition.quality()

# examples of indexing:
#partition[0] # this will list all the nodes that are part of the first community
#partition[0][1] # this gets the first node of the first community
#partition.membership[1364] # this will return the community number of 1364 node

# This prints out the VERTICE NAMES in each community.
# NOTE: this is different than partition[0] which will print out the arbitrary vertice indexes, which I don't want.
#print(partition)
# so that actually go through and get each community and vertice names, you have to use subgraph

df_c = pd.DataFrame(columns=['patch_id','community_id'])
#for p in range(len(partition)):
#    for v in partition.subgraph(p).vs['name']:
#        df_c = df_c.append({'patch_id':v, 'community_id': p}, ignore_index=True)
for p in range(len(partition)):
    if len(partition.subgraph(p).vs['name'])>1:
        for v in partition.subgraph(p).vs['name']:
            df_c = df_c.append({'patch_id':v, 'community_id': p}, ignore_index=True)
df_c.to_csv(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\df_c.csv', index=False)

# Not ideal, but now you need to run another script to join this to a feature class.
# Leidenalg requires python3, but I couldn't manage to get an install of arcpy pro into a conda environment.
# Done is better than perfect
