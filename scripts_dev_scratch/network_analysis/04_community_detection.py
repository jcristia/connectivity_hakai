# community detection using leidenalg (Leiden algorithm), which is an improved version of louvain
# after running this, you need to run the community_detection_join.py script

# reference Traag et al. 2019 for notes and highlights

# plotting in igraph doesn't work in visual studio
# will work from command line though

# packages and install instructions for leidenalg and igraph:
# https://anaconda.org/vtraag/repo

# environment: community_detection

# Decision for which leiden method to use:
# Remove self connections. Once, I did this the communities changed drastically. It was clear that the large probabilities of self-connections were swamping out dispersal connections. However, its important to note that this is the real case - most individuals will only move about in their own meadow, but since I am trying to look at dispersal, I don't want to consider limited movement here.
# Use la.ModularityVertexPartition. This is the function that is guaranteed to be optimized (hence why there is no resolution parameter to alter). In the Leiden paper and in other papers, the author notes that just maximizing the modularity score shouldn't be the main goal. A maximum score can still have non-optimal communities (see example in paper). The goal of leiden is to find optimal communities, not necessarily maximum modularity (although for small graphs, these may be the same).
# I did some tests with the CPM function and I can get similar communities, but it will be hard to justify why I used the values I did. If I just use the default Leiden function (the one used in the introduction examples), then there is less I have to defend.



import igraph as ig
import leidenalg as la
import pandas as pd
import geopandas as gp
import os
import matplotlib.pyplot as plt
import seaborn as sns


#################
# Examples of Leidenag use
#################

## example from the leidenalg documentation:
#G = ig.Graph.Famous('Zachary')
#partition = la.find_partition(G, la.ModularityVertexPartition)
#partition.modularity
## Modularity is the fraction of the edges that fall within the given groups minus the expected fraction if edges were distributed at random. The value of the modularity lies in the range [-1,1].It is positive if the number of edges within groups exceeds the number expected on the basis of chance.
#partition.summary()
#partition.quality()
##ig.plot(partition)
## plotting in igraph doesn't work in visual studio
## will work from command line though

## Although this is the optimal partition, it does not correspond to the split in two factions that was observed for this particular network. We can uncover that split in two using a different method
#partition = la.find_partition(G, la.CPMVertexPartition, resolution_parameter = 0.1)
#partition.modularity
#partition.summary()
#partition.quality()
## I'm not sure I understand the difference between modularity and quality, but if you play around with the resolution_parameter, you can see that Leiden finds the highest value you can get for modularity.
## So based on the Louvain documentation, I think the quality value is calculated differently based on which parition method you are using, so perhaps I shouldn't be comparing those values. 
##ig.plot(partition)

## example of how to look at membership
#partition.membership[10]
#for p in partition.membership:
#    print(p)



#################
# User input
#################

root = r'D:\Hakai\script_runs\seagrass\seagrass_20200327_SS201408'

shp_conn = r'shp_merged\connectivity_average.shp' #connectivity average lines shapefile
shp_pts = r'conefor\conefor_connectivity_average\conefor_metrics.shp' #output from the conefor script
out_shp = r'shp_merged\patch_centroids_metrics_commavg.shp' #output pt shapefile that will contain conefor metrics and community membership as attributes


#################
# Setup and format
#################

shp_conn = os.path.join(root, shp_conn)
shp_pts = os.path.join(root, shp_pts)
out_shp = os.path.join(root, out_shp)

df = gp.read_file(shp_conn)
df = df.drop(columns=['geometry','time_int','date_start', 'totalori', 'date_start'])
df_pts = gp.read_file(shp_pts)

# CODE BELOW KEPT FOR REFERENCE
## so it looks like leidenalg is going to renumber from_id based on the number of vertices with connections. Therefore, if my list is not complete (a meadow with no connections and no self-connections, so it isn't in the connectivity shapefile), then it screws up the numbering in leidenalg.
## So in pandas I will need to fix this to include every connection, but just make it a 0 probability.
## ACTUALLY, the indexing it uses is different than the vertice names regardless, so the above doesn't matter, but I will leave this code in anyway to make it complete.
## https://stackoverflow.com/questions/45726522/whats-the-best-way-to-tell-the-missing-row-in-pandas-dataframe 
#seq = pd.DataFrame(np.arange(df_pts.iloc[0].uID, df_pts.iloc[-1].uID))
#seq_miss_from = seq[~seq[0].isin(df.from_id)]
## now I should also check to_id, since a patch may only be present there
#seq_miss_to = seq[~seq[0].isin(df.to_id)]
## I would then take the values that are in both lists
#seq_missing = pd.merge(seq_miss_from, seq_miss_to, on=[0], how='inner')
## append in the missing IDs
#for miss in seq_missing[0]:
#    df = df.append({'from_id': miss, 'to_id': miss, 'prob': 0}, ignore_index=True)
#df = df.sort_values(by=['from_id'])


# igraph doesn't have a way to read a pandas df directly, must create tuples
tuples = []
for x in df.values:
    t = tuple([int(x[0]), int(x[1]), x[2]])
    tuples.append(t)
G = ig.Graph.TupleList(tuples, directed = True, edge_attrs = ['prob'])
#G.edge_attributes()
#G.is_weighted()
#G.es['prob']
G.es['weight'] = G.es['prob']
G.is_weighted()

# Create a version of the dataframe and graph with self connections removed
df_noselfconn = df[df['from_id'] != df['to_id']]
tuples = []
for x in df_noselfconn.values:
    t = tuple([int(x[0]), int(x[1]), x[2]])
    tuples.append(t)
G_noselfconn = ig.Graph.TupleList(tuples, directed = True, edge_attrs = ['prob'])
G_noselfconn.es['weight'] = G_noselfconn.es['prob']



#################
# Community detection
#################

# using igraphs clustering algorithm. Results are similar to leidenalg
#test1 = ig.Graph.community_infomap(G,edge_weights='weight')
#test1.modularity

partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight', seed=11)
# seed allows you to set the start of the random number generator. Setting this ensures you get the same clusters when you rerun it (I think).
partition.summary()
partition.modularity

# run on graph with self connections removed
partition_noselfconn = la.find_partition(G_noselfconn, la.ModularityVertexPartition, weights='weight', seed=11)
partition_noselfconn.summary()
partition_noselfconn.modularity


# community detection with CPM method
# so that I can alter the resolution paramter
# earlier test show that not including self conn is the best, so just use that df here
# resolution parameter is based on my own manual tests of rerunning this and altering the paramter and checking modularity to get the highest score.
# However, Traag warned that modularity is not necessarily the best metric. It may still best to use the previous method since it is not that much different.
partition_noselfconn_cpm = la.find_partition(G_noselfconn, la.CPMVertexPartition, weights='weight', seed=11, resolution_parameter=0.000008)
partition_noselfconn_cpm.summary()
partition_noselfconn_cpm.modularity



#################
# Format outputs
#################

# examples of indexing:
#partition[0] # this will list all the nodes that are part of the first community
#partition[0][1] # this gets the first node of the first community
#partition.membership[1364] # this will return the community number of 1364 node

# This prints out the VERTICE NAMES in each community.
#print(partition)
# NOTE: this is different than partition[0] which will print out the ARBITRARY VERTICE INDEXES that leidenalg asigns, which I don't want.
# so to actually go through and get each community and vertice names, you have to use SUBGRAPH

df_c = pd.DataFrame(columns=['pid','comid'])
#for p in range(len(partition)):
#    for v in partition.subgraph(p).vs['name']:
#        df_c = df_c.append({'patch_id':v, 'community_id': p}, ignore_index=True)
for p in range(len(partition)):
    if len(partition.subgraph(p).vs['name'])>1:
        for v in partition.subgraph(p).vs['name']:
            df_c = df_c.append({'pid':v, 'comid': p}, ignore_index=True)


# create dataframe from community without self connections
df_c_noselfconn = pd.DataFrame(columns=['pidn','comidns'])
for p in range(len(partition_noselfconn)):
    if len(partition_noselfconn.subgraph(p).vs['name'])>1:
        for v in partition_noselfconn.subgraph(p).vs['name']:
            df_c_noselfconn = df_c_noselfconn.append({'pidn':v, 'comidns': p}, ignore_index=True)


df_c_noselfconn_cpm = pd.DataFrame(columns=['pidnc','comidnsc'])
for p in range(len(partition_noselfconn_cpm)):
    if len(partition_noselfconn_cpm.subgraph(p).vs['name'])>1:
        for v in partition_noselfconn_cpm.subgraph(p).vs['name']:
            df_c_noselfconn_cpm = df_c_noselfconn_cpm.append({'pidnc':v, 'comidnsc': p}, ignore_index=True)


#################
# Merge outputs
#################

# get geometry and attributes from original shapefile
gdf = gp.read_file(shp_pts)
# join
gdf_all = gdf.merge(df_c, left_on='uID', right_on='pid', how='outer')
gdf_all = gdf_all.merge(df_c_noselfconn, left_on='uID', right_on='pidn', how='outer')
gdf_all = gdf_all.merge(df_c_noselfconn_cpm, left_on='uID', right_on='pidnc', how='outer')

gdf_all.to_file(filename=out_shp, driver='ESRI Shapefile')


###############
# TEST CPM WITH DIFFERENT RESOLUTION VALUES
###############

#res = [0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
#mod = []
#for r in res:
#    partition_noselfconn_cpm = la.find_partition(G_noselfconn, la.CPMVertexPartition, weights='weight', seed=11, resolution_parameter=r)
#    partition_noselfconn_cpm.summary()
#    partition_noselfconn_cpm.modularity
#    mod.append(partition_noselfconn_cpm.modularity)

#fig, ax = plt.subplots()
#ax.set(xscale="log")
#sns.set()
#sb = sns.pointplot(x=res, y=mod, ax=ax, ci=None)
#sb.set(xlabel='CPM resolution parameter', ylabel='modularity')
#plt.show()