
# temporal community detection

# I am keeping a lot of notes in here. Mainly ones I made for testing and as I was learning what all of this meant.

import igraph as ig
import leidenalg as la
import pandas as pd
import geopandas as gp
import os
from shapely.geometry import Polygon, Point
import shapely.affinity
from math import atan2, degrees
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#################
# User input
#################

root = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs_cluster\seagrass'
dirs = [
    'seagrass_20200310_SS201101',
    'seagrass_20200310_SS201105',
    'seagrass_20200310_SS201108',
    'seagrass_20200327_SS201401',
    'seagrass_20200327_SS201405',
    'seagrass_20200327_SS201408',
    'seagrass_20200228_SS201701',
    'seagrass_20200309_SS201705',
    'seagrass_20200309_SS201708',
    ]

shp_conn = r'shp_merged\connectivity_average.shp'
shp_pts = r'shp_merged\patch_centroids.shp'
out_shp = r'output_figs_SALISHSEA_ALL\Communities\{}_communities.shp'
out_poly = r'output_figs_SALISHSEA_ALL\Communities\{}_patch_clusters_convexhull.shp'

#################
# Setup and format
#################

df_pts = gp.read_file(os.path.join(root, dirs[0], shp_pts)) # just need to get this once
out_shp = os.path.join(root, out_shp)
out_poly = os.path.join(root, out_poly)

graphs = []
for dir in dirs:
    df = gp.read_file(os.path.join(root, dir, shp_conn))
    df = df.drop(columns=['geometry','time_int','date_start', 'totalori', 'date_start'])
    df = df[df['from_id'] != df['to_id']]
    tuples = []
    for x in df.values:
        t = tuple([int(x[0]), int(x[1]), x[2]])
        tuples.append(t)
    G = ig.Graph.TupleList(tuples, directed = True, edge_attrs = ['weight'])
    G.vs['id'] = G.vs['name']
    graphs.append(G)


#################
# Community detection EXPLORATION
#################

# This is the default method. However, you can't apply different interslice weights.
#membership, improvement = la.find_partition_temporal(graphs, la.ModularityVertexPartition, interslice_weight=0.1)
# membership is the community id of each node. To get the node id print out G.vs['id']

# Understanding multiplex structure:
# These functions are BUILDING/DETECTING the multiplex network. It has nothing to do with averaging community membership.
# Nodes can change membership in each slice and can therefore take on multiple community IDs. Think of it as overlapping convex hulls as the final product. This would represent overall communities and symbolize how they vary.

# The interslice link is having an effect on this. Compare membership[8] to doing just a single find_partition on the last graph:
# partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight', seed=11)
# you can see that they are different memberships

# Additional exploration:
# I was confused how an optimiser was acting on the partition object because it looked like the returned diff was just a number and partition wasn't changing (in traditional python-sense I would expect partition to be recreated as a new variable if it was going to change). However, you can see that partition does change.
# Print out partition before and after optimise to see.
#G = ig.Graph.Famous('Zachary')
#optimiser = la.Optimiser()
#partition = la.ModularityVertexPartition(G)
#diff = optimiser.optimise_partition(partition)
# if you look at the membership before, it first places each node into its own community, then the optimiser places them into optimal communities. You can also print out the quality before and after and see how it improves.
# that's why optimisers return the IMPROVEMENT OF QUALITY


#################
# Community detection
#################

# find_partition_temporal is a helper that packages up the optimiser and partition creation and makes the assumption to weight the interslices the same. However, if I want to weight the interslices differently then you need to do these steps manually. I would do this because there are different lengths of time between my slices and I want to connect between seasons different than within seasons.

# Set up interslice structure
G_coupling = ig.Graph.Formula("1--2--3--4--5--6--7--8--9, 1--4--7, 2--5--8, 3--6--9")
# The interslice layer is itself a graph and the individual graphs are nodes. The above formula first connects the graphs chronologically, and then the other three pieces connects similar seasons. This is so that I can weight within season connections higher, which will smooth them out and allow them to vary less, and then weigh between seasons lower so that they can vary more and I can draw out how communities change with season. The lower we set the weight, the more we let the difference in slices come out
# I've now changed this to be undirected as opposed to 1-->2-->3... . I think this makes more sense for time because 1 and 2 are equally related. I think you use direction when you actually have a directed movement like flow. Vincent suggested this.

## Weighting
#print(G_coupling) # you can see how it orders the edges here and how you will need to order your weights
# testing different weightings: find the threshold where it starts to change. Test this with ModularityVertexPartition.
# start with just one weight, then do different levels for within/between seasons
w = 1 # within
b = 0.00001 # between
G_coupling.es['weight'] = [b, w, b, w, b, w, b, w, b, w, b, w, b, b]

## Testing by looking at the number of resulting clusters
# with just one weight:
# 0 601 clusters
# 0.0000000001 91 clusters
# 0.000000001 93 clusters
# 0.00000001 96 clusters
# 0.0000001 91 clusters
# 0.000001 82 clusters
# 0.00001 68 clusters
# 0.0001 61 clusters
# 0.001 62 clusters
# 0.01 62 clusters
# 0.1 62 clusters
# 1 62 clusters
# 1000 63 clusters

# lock in 1 for WITHIN seasons, then test the range again
# 0.000000000000000000000001 167
# 0.000000000000000000001 171
# 0.000000000000000001 176
# 0.000000000000001 77
# 0.000000000001 76
# 0.000000001 77
# 0.00000001 77
# 0.0000001 75
# 0.000001 66
# 0.00001 65
# 0.0001 61

# there's not a clear threshold where clusters make a huge jump (except at very low values), but it stabilizes at 77 at very low numbers. I could pick the most extreme values (e.g. 1000000, 0.00000000001) in which case I get 171, but I'm not sure that make sense. I am essentially perfectly connecting within seasons and saying between seasons aren't connected at all. I should therefore choose 2 non-extreme values, but values that still create some difference.
# within: 1
# between: 0.00001


# I like this graph structure of distinguishing between/within seasons.
# notice how tsawwassen extends to mayne island, but there is overlap with 2 polygons on either side. This is exactly what i want to indicate. The point sticking out most at the port is in one community 6 times and another 3 times. These are the patterns I want to draw out. I could somewhat tell these where in the data just by looking at the seasonal variation in connectivity.
# I could write something like:
# I tested a range of interslice weightings and different arrangements of slices. I am interested in seasonal variation, and repetition across years is just to get more of an average for a season, not to look at yearly variation. Therefore, I arranged slices by season. I used a higher interslice weighting between slices of the same season to smooth out any differences in community structure, then I used a lower weighting between seasons to allow community structure to vary more so as to see if there are changing dynamics between seasons. The general structure is not sensitive to any weights above 0.0001. I tested all way up to 1000 and it stays stable. However, below that weight we start to see differences and a lot more overlap of communities. Therefore, for within seasons, I stayed above this weighting, and between seasons I used a value just below it.

## Add the 9 graphs to G_coupling
G_coupling.vs['slice'] = graphs


## NOTES ON USING CPM:
# I was using the using the ModularityVertexPartition method to begin with. I wanted to use ModularityVertexPartition because I did not know how to set the CPM resolution value, but now after reading Thomas et al 2014, I understand it better. The resolution value let's us control the level of connectivity within a cluster. The default "find_partition" is just finding the resolution value that minimizes the H values (see the paper). So in a way that is the best arrangment of clusters, but it of course doesn't mean anything ecologically, which is why we need to explore a range.
# Refer to my notes in the paper in mendeley for more detailed comments. Also refer to my email with Vincent Traag:
# "Modularity assumes a specific null-model, which might not make physical sense for you interpretation. In that work about marine ecology we also used CPM. You may want to read that section for some arguments why to use CPM instead of modularity. But you are absolutely right, the difficult choice is the resolution parameter. In practice, it is typically a matter of trying out various values and try to see which types of partitions seem to make most sense in your particular application (and be clear about how you got to that conclusion and particular resolution value)."
# For the interslice partitioning, we set the resolution to 0 because we don't want to put any limit on it.


# CODE FOR TESTING INDIVIDUAL CPM RESOLUTION VALUES (DO THIS TO GET A RANGE OF VALUES TO THEN DO A MORE DETAILED TEST OF)
# layers, interslice_layer, G_full = la.slices_to_layers(G_coupling)
# partitions = [la.CPMVertexPartition(H, weights='weight', node_sizes='node_size', resolution_parameter=0.1) for H in layers]
# interslice_partition = la.CPMVertexPartition(interslice_layer, resolution_parameter=0, node_sizes='node_size', weights='weight')
# optimiser = la.Optimiser()
# optimiser.set_rng_seed(1)
# diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition], n_iterations=2)
# print(partitions[0].summary())
# print(diff)
#print(partitions[0].quality())
# NOTE: for optimise_partition_multiplex, the returned improvement quality is the SUM of the individual qualities of all the partitions.
# I'm not sure the quality really means anything in the context of CPM though.

# test a range of values so that I know the range to do a more detailed test within. I am doing the same analysis as Thomas et al 2014 figure 5.
# 0.9 950 clusters
# 0.6 950
# 0.5 950
# 0.4 950
# 0.3 948
# 0.2 942
# 0.1 903
# 0.01 599
# 0.001 320
# 0.0001 162
# 0.00001 90
# 0.000001 47
# 0.0000001 33
# 0.00000001 22
# 0.000000001 12
# 0.0000000001 12
# 0 12
# greater than 0.000000001 results in a change in the number of clusters. This would make sense since that is probably my smallest edge weight
# at 0.4 and above I get 950 clusters. This is because there are so few (or zero) connections that are of this weight. I will still get more clusters than nodes though because through time connectivity may still be different enough.


#################
# PLOTTING INTERCOMMUNITY CONNECTIVITY VS CPM RESOLUTION PARAMETER
#################
# I can then select values from this to run individually.
# test from 10e-10 to 10e0, within each of those, test 1 to 9

## first get total connectivity
# conns_com_all = pd.DataFrame(columns=['from_id','to_id', 'prob_avg'])
# for dir in dirs:
#     df = gp.read_file(os.path.join(root, dir, shp_conn))
#     df = df.drop(columns=['geometry','time_int','date_start', 'totalori', 'date_start'])
#     df = df[df['from_id'] != df['to_id']]
#     conns_com_all = conns_com_all.append(df)
# total_conn = conns_com_all.prob_avg.sum()

# res_conn = []
# for i in range(-10,1):
#     for j in range(1,10):
#         res=10.0**i * j
#         layers, interslice_layer, G_full = la.slices_to_layers(G_coupling)
#         partitions = [la.CPMVertexPartition(H, weights='weight', node_sizes='node_size', resolution_parameter=res) for H in layers]
#         interslice_partition = la.CPMVertexPartition(interslice_layer, resolution_parameter=0, node_sizes='node_size', weights='weight')
#         optimiser = la.Optimiser()
#         optimiser.set_rng_seed(1)
#         diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition], n_iterations=2)

#         # get inter community connectivity average

#         df_c = pd.DataFrame(columns=['pid','comid'])
#         for p in range(len(partitions[0])): # partitions are all the same at this point, so just need the first one
#             if len(partitions[0].subgraph(p).vs['name'])>1:
#                 for v in partitions[0].subgraph(p).vs['name']:
#                     df_c = df_c.append({'pid':v, 'comid': p}, ignore_index=True)
#         # frequency
#         df_freq = df_c.groupby(['pid', 'comid']).agg(
#             freq = ('pid', 'count'),
#             ).reset_index()
#         gdf_all = df_pts.merge(df_freq, left_on='uID', right_on='pid', how='outer')
#         # if NaN make -1 or else arcgis reads it as 0 and there is already a 0 community
#         gdf_all = gdf_all.fillna({'pid':-1, 'comid':-1, 'freq':-1})

#         # to get inter community connectivity for each resolution value:
#         # in gdf_all, for each unique comid, get the uIDs for the community.
#         # then go through all of my connection files in all of my directories
#         # get the connections to and from those nodes, but not between those nodes
#         # get the total of those connections
#         # then get the total of ALL connections and find the % that is just the intercommunity connectivity
#         # from Vincent: "what fraction of the total weight in the graph is on edges between communities. This goes from 0% (single large cluster) to 100% (the singleton partition, if there were no loops)."
#         conns_com_inter = pd.DataFrame(columns=['from_id','to_id', 'prob_avg'])
#         for com in gdf_all.comid.unique():
#             if com != -1.0:
#                 gdf_com = gdf_all.uID[gdf_all.comid==com].to_list()
#                 for dir in dirs:
#                     df = gp.read_file(os.path.join(root, dir, shp_conn))
#                     df = df.drop(columns=['geometry','time_int','date_start', 'totalori', 'date_start'])
#                     df = df[df['from_id'] != df['to_id']]
#                     df_inter = df[(df.from_id.isin(gdf_com)) | (df.to_id.isin(gdf_com))]
#                     # remove intraconnectivity
#                     df_remove = df_inter[(df_inter.from_id.isin(gdf_com)) & (df_inter.to_id.isin(gdf_com))]
#                     df_inter = pd.concat([df_inter, df_remove]).drop_duplicates(keep=False)
#                     conns_com_inter = conns_com_inter.append(df_inter).drop_duplicates()
#                     # I initially didn't have drop_duplicates, which resulted in interconnections being added multiple times. Even if a connection is directional, it would get added each time each of the two nodes is considered.
#         total_conninter = conns_com_inter.prob_avg.sum()
#         percent_inter = (total_conninter / total_conn) * 100
#         print(percent_inter)

#         res_conn.append([res, percent_inter])

# # output values to csv so that I don't need to run the above again
# df_resconn = pd.DataFrame(res_conn, columns =['Resolution', 'InterConn_percent']) 
# df_resconn.to_csv('interconnectivity.csv', index=False)

# # create plot
# df_resconn = pd.read_csv('interconnectivity.csv')
# sns.set_context('notebook')
# fig, ax = plt.subplots(figsize=(18, 12))
# ax.set(xscale='log', yscale='log')
# sline = sns.lineplot(x="Resolution", y="InterConn_percent", data=df_resconn, ax=ax, marker='o')
# fig
# fig.savefig('resconn.png')



#################
# SELECT LEVELS FROM ABOVE, DETECT COMMUNITIES AND CREATE SHAPEFILES
#################
# from interconnectivity.csv and the plot, I color coded the values that seem to cluster and/or plateau. I will now run 1 'average' value from each of these and create shapefiles from them.
# The logic behind looking for a plateau:
# We are looking for dispersal barriers. If we don't see any change to interconnectivity with DECREASING intraconn strength required then this is really interesting and indicates that even though things are more strongly connected within a cluster, it still is not enough to overcome a barrier and let more nodes in. Eventually, we raise it enough to "break though" a barrier and increase the interconnectivity in the system.
# Decision: from that it looks like I can just do 10**(-9 to 0)
for i in range(-9,1):
    res=10.0**i
    print(res)
    layers, interslice_layer, G_full = la.slices_to_layers(G_coupling)
    partitions = [la.CPMVertexPartition(H, weights='weight', node_sizes='node_size', resolution_parameter=res) for H in layers]
    interslice_partition = la.CPMVertexPartition(interslice_layer, resolution_parameter=0, node_sizes='node_size', weights='weight')
    optimiser = la.Optimiser()
    optimiser.set_rng_seed(1)
    diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition], n_iterations=2)


    # Partitions explanation:
    # each subgraph gives me all the node ids in a cluster, some nodes are repeated, which would be from different time steps
    # its currently a bit confusing relating the structure of "membership" from find_partition_temporal to the structure of partitions[0] from optimise_partition_multiplex. They both are of length 9, but I think membership shows the clusters at each timestep, whereas partitions are all the same because it is taking a "cross" of membership, but within each subgraph of partitions, things can be repeated and occur in different graphs. It get's evened out in a way, hence the optimiser.

    # output a dataset with attributes: node_id (uID), community, frequency
    # frequency will just be the count of each node in a community.
    # This will result in a point feature class where uID is no longer unique.
    df_c = pd.DataFrame(columns=['pid','comid'])
    for p in range(len(partitions[0])): # partitions are all the same at this point, so just need the first one
        if len(partitions[0].subgraph(p).vs['name'])>1:
            for v in partitions[0].subgraph(p).vs['name']:
                df_c = df_c.append({'pid':v, 'comid': p}, ignore_index=True)
    # frequency
    df_freq = df_c.groupby(['pid', 'comid']).agg(
        freq = ('pid', 'count'),
        ).reset_index()
    gdf_all = df_pts.merge(df_freq, left_on='uID', right_on='pid', how='outer')
    # if NaN make -1 or else arcgis reads it as 0 and there is already a 0 community
    gdf_all = gdf_all.fillna({'pid':-1, 'comid':-1, 'freq':-1})
    gdf_all.to_file(filename=out_shp.format(str(abs(i))), driver='ESRI Shapefile')


    # create convex polys of communities
    input = out_shp.format(str(abs(i)))
    df = gp.read_file(input)
    # drop rows that are -1
    df = df[df.comid != -1]
    clusters = df.groupby('comid')
    polys_all = []
    for name, cluster in clusters:
        point_count = len(cluster)
        if point_count > 2:
            poly = Polygon([[p.x, p.y] for p in cluster.geometry.values])
            convex_hull = poly.convex_hull
            polys_all.append([name, point_count, convex_hull, convex_hull.area])
        if point_count ==  2:  # for clusters with only 2 points, create a narrow ellipse
            # coordinates of midpoint
            point1 = cluster.iloc[0].geometry
            point2 = cluster.iloc[1].geometry
            mid_x = (point1.x + point2.x)/2
            mid_y = (point1.y + point2.y)/2        
            dist = point1.distance(point2)
            angle = degrees(atan2(point2.y - point1.y, point2.x - point1.x))
            # create ellipse
            # 1st elem = center point (x,y) coordinates
            # 2nd elem = the two semi-axis values (along x, along y)
            # 3rd elem = angle in degrees between x-axis of the Cartesian base
            #            and the corresponding semi-axis
            ellipse = ((mid_x, mid_y),(dist, 100),angle)
            # create a circle of radius 1 around center point:
            circ = shapely.geometry.Point(ellipse[0]).buffer(1)
            # create the ellipse along x and y:
            ell  = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
            # rotate the ellipse (clockwise, x axis pointing right):
            ellr = shapely.affinity.rotate(ell,ellipse[2])
            # If one need to rotate it clockwise along an upward pointing x axis:
            #elrv = shapely.affinity.rotate(ell,90-ellipse[2])
            # According to the man, a positive value means a anti-clockwise angle,
            # and a negative one a clockwise angle.
            polys_all.append([name, point_count, ellr, ellr.area])

    gdf = gp.GeoDataFrame(polys_all, columns=['comid', 'pt_count', 'geometry', 'area'])
    gdf.crs = df.crs
    gdf.to_file(filename=out_poly.format(str(abs(i))), driver='ESRI Shapefile')



# some more general notes:
# https://github.com/vtraag/leidenalg/issues/14
# Vtraag: "I think there is some confusion around how the multiplex community detection works.
# The graphs that are passed to find_partition_multiplex should all have an identical vertex set. Each node in each graph that is passed to find_partition_multiplex is assigned to the same cluster. In other words, the clustering is identical across the different layers.
# Similarly, all partitions that are passed to optimise_partition_multiplex are assumed to have an identical vertex set. The membership of all partitions that are passed are again identical.
# In order to deal with various time slices in which the "same node" in a different time slice may be assigned to different clusters, we create one large network, which embeds the first time slice and the second time slice, and also the interlisce links between the two slices. This is explained in the section slices to layers in the documentation. The membership vector of any partition will hence contain the membership of all nodes in all time slices. See also the documentation for slices_to_layers for more details."
# https://github.com/vtraag/leidenalg/issues/26
# Regarding CPM and the resolution parameter:
# Vtraag: "Personally, I favour CPM, because it does not suffer from the resolution-limit (see http://arxiv.org/abs/1104.3083). I think it also makes sense in the context of similarities. The resolution parameter of CPM can then be understood as the desirable average similarity between nodes within communities. For example, if you use a resolution of 0.9, nodes in the same community will have an average similarity of at least 0.9 with the rest of its community, and a similarity with the rest of the network that is at most 0.9. There is also a nice connection with a layout methodology (see https://arxiv.org/abs/1006.1032). It seems that most analyses still use modularity however (with resolution parameter, so la.RBConfigurationVertexPartition).
# Quality:
#  la.ModularityVertexPartition.quality() is normalised with the number of links (or total weight), while la.RBConfigurationVertexPartition.quality() is unnormalised. This is a completely trivial difference of course, but some people were confused why some quality functions did not correspond exactly to the quality functions as defined in the literature.

