

# temporal community detection


import igraph as ig
import leidenalg as la
import pandas as pd
import geopandas as gp
import os
from shapely.geometry import Polygon, Point
import shapely.affinity
from math import atan2, degrees


#################
# User input
#################

root = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs_cluster\seagrass'
# dirs = [
#     'seagrass_20200310_SS201101',
#     'seagrass_20200310_SS201105',
#     'seagrass_20200310_SS201108',
#     'seagrass_20200327_SS201401',
#     'seagrass_20200327_SS201405',
#     'seagrass_20200327_SS201408',
#     'seagrass_20200228_SS201701',
#     'seagrass_20200309_SS201705',
#     'seagrass_20200309_SS201708',
#     ]

# reordered by seasons
dirs = [
    'seagrass_20200310_SS201101',
    'seagrass_20200327_SS201401',
    'seagrass_20200228_SS201701',
    'seagrass_20200310_SS201105',
    'seagrass_20200327_SS201405',
    'seagrass_20200309_SS201705',
    'seagrass_20200310_SS201108',
    'seagrass_20200327_SS201408',
    'seagrass_20200309_SS201708',
    ]

shp_conn = r'shp_merged\connectivity_average.shp'

shp_pts = r'shp_merged\patch_centroids.shp'
out_shp = r'output_figs_SALISHSEA_ALL\TEST\communities_ALL_REORDSEAS.shp'

out_poly = r'output_figs_SALISHSEA_ALL\TEST\patch_clusters_convexhull_REORDSEAS.shp'


#################
# Setup and format
#################

df_pts = gp.read_file(os.path.join(root, dirs[0], shp_pts)) # just need to get this once
out_shp = os.path.join(root, out_shp)
out_poly = os.path.join(root, out_poly)

graphs = []

# Remove weak links:
# justification: the smallest prob that the smallest meadow can be is 0.002 (1/84 / 5plds)
# so round down to 0.001. For the biggest meadow this would be 400 particles. For the mean, this is 5 particles.
for dir in dirs:
    df = gp.read_file(os.path.join(root, dir, shp_conn))
    df = df.drop(columns=['geometry','time_int','date_start', 'totalori', 'date_start'])
    df = df[df['from_id'] != df['to_id']]
    df = df[df.prob_avg >= 0.001] # results in 76 clusters, USE THIS LEVEL. IT HAS GOOD JUSTIFICATION
    # still connects the far one near neah. However, when you look into that meadow, it does indeed have a couple very strong connections to meadows on the edge of the gulf islands. This is SUPER interesting. It is not present in the winter or spring, but present STRONGLY in the summer. It is about 8 particles out of 900 that make it there.
    # test 0.0001
    #df = df[df.prob_avg >= 0.0001] # 70 clusters
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

membership, improvement = la.find_partition_temporal(graphs, la.ModularityVertexPartition, interslice_weight=0.1)
# This is the default method. However, you can't apply different interslice weights.
#membership, improvement = la.find_partition_temporal(graphs, la.ModularityVertexPartition, interslice_weight=0.1)

# membership is the community id of each node. To get the node id print out G.vs['id']
# Understanding multiplex structure:
# These functions are BUILDING the multiplex network. It has nothing to do with averaging communities.
# Nodes can change membership in each slice and can therefore take on multiple community IDs. Think of it as overlapping convex hulls as the final product. This would represent overall communities and symbolize how they vary.

# The interslice link is having an effect on this. Compare membership[8] to doing just a single find_partition on the last graph:
# partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight', seed=11)
# you can see that they are different memberships


# Additional exploration:
# I was confused how an optimiser was acting on the partition objest because it looked like the returned diff was just a number and partition wasn't changing (in traditional python-sense I would expect partition to be recreated as a new variable if it was going to change). However, you can see that partition does change.
# Print out partition before and after optimise to see.
#G = ig.Graph.Famous('Zachary')
#optimiser = la.Optimiser()
#partition = la.ModularityVertexPartition(G)
#diff = optimiser.optimise_partition(partition)


#################
# Community detection
#################

# find_partition_temporal is a helper that packages up the optimiser and partition creation and makes the assumption to weight the interslices the same. However, if I want to weight the interslices differently then you need to do these steps manually.
# I would do this because there are different lengths of time between my slices (months within a year, then 2 years between years). Ones that are closer together in time should be considered more similar.

G_coupling = ig.Graph.Formula("1-->2-->3-->4-->5-->6-->7-->8-->9>") # I need a <> before or after, which isn't mentioned at all by igraph, but I could not get it to work otherwise.
# The interslice layer is itself a graph and the individual graphs are nodes. Therefore the weights are similar to my connection weights.

# So perhaps I'll break it down by months. If something is in the following month then it would have a connection strength of 0.5. Then I'll decrease it proportionally from there.
# difference between seasons is 0.5-2.5 months. Use 1.5.
# difference between years is 2 years and 2.5 months, so 26.5 months
# so the ratio is ~1:17.6
# I'll simplify:
# weight between seasons: 0.5
# weight between years: 0.03

# testing different weightings:
G_coupling.es['weight'] = [0.001, 0.001, 0.00001, 0.001, 0.001, 0.00001, 0.001, 0.001]
# the lower we set the weight, the more we let the difference in slices come out
# therefore, find the threshold where it starts to change
# 0.000001 127 clusters
# 0.00001 99 clusters
# 0.0001 77 clusters
# 0.001 77 clusters
# 0.01 77 clusters
# 0.1 77 clusters
# 1 76 clusters
# 1000 76 clusters

# testing reordering by season:
# I think I will avoid setting differences between years. I'm most curious about drawing out differences in seasonal variation. Plus looking across years was more about getting an average between them. I didn't really expect to see something with the blob.
# but therefore, maybe it makes sense to set things as:
# w-w-w-sp-sp-sp-su-su-su
# then use that break point of 0.0001 and 0.00001 to draw out differences between seasons
# TEST:
# ordered by time, weighted evenly
# ordered by time, weighted with break point at years
# reordered by season and weighted evenly
# reordered by season with breakpoint between seasons

# I really like the reorderd by seasons and weighted differently between seasons.
# notice how tsawwassen extends to mayne island, but there is overlap with 2 polygons on either side. This is exactly what i want to indicate.
# the point sticking out most at the port is in one community 6 times and another 3 times
# DECISION: reorder by season with breakpoint between seasons. I could say something like:
# I tested a range of interslice weightings and different arrangements of slices. I am interested in seasonal variation, and repetition across years is just to get more of an average for a season, not to look at yearly variation. Therefore, I arranged slices by season. I used a higher interslice weighting between slices of the same season to smooth out any differences in community structure, then I used a lower weighting between seasons to allow community structure to vary more so as to see if there are changing dynamics between seasons. The general structure is not sensitive to any weights above 0.0001. I tested all way up to 1000 and it stays stable. However, below that weight we start to see differences and a lot more overlap of communities. Therefore, for within seasons, I stayed above this weighting, and between seasons I used a value just below it.


G_coupling.vs['slice'] = graphs

# Then we can use the established manual procedure

layers, interslice_layer, G_full = la.slices_to_layers(G_coupling)
partitions = [la.ModularityVertexPartition(H, weights='weight') for H in layers]
interslice_partition = la.CPMVertexPartition(interslice_layer, resolution_parameter=0, node_sizes='node_size', weights='weight')
optimiser = la.Optimiser()
diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition], n_iterations=-1)

# I was using the above method to begin with. I wanted to use ModularityVertexPartition because I did not know how to set the CPM resolution value, but now after reading Thomas et al 2014, I understand it better. The resolution value let's us control the level of connectivity within a cluster. The default "find_partition" is just finding the resolution value that minimizes the H values (see the paper). So in a way that is the best arrangment of clusters, but it of course doesn't mean anything ecologically, which is why we need to explore a range.
# Refer to my notes in the paper in mendeley for more detailed comments
# For the interslice partitioning, we set the resolution to 0 because we don't want to put any limit on it.

layers, interslice_layer, G_full = la.slices_to_layers(G_coupling)
partitions = [la.CPMVertexPartition(H, weights='weight', node_sizes='node_size', resolution_parameter=0.0001) for H in layers]
interslice_partition = la.CPMVertexPartition(interslice_layer, resolution_parameter=0, node_sizes='node_size', weights='weight')
optimiser = la.Optimiser()
diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition], n_iterations=-1)




# Partitions:
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
gdf_all.to_file(filename=out_shp, driver='ESRI Shapefile')


# create convex polys
input = out_shp
df = gp.read_file(input)
# drop rows that are -1
df = df[df.comid != -1]
# create convex hull polys
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
gdf.to_file(filename=out_poly, driver='ESRI Shapefile')



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





print(len(partitions[0]))