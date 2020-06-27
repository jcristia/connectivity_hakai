# for one community average connectivity, detect the shortest path between any two points
# I can then create a map of likely dispersal corridors

# see Hock and Mumby 2015 for methods (sectioh 2.2)
# use networkx generic shortest path
# Hock and Mumby 2015: They found that a multiplicative approach is better to predict the most likely path than an additive approach (which is equivalent to a least cost path). However, I did read everything about this and it didn't make sense how they determined this with simulations.
# My thoughts: it seems to me that from an organisms perspective, you would always take the path of least resistance. An individual would not know what resistance would be on the next step. However, from a multi-generational standpoint, a multiplicative approach would provide the most likely path for an individual and offspring to traverse an area.
# see 8_shortest_path.xlsx for examples of using a negative LN transformation, adddition, and taking the minimum and how this is equivalent to a multiplicative approach


import os
import pandas as pd
import geopandas as gp
import numpy as np
import networkx as nx
from shapely.geometry import LineString
from shapely import geometry, ops


project = 'seagrass_20200228_SS201701'
shp_conn = 'connectivity_average.shp'

root = r'D:\Hakai\script_runs\seagrass\{}\shp_merged\{}'
path = root.format(project, shp_conn)
out = root.format(project, os.path.splitext(shp_conn)[0] + '_shortpath.shp')


df = gp.read_file(path)
df = df[df.from_id != df.to_id]

# do negative LN transformation
df['prob_avg_negLN'] = -(np.log(df.prob_avg))

# read pandas df as networkx edge list
G = nx.from_pandas_edgelist(df, 'from_id', 'to_id', ['prob_avg_negLN'], nx.DiGraph())

# shortest path
p_nodes = nx.shortest_path(G, weight='prob_avg_negLN', method='dijkstra')
p_length = dict(nx.shortest_path_length(G, weight='prob_avg_negLN', method='dijkstra'))

# create a line shapefile of all paths
lines = []
for node in p_length:
    for dest in p_length[node]:
        from_id = node
        to_id = dest
        # don't convert probabilities back, they are way too small
        prob = p_length[node][dest]
        # get individual lines in path
        path = p_nodes[from_id][to_id]
        # from individual lines, build one line
        if len(path) > 2:
            i=0
            geoms = []
            while i < len(path)-1:
                geom = df.geometry[(df.from_id == path[i]) & (df.to_id == path[i+1])]
                geoms.append(geom.values[0])
                i+=1
            shortest_path = ops.linemerge(geoms)
            lines.append([from_id, to_id, prob, shortest_path])

gdf = gp.GeoDataFrame(lines, columns=['from_id', 'to_id', 'prob_negLN', 'geometry'])
gdf.crs = df.crs
gdf['length'] = gdf.geometry.length
gdf.to_file(filename=out, driver='ESRI Shapefile')