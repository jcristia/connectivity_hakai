""" This script calculates connectivity metrics using the resulting connectivity lines """

## Modules ##
import networkx as nx
import pandas as pd
import numpy as np
import arcpy
import os

#################################
### Configure these variables ###
#################################

arcpy.env.overwriteOutput = True

### Results directory ###
#
# The directory that has all of the results folders for each run of the simualation
#
##

results_directory = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output'

### Paths and file names ###
#
# The connectivity lines output from the biology script
# The seagrass patch centers
# The output gdb to hold copies of lines and points with network metrics added in
# 
##

results_shp = 'connectivity.shp'
patch_centroids = 'patch_centroids.shp'
project_gdb = "metrics.gdb"

### Connectivity metrics feature classes ###
#
# These feature classes will be created to hold the connectivity metrics
# The fc_conn_lines is a copy of the Connectivity results line feature class
# and holds metrics that are connection specific.
# The fc_conn_pts holds metrics that are specific to patches. It uses a copy
# of patch point data.
#
##
fc_conn_lines = 'connectivity_metrics_connections'
fc_conn_pts = 'connectivity_metrics_patches'

### Merged gdb ###
#
# This gdb holds the merged feature classes that are project-level and not for just one simulation's results
#
##

merged_gdb = r"metrics_merged.gdb"




######################
### Implementation ###
######################

###         ###
## Functions ##
###         ###

## createMetricFCs ##
#
# checks if connectivity metrics feature classes exists for each results folder
# create fcs where they do not exist
#

def createMetricFCs(results_directory, results_shp, patch_centroids, fc_conn_lines, fc_conn_pts, project_gdb):
    # go through each results folder. This gets the intermediate folder between results_directory and results_gdb. This is the one labeled by date and time.
    folders = os.listdir(results_directory)
    for folder in folders:
        gdb = os.path.join(results_directory, folder, project_gdb)
        in_shp_lines = os.path.join(results_directory, folder, results_shp)
        in_shp_pts = os.path.join(results_directory, folder, patch_centroids)
        out_fc_lines = os.path.join(gdb, fc_conn_lines)
        out_fc_pts = os.path.join(gdb, fc_conn_pts)
        if not arcpy.Exists(gdb):
            arcpy.CreateFileGDB_management(os.path.join(results_directory, folder), project_gdb)
        if arcpy.Exists(out_fc_lines):
            print folder + ": fc_conn_lines already exists"
        else:
            arcpy.CopyFeatures_management(in_shp_lines, out_fc_lines)
            print folder + ": fc_conn_lines created"
        if arcpy.Exists(out_fc_pts):
            print folder + ": fc_conn_pts already exists"
        else:
            arcpy.CopyFeatures_management(in_shp_pts, out_fc_pts)
            print folder + ": fc_conn_pts created"

## createDateField ##
#
# creates a date field and populates with the date of the simulation
# geopandas prints date as a string, but I want the date to be recognized by Arc as the DATE type
#

def createDateField(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):

    folders = os.listdir(results_directory)
    for folder in folders:
        gdb = os.path.join(results_directory, folder, project_gdb)
        fc_ln = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)
        fc_pt = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        fieldNames_ln = [field.name for field in arcpy.ListFields(fc_ln)]
        fieldNames_pt = [field.name for field in arcpy.ListFields(fc_pt)]
        if "date" in fieldNames_ln:
            print folder + ": connections date field already exists"
            continue
        else:
            arcpy.AddField_management(fc_ln, "date", "DATE")
        if "date" in fieldNames_pt:
            print folder + ": patches date field already exists"
            continue
        else:
            arcpy.AddField_management(fc_pt, "date", "DATE")

        with arcpy.da.SearchCursor(fc_ln, ["date_start"]) as cursor:
            for row in cursor:
                date_start = row[0]
                break

        # why you use updateCursor instead of CalculateField when working with dates:
        # https://gis.stackexchange.com/questions/153978/why-is-arcpy-calculatefield-management-writing-1899-12-30-000000-instead-of

        with arcpy.da.UpdateCursor(fc_ln, ["date"]) as cursor:
            for row in cursor:
                row[0] = date_start
                cursor.updateRow(row)

        with arcpy.da.UpdateCursor(fc_pt, ["date"]) as cursor:
            for row in cursor:
                row[0] = date_start
                cursor.updateRow(row) 

        deleteField(results_directory, project_gdb, fc_ln, ['date_start'])
        deleteField(results_directory, project_gdb, fc_pt, ['date_start'])

        print folder + ": date field added complete"

## sourceSink ##
#
# calculates various metrics to determine the degree to which a patch is acting as a source or sink
# creates new fields in the Connectivity_metrics_patches feature class
#

def calcSourceSink(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):

    # add source-sink fields
    folders = os.listdir(results_directory)
    for folder in folders:
        fc = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        fieldNames = [field.name for field in arcpy.ListFields(fc)]
        if "From_count" in fieldNames:
            print folder + ": source-sink fields already exists"
            continue    # skips this folder in the for loop
        else:
            arcpy.AddField_management(fc, "from_count", "SHORT")
            arcpy.AddField_management(fc, "from_quantity", "DOUBLE")
            arcpy.AddField_management(fc, "to_count", "SHORT")
            arcpy.AddField_management(fc, "to_quantity", "DOUBLE")
            arcpy.AddField_management(fc, "net_count", "SHORT")
            arcpy.AddField_management(fc, "net_quantity", "DOUBLE")

        # update with info from connectivity line fc
        c_fc = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)
        cursor_patchIDs = arcpy.da.UpdateCursor(fc, ['uID', 'from_count', 'from_quantity', 'to_count', 'to_quantity'])
        cursor_connectivity = arcpy.da.SearchCursor(c_fc, ['from_id', 'to_id', 'quantity'])
        for row in cursor_patchIDs:
            count_from = 0
            quantity_from = 0.0
            count_to = 0
            quantity_to = 0.0
            for row_c in cursor_connectivity:
                if row_c[0] == row[0]:
                    count_from += 1
                    quantity_from += row_c[2]
                if row_c[1] == row[0]:
                    count_to += 1
                    quantity_to += row_c[2]
            row[1] = count_from
            row[2] = quantity_from
            row[3] = count_to
            row[4] = quantity_to
            cursor_patchIDs.updateRow(row)
            cursor_connectivity.reset()  # super important
    
        del cursor_patchIDs, cursor_connectivity

        # calculate net fields
        # I will consider a sink to have negative values, so therefore I will From minus To
        arcpy.CalculateField_management(fc, "net_count", "!from_count! - !to_count!", "PYTHON_9.3")
        arcpy.CalculateField_management(fc, "net_quantity", "!from_quantity! - !to_quantity!", "PYTHON_9.3")

        print folder + ": source-sink calculation complete"

## betweenness ##
#
# calculates betweenness using networkX
# creates new field in the Connectivity_metrics_patches feature class
# Betweenness: 
# https://en.wikipedia.org/wiki/Betweenness_centrality #Definition
# http://algo.uni-konstanz.de/publications/b-vspbc-08.pdf search for "normalized"
# the number of shortest paths between nodes s and t that pass through node v, divided by the total number of paths between s and t
# therefore the largest number possible for a node is the total number of paths (if every combination of nodes passes through that point)
# we can then normalize this by scaling it to between 0 and 1
# keep in mind that in directional graphs we consider both directions (s-t and t-s), so we do not divide it by two
# (n-1)(n-2) can be thought of as: n-1 comes from the sample size minus the starting point s, and n-2 is every possible combination of nodes not including s (itself) or a connection to v (the node in question)
#

def calcBetweenness(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):
    
    # note: even though there is a shapefile import in networkx, it is not the most efficient way. When it creates the nodes, it doesn't recognize the from and to as the node labels. To create a script that does this is more complicated than just exporting the fc out as a table then creating a pandas dataframe. I need the nodes labeled correctly because when I put them back into the GIS as points, they don't necessarily overlay exactly with the raster patch IDs, since many of these patch IDs are not continuous and the centroid falls in a nodata cell
    
    arcpy.env.overwriteOutput = True

    folders = os.listdir(results_directory)
    for folder in folders:
        
        fc = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        c_fc = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)

        fieldNames = [field.name for field in arcpy.ListFields(fc)]
        if "betweenness_centrality" in fieldNames:
            print folder + ":  betweenness field already exists"
            continue
        else:
            arcpy.AddField_management(fc, "betweenness_centrality", "DOUBLE")

        # convert connectivity line feature class to xls, then to pandas df
        # note: I would have prefered going to csv or txt first, but there was a problem with matplotlib library required
        temp_xls = os.path.join(results_directory, folder, "temp.xls")
        arcpy.TableToExcel_conversion(c_fc, temp_xls)
        df = pd.read_excel(temp_xls, 'temp', header=0, usecols=['from_id','to_id','prob'])  #oddly, usecols is not in documentation for read_excel
        # read pandas df in as networkX graph
        G = nx.from_pandas_edgelist(df, 'from_id', 'to_id', ['prob'], nx.DiGraph())
        # explore the data
        # comment this out, but keep it available for reference
        #G.number_of_nodes()
        #G.number_of_edges()
        #G.number_of_selfloops()
        #list(G.nodes)
        #list(G.edges)[0]
        ## access the attributes of the edges
        #Quantity = nx.get_edge_attributes(G, 'Quantity')
        #Quantity[list(G.edges)[0]]
        ## another method:
        #list(G.nodes(data=True))[0]
        #list(G.edges(data=True))[0]
        
        bt = nx.betweenness_centrality(G, None, True, 'prob', False)
        
        # add betweenness as a node attribute
        nx.set_node_attributes(G, bt, "bt")
        
        # output node dictionary to pandas dataframe then to csv:
        df = pd.DataFrame(columns=['uID','bt'])    # create an empty dataframe with the column headers
        nd = list(G.nodes(data=True))
        i=0
        for v in bt:
            df.loc[i] = [nd[i][0], nd[i][1]['bt']]   # this accesses the x, y, and betweeness values of the node list. Note that the bt value is stored in a dictionary which is within a list
            i+=1
        temp_csv = os.path.join(results_directory, folder, "temp.csv")
        df.to_csv(temp_csv, index=False)
        
        # join to connectivity_metrics_patches fc
        arcpy.MakeFeatureLayer_management(fc, "out_lyr_temp")
        arcpy.AddJoin_management("out_lyr_temp", "uID", temp_csv, "uID")
        arcpy.CalculateField_management("out_lyr_temp", "betweenness_centrality", "!temp.csv.bt!", "PYTHON_9.3")
    
        # del input excel file, del output csv
        arcpy.Delete_management(temp_xls)
        arcpy.Delete_management(temp_csv)

        print folder + ": betweenness calculation complete"

## closeness centrality ##
#
# calculates closeness centrality using networkX
# creates new fields in the Connectivity_metrics_patches feature class
# Closeness: the reciprocal of the average shortest path distance to a node over all reachable nodes
# I calculate two values of closeness using euclidean distance and quantity.
# update: I use probability. Also, it's not euclidean distance since that information is never read in.
# Instead, I think it is just considering any connection as equal to any other connection.
#

def calcCloseness(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):
    
    # closeness works well for distance. I did distance with Quantity and it works, but I think I need to take the inverse. With normal distance, higher values of closeness indicates higher values of centrality. However, since higher quantity indicates being more close then I need to consider that lower values of closeness indicate higher centrality.
    # for weighted closeness in networkx, do 1/quantity before calculating closeness

    arcpy.env.overwriteOutput = True

    folders = os.listdir(results_directory)
    for folder in folders:
        
        fc = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        c_fc = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)

        fieldNames = [field.name for field in arcpy.ListFields(fc)]
        if "closeness_centrality_prob" in fieldNames:
            print folder + ":  closeness fields already exist"
            continue
        else:
            arcpy.AddField_management(fc, "closeness_centrality_distance", "DOUBLE")
            arcpy.AddField_management(fc, "closeness_centrality_prob", "DOUBLE")

        # convert connectivity line feature class to xls, then to pandas df
        # note: I would have prefered going to csv or txt first, but there was a problem with matplotlib library required
        temp_xls = os.path.join(results_directory, folder, "temp.xls")
        arcpy.TableToExcel_conversion(c_fc, temp_xls)
        df = pd.read_excel(temp_xls, 'temp', header=0, usecols=['from_id','to_id','prob'])  #oddly, usecols is not in documentation for read_excel
        # read pandas df in as networkX graph
        G = nx.from_pandas_edgelist(df, 'from_id', 'to_id', ['prob'], nx.DiGraph())
    
        # closeness with normal distance
        cc = nx.closeness_centrality(G, None, None, True)
        # closeness with quantity as distance
        ccq = nx.closeness_centrality(G, None, 'prob', True)
    
        # add metrics as node attributes
        nx.set_node_attributes(G, cc, "cc")
        nx.set_node_attributes(G, ccq, "ccq")
    
        # output node dictionary to pandas dataframe then to csv:
        df = pd.DataFrame(columns=['uID','cc_d','cc_q'])    # create an empty dataframe with the column headers
        nd = list(G.nodes(data=True))
        for v in range(len(nd)):
            df.loc[v] = [nd[v][0], nd[v][1]['cc'], nd[v][1]['ccq']]  # this accesses the node id and attribute values of the node list. Note that the attribute value is stored in a dictionary which is within a list
        temp_csv = os.path.join(results_directory, folder, "temp.csv")
        df.to_csv(temp_csv, index=False)

        # join to connectivity_metrics_patches fc
        arcpy.MakeFeatureLayer_management(fc, "out_lyr_temp")
        arcpy.AddJoin_management("out_lyr_temp", "uID", temp_csv, "uID")
        arcpy.CalculateField_management("out_lyr_temp", "closeness_centrality_distance", "!temp.csv.cc_d!", "PYTHON_9.3") 
        arcpy.CalculateField_management("out_lyr_temp", "closeness_centrality_quantity", '1/ !temp.csv.cc_q!', "PYTHON_9.3")   
    
        arcpy.Delete_management(temp_xls)
        arcpy.Delete_management(temp_csv)

        print folder + ": closeness calculation complete"

## eigenvenctor centrality ##
#
# calculates eigenvector centrality using networkX
# creates new field in the Connectivity_metrics_patches feature class
# Eigenvector Centrality: the centrality for a node based on the centrality of its neighbors.
# This will help determine how much a node is connected to high scoring nodes (could help to identify zones)
#

def calcEigenvectorCentrality(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):


    folders = os.listdir(results_directory)
    for folder in folders:
        
        fc = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        c_fc = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)

        fieldNames = [field.name for field in arcpy.ListFields(fc)]
        if "eigenvector_centrality_in" in fieldNames:
            print folder + ":  eigenvector centrality field already exists"
            continue
        else:
            arcpy.AddField_management(fc, "eigenvector_centrality_in", "DOUBLE")

        # convert connectivity line feature class to xls, then to pandas df
        # note: I would have prefered going to csv or txt first, but there was a problem with matplotlib library required
        temp_xls = os.path.join(results_directory, folder, "temp.xls")
        arcpy.TableToExcel_conversion(c_fc, temp_xls)
        df = pd.read_excel(temp_xls, 'temp', header=0, usecols=['from_id','to_id','prob'])  #oddly, usecols is not in documentation for read_excel
        # read pandas df in as networkX graph
        G = nx.from_pandas_edgelist(df, 'from_id', 'to_id', ['prob'], nx.DiGraph())
    
        # eigenvector centrality
        eci = nx.eigenvector_centrality(G, 1000, 1.0e-6, None,'prob')
    
        # add metrics as node attributes
        nx.set_node_attributes(G, eci, "eci")
    
        # output node dictionary to pandas dataframe then to csv:
        df = pd.DataFrame(columns=['uID','eci'])    # create an empty dataframe with the column headers
        nd = list(G.nodes(data=True))
        for v in range(len(nd)):
            df.loc[v] = [nd[v][0], nd[v][1]['eci']]  # this accesses the node id and attribute values of the node list. Note that the attribute value is stored in a dictionary which is within a list
        temp_csv = os.path.join(results_directory, folder, "temp.csv")
        df.to_csv(temp_csv, index=False)

        # join to connectivity_metrics_patches fc
        arcpy.MakeFeatureLayer_management(fc, "out_lyr_temp")
        arcpy.AddJoin_management("out_lyr_temp", "uID", temp_csv, "uID")
        arcpy.CalculateField_management("out_lyr_temp", "eigenvector_centrality_in", "!temp.csv.eci!", "PYTHON_9.3") 
    
        arcpy.Delete_management(temp_xls)
        arcpy.Delete_management(temp_csv)

        print folder + ": eigenvector centrality calculation complete"


## degree centrality ##
#
# calculates degree centrality using networkX
# creates new fields in the Connectivity_metrics_patches feature class
# This is similar to my source/sink metrics, but is normalized.
# The degree centrality for a node v is the fraction of nodes it is connected to.
# The in-degree centrality for a node v is the fraction of nodes its incoming edges are connected to.
# The out-degree centrality for a node v is the fraction of nodes its outgoing edges are connected to.
#

def calcDegreeCentrality(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):

    folders = os.listdir(results_directory)
    for folder in folders:
        
        fc = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        c_fc = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)

        fieldNames = [field.name for field in arcpy.ListFields(fc)]
        if "degree_centrality_all" in fieldNames:
            print folder + ":  degree centrality fields already exist"
            continue
        else:
            arcpy.AddField_management(fc, "degree_centrality_all", "DOUBLE")
            arcpy.AddField_management(fc, "degree_centrality_in", "DOUBLE")
            arcpy.AddField_management(fc, "degree_centrality_out", "DOUBLE")

        # convert connectivity line feature class to xls, then to pandas df
        # note: I would have prefered going to csv or txt first, but there was a problem with matplotlib library required
        temp_xls = os.path.join(results_directory, folder, "temp.xls")
        arcpy.TableToExcel_conversion(c_fc, temp_xls)
        df = pd.read_excel(temp_xls, 'temp', header=0, usecols=['from_id','to_id','prob'])  #oddly, usecols is not in documentation for read_excel
        # read pandas df in as networkX graph
        G = nx.from_pandas_edgelist(df, 'from_id', 'to_id', ['prob'], nx.DiGraph())
    
        # degree centrality
        dca = nx.degree_centrality(G)
        dci = nx.in_degree_centrality(G)
        dco = nx.out_degree_centrality(G)

        # add metrics as node attributes
        nx.set_node_attributes(G, dca, "dca")
        nx.set_node_attributes(G, dci, "dci")
        nx.set_node_attributes(G, dco, "dco")
    
        # output node dictionary to pandas dataframe then to csv:
        df = pd.DataFrame(columns=['uID','dca','dci', 'dco'])    # create an empty dataframe with the column headers
        nd = list(G.nodes(data=True))
        for v in range(len(nd)):
            df.loc[v] = [nd[v][0], nd[v][1]['dca'], nd[v][1]['dci'], nd[v][1]['dco']]  # this accesses the node id and attribute values of the node list. Note that the attribute value is stored in a dictionary which is within a list
        temp_csv = os.path.join(results_directory, folder, "temp.csv")
        df.to_csv(temp_csv, index=False)

        # join to connectivity_metrics_patches fc
        arcpy.MakeFeatureLayer_management(fc, "out_lyr_temp")
        arcpy.AddJoin_management("out_lyr_temp", "uID", temp_csv, "uID")
        arcpy.CalculateField_management("out_lyr_temp", "degree_centrality_all", "!temp.csv.dca!", "PYTHON_9.3") 
        arcpy.CalculateField_management("out_lyr_temp", "degree_centrality_in", "!temp.csv.dci!", "PYTHON_9.3")
        arcpy.CalculateField_management("out_lyr_temp", "degree_centrality_out", "!temp.csv.dco!", "PYTHON_9.3")
    
        arcpy.Delete_management(temp_xls)
        arcpy.Delete_management(temp_csv)

        print folder + ": degree centrality calculation complete"

## reciprocity ##
#
# calculates reciprocity using networkX
# creates new field in the Connectivity_metrics_patches feature class
# The reciprocity of a single node u is the ratio of the number of edges in both
# directions to the total number of edges attached to node u.
#

def calcReciprocity(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):

    folders = os.listdir(results_directory)
    for folder in folders:
        
        fc = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        c_fc = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)

        fieldNames = [field.name for field in arcpy.ListFields(fc)]
        if "reciprocity" in fieldNames:
            print folder + ":  reciprocity field already exists"
            continue
        else:
            arcpy.AddField_management(fc, "reciprocity", "DOUBLE")

        # convert connectivity line feature class to xls, then to pandas df
        # note: I would have prefered going to csv or txt first, but there was a problem with matplotlib library required
        temp_xls = os.path.join(results_directory, folder, "temp.xls")
        arcpy.TableToExcel_conversion(c_fc, temp_xls)
        df = pd.read_excel(temp_xls, 'temp', header=0, usecols=['from_id','to_id','prob'])  #oddly, usecols is not in documentation for read_excel
        # read pandas df in as networkX graph
        G = nx.from_pandas_edgelist(df, 'from_id', 'to_id', ['prob'], nx.DiGraph())
    
        # eigenvector centrality
        nd = list(G.nodes(data=False))
        rec = nx.reciprocity(G, nd)
    
        # add metrics as node attributes
        nx.set_node_attributes(G, rec, "rec")
    
        # output node dictionary to pandas dataframe then to csv:
        df = pd.DataFrame(columns=['uID','rec'])    # create an empty dataframe with the column headers
        nd = list(G.nodes(data=True))
        for v in range(len(nd)):
            df.loc[v] = [nd[v][0], nd[v][1]['rec']]  # this accesses the node id and attribute values of the node list. Note that the attribute value is stored in a dictionary which is within a list
        temp_csv = os.path.join(results_directory, folder, "temp.csv")
        df.to_csv(temp_csv, index=False)

        # join to connectivity_metrics_patches fc
        arcpy.MakeFeatureLayer_management(fc, "out_lyr_temp")
        arcpy.AddJoin_management("out_lyr_temp", "uID", temp_csv, "uID")
        arcpy.CalculateField_management("out_lyr_temp", "reciprocity", "!temp.csv.rec!", "PYTHON_9.3") 
    
        arcpy.Delete_management(temp_xls)
        arcpy.Delete_management(temp_csv)

        print folder + ": reciprocity calculation complete"


## direction ##
#
# calculates the bearing direction of each connection line
# creates a new field in the Connectivity_metrics_connections feature class
#

def calcDirection(results_directory, project_gdb, fc_conn_lines):
    
    folders = os.listdir(results_directory)
    for folder in folders:
        # check if field exists, add column for timeOfConnection
        fc = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)
        fieldNames = [field.name for field in arcpy.ListFields(fc)]
        if "BEARING" in fieldNames:
            print folder + ": bearing field already exists"
            continue    # skips this folder in the for loop

        arcpy.AddGeometryAttributes_management(fc, "LINE_BEARING")

        with arcpy.da.UpdateCursor(fc, ["from_id", "to_id", "BEARING"]) as cursor:
            for row in cursor:
                if row[2] is None:
                    row[2] = 0.0
                if row[0] == row[1]:
                    row[2] = None
                cursor.updateRow(row)
            
        print folder + ": direction calculation complete"

## mergeFCs ##
#
# Combines all the feature classes together into a new feature class.
# This allows me to query by date and see trends.
# When it creates a new fc, it adds todays date to the file name.
# I chose to merge instead of append so that I get a new fc each time.
# This accounts for when I add new simulation results and new attribute fields.
#

def mergeFCs(results_directory, project_gdb, fc_conn_lines, fc_conn_pts, merged_gdb):
    
    mgdb = os.path.join(os.path.dirname(results_directory), merged_gdb) #dirname gets the parent directory. I don't want to store the merged gdb in the same directory as the results.
    if not arcpy.Exists(mgdb):
        arcpy.CreateFileGDB_management(os.path.dirname(results_directory), merged_gdb)

    fcs_ln_ALL = []
    fcs_pt_ALL = []

    folders = os.listdir(results_directory)
    for folder in folders:
        gdb = os.path.join(results_directory, folder, project_gdb)
        fc_ln = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)
        fc_pt = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        fcs_ln_ALL.append(fc_ln)
        fcs_pt_ALL.append(fc_pt)

    time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    fcs_ln_ALL_output = os.path.join(mgdb, fc_conn_lines + "_" + time)
    fcs_pt_ALL_output = os.path.join(mgdb, fc_conn_pts + "_" + time)

    arcpy.Merge_management(fcs_ln_ALL, fcs_ln_ALL_output)
    arcpy.Merge_management(fcs_pt_ALL, fcs_pt_ALL_output)

    print "merge complete"

## delete field ##
#
# deletes specified fields from specified fc
#

def deleteField(results_directory, project_gdb, fc, fields):

    folders = os.listdir(results_directory)
    for folder in folders:
        fc_fdel = os.path.join(results_directory, folder, project_gdb, fc)
        fieldNames = [field.name for field in arcpy.ListFields(fc_fdel)]
        for field in fields:
            if field in fieldNames:
                arcpy.DeleteField_management(fc_fdel, field)

        print folder + ": field deletion complete"

## NUCLEAR delete ##
#
# Deletes the connectivity_metrics_connections and connectivity_metrics_lines feature classes
# this is mainly used in development when I need to start over
# (e.g. I added "date" function later, and I want this field to be the first one created)
# It's not really that nuclear since it doesn't delete any of the simulation results.
#

def deleteFeatureClasses(results_directory, project_gdb, fc_conn_lines, fc_conn_pts):

    folders = os.listdir(results_directory)
    for folder in folders:
        fc_ln = os.path.join(results_directory, folder, project_gdb, fc_conn_lines)
        fc_pt = os.path.join(results_directory, folder, project_gdb, fc_conn_pts)
        arcpy.Delete_management(fc_ln)
        arcpy.Delete_management(fc_pt)

        print folder + ": fc deletion complete"


###             ###
## Program Start ##
###             ###

createMetricFCs(results_directory, results_shp, patch_centroids, fc_conn_lines, fc_conn_pts, project_gdb)
createDateField(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)

calcSourceSink(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)
calcBetweenness(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)
calcCloseness(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)
calcEigenvectorCentrality(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)
calcDegreeCentrality(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)
calcReciprocity(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)
calcDirection(results_directory, project_gdb, fc_conn_lines)

mergeFCs(results_directory, project_gdb, fc_conn_lines, fc_conn_pts, merged_gdb)

###deleteField(results_directory, project_gdb, fc_conn_pts, ["vitality_closeness","constraint"])
####
## NUCLEAR OPTION (keep this commented out) - deletes feature classes
####
###deleteFeatureClasses(results_directory, project_gdb, fc_conn_lines, fc_conn_pts)

# NOTE: if you need to recalculate any metrics for fields that are already created you can just comment out the "continue" in the if statement. Everything else should recalculate since the "addField" is done in the else statement.