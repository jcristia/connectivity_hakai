# see if retention scales with area


nodes = r"C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs_cluster\seagrass\seagrass_20200228_SS201701\conefor\conefor_connectivity_average\nodes_conefor.txt"
conn_all = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs_cluster\seagrass\output_figs_SALISHSEA_ALL\connectivity_average_ALL.shp'

import pandas as pd
import geopandas as gp
import numpy as np
import seaborn as sns

# get just self connections
# average?
# join to areas

df = pd.read_csv(nodes, delimiter = "\t", names=['uid', 'area_log'])
df['area_norm'] = np.power(10, df.area_log)

df_shp = gp.read_file(conn_all)
df_shp = df_shp[df_shp.from_id == df_shp.to_id]
df_firststep = df_shp[df_shp.timeintmax==1] # retention would be ones where they don't leave at all
df_shp = df_shp[['from_id', 'probavgm']]

df_m = df.merge(df_shp, left_on='uid', right_on='from_id')
df_fs = df.merge(df_firststep, left_on='uid', right_on='from_id')

g = sns.lmplot(data=df_m, x='area_log', y='probavgm')
g = sns.lmplot(data=df_m, x='area_norm', y='probavgm')
g = sns.lmplot(data=df_fs, x='area_log', y='probavgm')