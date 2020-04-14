# for one community average simulation:
# distribution of cluster area size
# distribution of # nodes per cluster

import os
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import seaborn as sns
from math import log10, floor, ceil
import numpy as np

project = 'seagrass_20200228_SS201701'
shp_poly = 'shp_merged\patch_clusters_convexhull.shp'

root = r'D:\Hakai\script_runs\seagrass'
output_ind = 'output_figs'
shp_ply = os.path.join(root, project, shp_poly)
out_figs = os.path.join(root, project, output_ind)


df = gp.read_file(shp_ply)

# plots
sns.set()
sns.set_style('white')
sns.set_context('notebook')

# dist cluster area size
fig, ax = plt.subplots()
ax.set(xscale='log')
min = df['area'].min()
max = df['area'].max()
min_round = 10**(int(floor(log10(abs(min)))))
max_round = 10**(int(ceil(log10(abs(max)))))
bins = np.logspace(np.log10(min_round),np.log10(max_round), 50)
sdist1 = sns.distplot(df['area'], ax=ax, bins=bins, kde=False)
sdist1.set(xlabel = 'community area (m^2) 2017-01')
#plt.show()
fig.savefig(os.path.join(out_figs, 'community_area_dist_201701.svg'))

fig, ax = plt.subplots()
ax.set(xscale='log')
sdist1 = sns.boxplot(df['area'])
sdist1.set(xlabel = 'community area (m^2) 2017-01')
#plt.show()
fig.savefig(os.path.join(out_figs, 'community_area_box_201701.svg'))


# dist # of nodes per cluster
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.pt_count, ax=ax, kde=False)
sdist1.set(xlabel = 'patches per community 2017-01')
#plt.show()
fig.savefig(os.path.join(out_figs, 'community_ptcount_dist_201701.svg'))

fig, ax = plt.subplots()
sdist1 = sns.boxplot(df.pt_count)
sdist1.set(xlabel = 'patches per community 2017-01')
#plt.show()
fig.savefig(os.path.join(out_figs, 'community_ptcount_box_201701.svg'))

plt.show()