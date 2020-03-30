#
# place questions here once finalized
#

import os
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import log10, floor

###########################
# paths
###########################

root = r'D:\Hakai\script_runs\seagrass'
paths_years = {
    '2011':['seagrass_20200310_SS201101', 'seagrass_20200310_SS201105', 'seagrass_20200310_SS201108'],
    #'2014':[],
    '2017':['seagrass_20200228_SS201701', 'seagrass_20200309_SS201705', 'seagrass_20200309_SS201708']
    }
shp_merged = 'shp_merged'
conn_ind = 'connectivity_pld{}.shp'
conn_avg = 'connectivity_average.shp'
plds = ['01', '03', '07', '21', '60']

output_ind = 'output_figs'
output_all = 'salishsea_output_all'


###########################
# FOR ONE SIMULATION, COMPARE PLDS
# distributions of connection strength and distance
###########################

##### Get data for one simulation directory

dir = 'seagrass_20200228_SS201701'
# create out directory
if not os.path.exists(os.path.join(root, dir, output_ind)):
    os.mkdir(os.path.join(root, dir, output_ind))
paths = []
for pld in plds:
    p = os.path.join(root, dir, shp_merged, conn_ind.format(pld))
    paths.append(p)

gdf = gp.GeoDataFrame(columns=['pld', 'prob', 'length'])
for shp, pld in zip(paths, plds):
    df = gp.read_file(shp)
    df['length'] = df.geometry.length
    df['pld'] = pld
    # I'm just realizing now that the from_id column is being created as a string and to_id is a double. This would be something in the biology script, which I'm not going to go back and change at this point. So just deal with it here.
    df.from_id = df.from_id.astype('int64')
    df = df[df['from_id'] != df['to_id']]
    gdf = gdf.append(df[['pld', 'prob', 'length']], ignore_index=True)


# notes:
# I also tried making box plots and violin plots
# However, they looked like shit, even with the data scaled


###########################
# connection probability
###########################

sns.set() # switch to seaborn aesthetics
sns.set_style('white')
sns.set_context('notebook')

# histogram of 1 pld
fig, ax = plt.subplots()
ax.set(xscale='log')
gdf_1 = gdf[gdf['pld']=='01']
# get minimum value to set bin minimum, and round to nearest .1
min = gdf_1.prob.min()
min_round = 10**(int(floor(log10(abs(min)))))
bins = np.logspace(np.log10(min_round),np.log10(1.0), 50)
sdist1 = sns.distplot(gdf_1.prob, ax=ax, bins=bins, kde=False)
sdist1.set(xlabel = 'connection probability 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'prob_hist_01_201701.svg'))

# histogram of 01 and 60 pld
fig, ax = plt.subplots()
ax.set(xscale='log')
for pld in ['60', '01']:
    data = gdf.prob[gdf.pld == pld]
    min = data.min()
    min_round = 10**(int(floor(log10(abs(min)))))
    bins = np.logspace(np.log10(min_round),np.log10(1.0), 50)
    sns.distplot(data, ax=ax, bins=bins, kde=False, label='pld'+pld)
plt.legend()
ax.set_xlabel('connection probability 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'prob_hist_01_60_201701.svg'))

# kde of one pld
fig, ax = plt.subplots()
gdf_1 = gdf[gdf['pld']=='01']
skde = sns.kdeplot(np.log10(gdf_1.prob), shade=True, ax=ax)
fig.canvas.draw()
# matplotlib draws labels afterwards and it doesn't recognize log scale so we have to do log labels manually
locs, labels = plt.xticks()
# u2212 is the matplotlib's medium dash for negative numbers.
ax.set(xticklabels=[10 ** int(i.get_text().replace(u'\u2212', '-'))
                    for i in labels])
# Or for scientific notation:
# ax.set(xticklabels=["$10^" + i.get_text() + "$" for i in labels])
ax.set_xlabel('connection probability 2017-01')
skde.get_legend().remove()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'prob_kde_01_201701.svg'))


# kde of all plds
fig, ax = plt.subplots()
for pld in plds:
    data = gdf.prob[gdf.pld == pld]
    sns.kdeplot(np.log10(data), shade=False, ax=ax, label='pld'+pld)
    fig.canvas.draw()
# matplotlib draws labels afterwards and it doesn't recognize log scale so we have to do labels manually
locs, labels = plt.xticks()
# u2212 is the matplotlib's medium dash for negative numbers.
ax.set(xticklabels=[10 ** int(i.get_text().replace(u'\u2212', '-'))
                    for i in labels])
# Or for scientific notation:
# ax.set(xticklabels=["$10^" + i.get_text() + "$" for i in labels])
ax.set_xlabel('connection probability 2017-01')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'prob_kde_all_201701.svg'))


# cumulative lines of all plds
# this is a bit of a hack to get a kde type plot on a log axis
# kde has a 'cumulative' type, but it requires another package to work
fig, ax = plt.subplots()
ax.set(xscale='log')
for pld in plds:
    data = gdf.prob[gdf.pld == pld]
    data_sorted = np.sort(data)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    ax = sns.lineplot(data_sorted, p, label='pld'+pld)
ax.set_xlabel('connection probability 2017-01')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'prob_cumulative_all_201701.svg'))


###########################
# distance
###########################




###########################
# distance vs. connection strength
###########################