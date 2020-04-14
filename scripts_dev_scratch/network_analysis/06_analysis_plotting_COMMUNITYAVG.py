# Question: What is the community connectivity structure?
# Significance: What connections are most common to all members in a community? Most logical way to summarize connectivity across species.

# these plots are meant to be run for just 1 simulation. There is no averaging of multiple simulations.


import os
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import log10, floor
import statsmodels

###########################
# configuration
###########################

##### Get data for one simulation directory
dir = 'seagrass_20200228_SS201701'

root = r'D:\Hakai\script_runs\seagrass'
shp_merged = 'shp_merged'
conn_merge = 'connectivity_average.shp'
output_ind = 'output_figs'

# for comparing to individual sims
conn_ind = 'connectivity_pld{}.shp'
plds = ['01', '03', '07', '21', '60']

# create out directory
if not os.path.exists(os.path.join(root, dir, output_ind)):
    os.mkdir(os.path.join(root, dir, output_ind))

conn_merge = os.path.join(root, dir, shp_merged, conn_merge)

# universal mapping
sns.set() # switch to seaborn aesthetics
sns.set_style('white')
sns.set_context('notebook')



###########################
# SPATIAL SCALE
# distributions of connection probability and distance
###########################

df = gp.read_file(conn_merge)

##########
# connection probability

# histogram self conn
fig, ax = plt.subplots()
ax.set(xscale='log')
min = df.prob_avg.min()
min_round = 10**(int(floor(log10(abs(min)))))
bins = np.logspace(np.log10(min_round),np.log10(1.0), 50)
sdist1 = sns.distplot(df.prob_avg, ax=ax, bins=bins, kde=False)
sdist1.set(xlabel = 'connection probability 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_prob_hist_201701_selfconn.svg'))


# remove self connections for the rest of the plots
df = df[df['from_id'] != df['to_id']]

# histogram
fig, ax = plt.subplots()
ax.set(xscale='log')
min = df.prob_avg.min()
min_round = 10**(int(floor(log10(abs(min)))))
bins = np.logspace(np.log10(min_round),np.log10(1.0), 50)
sdist1 = sns.distplot(df.prob_avg, ax=ax, bins=bins, kde=False)
sdist1.set(xlabel = 'connection probability 2017-01 (no self conn)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_prob_hist_201701.svg'))

# kde
fig, ax = plt.subplots()
sns.kdeplot(np.log10(df.prob_avg), shade=False, ax=ax, label='')
fig.canvas.draw()
locs, labels = plt.xticks()
ax.set(xticklabels=[10 ** int(i.get_text().replace(u'\u2212', '-'))
                    for i in labels])
ax.set_xlabel('connection probability 2017-01 (no self conn)')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_prob_kde_201701.svg'))


##################
# distance
# don't need to go into distance much since distances won't be any different than the pld60. It's just probabilities that change.

df = gp.read_file(conn_merge)
df['length'] = df.geometry.length

# histogram
fig, ax = plt.subplots()
sdist2 = sns.distplot(df.length, kde=False)
sdist2.set(xlabel = 'connection length (m) 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_leng_hist_201701_selfconn.svg'))

# remove self connections for the rest of the plots
df = df[df['from_id'] != df['to_id']]

# histogram on log scale and self connections removed
fig, ax = plt.subplots()
ax.set(xscale='log')
max = df.length.max()
max_round = np.round(max, (-int(floor(log10(abs(max)))))+1)
bins = np.logspace(np.log10(100), np.log10(max_round), 50)
sns.distplot(df.length, kde=False, bins=bins, label='')
plt.legend()
ax.set_xlabel('connection length (m) 2017-01 (no self conn)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_leng_hist_201701_ log.svg'))


####################
# distance vs. connection strength

# 
fig, ax = plt.subplots()
sreg01 = sns.regplot(x=df.length, y=df.prob_avg, fit_reg=False)
sreg01.set(xlabel='connectivity distance metres', ylabel='connectivity probability')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_reg_201701_nolog.svg'))

# log transform
fig, ax = plt.subplots()
sreg01 = sns.regplot(x=df.length, y=np.log10(df.prob_avg), ci=None, fit_reg=True, lowess=True)
sreg01.set(xlabel='connectivity distance metres', ylabel='log10(connectivity probability)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_reg_201701.svg'))

# The residplot() function can be a useful tool for checking whether the simple regression model is appropriate for a dataset. It fits and removes a simple linear regression and then plots the residual values for each observation. Ideally, these values should be randomly scattered around y = 0:
fig, ax = plt.subplots()
sreg01 = sns.residplot(x=df.length, y=np.log10(df.prob_avg))
#plt.show()


#####################
# compare to individual pld data in previous script

df = gp.read_file(conn_merge)
df['length'] = df.geometry.length
df = df[df['from_id'] != df['to_id']]

paths = []
for pld in plds:
    p = os.path.join(root, dir, shp_merged, conn_ind.format(pld))
    paths.append(p)
gdf = gp.GeoDataFrame()
for shp, pld in zip(paths, plds):
    gdft = gp.read_file(shp)
    gdft['pld'] = pld
    gdft['length'] = gdft.geometry.length
    gdft.from_id = gdft.from_id.astype('int64')
    gdft = gdft[gdft['from_id'] != gdft['to_id']]
    gdf = gdf.append(gdft, ignore_index=True)

# compare histograms of connection strength
fig, ax = plt.subplots()
ax.set(xscale='log')
for pld in ['60', '01']:
    data = gdf.prob[gdf.pld == pld]
    min = data.min()
    min_round = 10**(int(floor(log10(abs(min)))))
    bins = np.logspace(np.log10(min_round),np.log10(1.0), 50)
    sns.distplot(data, ax=ax, bins=bins, kde=False, label='pld'+pld)
sns.distplot(df.prob_avg, ax=ax, bins=bins, kde=False, label='connavg')
plt.legend()
ax.set_xlabel('connection probability 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_prob_hist_201701.svg'))

# compare kdes of connection strength
fig, ax = plt.subplots()
for pld in plds:
    data = gdf.prob[gdf.pld == pld]
    sns.kdeplot(np.log10(data), shade=False, ax=ax, label='pld'+pld)
    fig.canvas.draw()
sns.kdeplot(np.log10(df.prob_avg), shade=False, ax=ax, label='conn_avg')
locs, labels = plt.xticks()
ax.set(xticklabels=[10 ** int(i.get_text().replace(u'\u2212', '-'))
                    for i in labels])
ax.set_xlabel('connection probability 2017-01 (no self conn)')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connavg_compare_prob_kde_201701.svg'))

# compare conn strength vs distance relationship
# add connavg dataset to individual pld dataset
gdf_2 = gdf[['pld', 'length']]
gdf_2['prob'] = np.log10(gdf['prob'])
df_2 = df[['length']]
df_2['prob'] = np.log10(df['prob_avg'])
df_2['pld'] = 'connavg'
gdf_3 = gdf_2.append(df_2)

slm = sns.lmplot(x='length', y='prob', data=gdf_3, hue='pld', hue_order=['60', '21', '07', '03', '01', 'connavg'], lowess=True)
slm.set(xlabel='connectivity distance metres', ylabel='log10(connectivity probability)')
#plt.show()
slm.savefig(os.path.join(root, dir, output_ind, 'connavg_comnpare_reg_allsep_201701.svg'))