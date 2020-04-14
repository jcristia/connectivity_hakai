# Question: What is the spatial and temporal scale of habitat connectivity for different groups of species in the Salish Sea?
# Significance: define the scale that of connectivity for species with different PLDs, which is largely unknown at this point

# these plots are meant to be run for just 1 simulation. There is no averaging of multiple simulations.
# the idea is to build intuition on how we can describe and characterize the results from 1 simulation for 1 PLD


import os
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import log10, floor

###########################
# configuration
###########################

##### Get data for one simulation directory
dir = 'seagrass_20200228_SS201701'

root = r'D:\Hakai\script_runs\seagrass'
shp_merged = 'shp_merged'
conn_ind = 'connectivity_pld{}.shp'
pts = 'dest_biology_pts_sg{}.shp'
subfolders = 9
subfolder = 'seagrass_{}'
plds = ['01', '03', '07', '21', '60']
output_ind = 'output_figs'

# create out directory
if not os.path.exists(os.path.join(root, dir, output_ind)):
    os.mkdir(os.path.join(root, dir, output_ind))
paths = []
for pld in plds:
    p = os.path.join(root, dir, shp_merged, conn_ind.format(pld))
    paths.append(p)
dest_paths = []
for sub in range(1, subfolders + 1):
    dest_pt = os.path.join(root, dir, subfolder.format(sub), 'outputs\\shp', pts.format(sub))
    dest_paths.append(dest_pt)

# universal mapping
sns.set() # switch to seaborn aesthetics
sns.set_style('white')
sns.set_context('notebook')



###########################
# SPATIAL SCALE
# FOR ONE SIMULATION, COMPARE PLDS
# distributions of connection probability and distance
###########################


gdf = gp.GeoDataFrame(columns=['pld', 'prob', 'length'])
for shp, pld in zip(paths, plds):
    df = gp.read_file(shp)
    df['length'] = df.geometry.length
    df['pld'] = pld
    # I'm just realizing now that the from_id column is being created as a string and to_id is a double. This would be something in the biology script, which I'm not going to go back and change at this point. So just deal with it here.
    df.from_id = df.from_id.astype('int64')
    df = df[df['from_id'] != df['to_id']]
    gdf = gdf.append(df[['pld', 'prob', 'length']], ignore_index=True)

##########
# connection probability
# notes:
# I also tried making box plots and violin plots
# However, they looked like shit, even with the data scaled

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
locs, labels = plt.xticks()
ax.set(xticklabels=[10 ** int(i.get_text().replace(u'\u2212', '-'))
                    for i in labels])
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


##################
# distance

# histogram of 1 pld
fig, ax = plt.subplots()
gdf_1 = gdf[gdf['pld']=='01']
sdist2 = sns.distplot(gdf_1.length, kde=False)
sdist2.set(xlabel = 'connection length (m) pld01 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'leng_hist_01_201701.svg'))

# histogram of 01 and 60 pld on log scale
fig, ax = plt.subplots()
ax.set(xscale='log')
# get bins based on largest value in pld60
data = gdf.length[gdf.pld == '60']
max = data.max()
max_round = np.round(max, (-int(floor(log10(abs(max)))))+1)
bins = np.logspace(np.log10(100), np.log10(max_round), 50)
for pld in ['60', '01']:
    data = gdf.length[gdf.pld == pld]
    sns.distplot(data, kde=False, bins=bins, label='pld'+pld)
plt.legend()
ax.set_xlabel('connection length (m) 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'leng_hist_01_60_201701.svg'))

# box plots
# still crappy looking
fig, ax = plt.subplots()
ax.set(yscale='log')
sns.boxplot(gdf.pld, gdf.length)
ax.set_xlabel('PLDs 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'leng_box_01_60_201701.svg'))

# kde of all plds
fig, ax = plt.subplots()
for pld in plds:
    data = gdf.length[gdf.pld == pld]
    sns.kdeplot(np.log10(data), shade=False, ax=ax, label='pld'+pld)
    fig.canvas.draw()
locs, labels = plt.xticks()
for i in labels:
    print(i)
ax.set(xticklabels=[10 ** int(float(i.get_text())) for i in labels])
ax.set_xlabel('connection length 2017-01')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'leng_kde_all_201701.svg'))


# cumulative lines of all plds
fig, ax = plt.subplots()
ax.set(xscale='log')
for pld in plds:
    data = gdf.length[gdf.pld == pld]
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    ax = sns.lineplot(data_sorted, p, label='pld'+pld)
ax.set_xlabel('connection probability 2017-01')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'leng_cumulative_all_201701.svg'))


####################
# distance vs. connection strength

# pld 1
gdf_1 = gdf[gdf.pld=='01']
fig, ax = plt.subplots()
sreg01 = sns.regplot(x=gdf_1.length, y=gdf_1.prob, fit_reg=False)
sreg01.set(xlabel='connectivity distance metres', ylabel='connectivity probability')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_01_201701_nolog.svg'))

# pld 1 log transform
gdf_1 = gdf[gdf.pld=='01']
fig, ax = plt.subplots()
# need to take log of data, not just axes, or else you get a curved regression line
sreg01 = sns.regplot(x=gdf_1.length, y=np.log10(gdf_1.prob))
sreg01.set(xlabel='connectivity distance metres', ylabel='log10(connectivity probability)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_01_201701.svg'))

# pld 60
gdf_1 = gdf[gdf.pld=='60']
fig, ax = plt.subplots()
sreg01 = sns.regplot(x=gdf_1.length, y=np.log10(gdf_1.prob))
sreg01.set(xlabel='connectivity distance metres', ylabel='log10(connectivity probability)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_60_201701.svg'))

# pld 01 and 60 together
fig, ax = plt.subplots()
gdf_1 = gdf[gdf.pld=='01']
gdf_60 = gdf[gdf.pld=='60']
sreg60 = sns.regplot(x=gdf_60.length, y=np.log10(gdf_60.prob))
sreg01 = sns.regplot(x=gdf_1.length, y=np.log10(gdf_1.prob))
ax.set(xlabel='connectivity distance metres', ylabel='log10(connectivity probability)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_01_60_201701.svg'))

# all data colored separately
gdf_2 = gdf[['pld', 'length']]
gdf_2['prob'] = np.log10(gdf['prob'])
slm = sns.lmplot(x='length', y='prob', data=gdf_2, hue='pld', hue_order=['60', '21', '07', '03', '01'])
slm.set(xlabel='connectivity distance metres', ylabel='log10(connectivity probability)')
#plt.show()
slm.savefig(os.path.join(root, dir, output_ind, 'reg_allsep_201701.svg'))

# all data together
gdf_3 = gdf[['pld', 'length']]
gdf_3['prob'] = np.log10(gdf['prob'])
slm = sns.lmplot(x='length', y='prob', data=gdf_3)
slm.set(xlabel='connectivity distance metres', ylabel='log10(connectivity probability)')
#plt.show()
slm.savefig(os.path.join(root, dir, output_ind, 'reg_all_201701.svg'))



###########################
# FOR ONE SIMULATION
# plot of increasing strength of connections
# but only for ones that existed started in PLD
# this is just to demonstrate how connectivity strength may change through time
###########################

gdf = gp.GeoDataFrame(columns=['pld', 'prob', 'length'])
for shp, pld in zip(paths, plds):
    df = gp.read_file(shp)
    df['length'] = df.geometry.length
    df['pld'] = pld
    df.from_id = df.from_id.astype('int64')
    df = df[df['from_id'] != df['to_id']]
    gdf = gdf.append(df[['pld', 'from_id', 'to_id', 'prob', 'length']], ignore_index=True)
# get frequency of connections
gdfc = gdf.groupby(['from_id', 'to_id']).size().reset_index(name='count')
gdfc['unique'] = gdfc.index
# join back to original
gdfm = pd.merge(gdf, gdfc, on=['from_id', 'to_id'])
# remove rows that don't have a count of 5
gdfd = gdfm.drop(gdfm[gdfm['count'] < 5].index)
# get percent change from first value
gdffirst = gdfd.groupby(['unique']).agg(
    firstprob = ('prob', 'first')
    ).reset_index()
gdf_ff = pd.merge(gdfd, gdffirst, on='unique')
gdf_ff['percent_change'] = 1 - gdf_ff.firstprob / gdf_ff.prob
gdf_ff.pld = gdf_ff.pld.astype('int64')
# get means for labels
gdf_mean = gdf_ff.groupby(['pld']).agg(
    meanprobchange = ('percent_change', 'mean')
    ).reset_index()
gdf_mean['meanround'] = np.round(gdf_mean.meanprobchange, 6)

fig, ax = plt.subplots()
ax = sns.pointplot('pld', 'percent_change', data=gdf_ff)
for p in zip(ax.get_xticks(), gdf_mean.meanround):
    ax.text(p[0], p[1]+0.008, p[1], color='g') 
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'connprob_increase_201701.svg'))


###########################
# TEMPORAL SCALE
# distribution of first connection times (on conn lines)
# distribution of settling times of individual particles
# distribution of mortality times
# % of all particles released that settle for each PLD
###########################


# distribution of first connection times (on conn lines)
# only need to get conn lines for pld60 since time_int will be the same for all plds
p = os.path.join(root, dir, shp_merged, conn_ind.format('60'))
gdf = gp.read_file(p)
gdf.from_id = gdf.from_id.astype('int64')
gdf = gdf[gdf['from_id'] != gdf['to_id']]
gdf['time_int_days'] = gdf.time_int * 0.5 / 24.0
fig, ax = plt.subplots()
sd = sns.distplot(gdf.time_int_days, ax=ax, kde=False)
sd.set(xlabel = 'timestep of first connection (days)', ylabel = 'connection count (no self conn)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'dist_timeconn_201701.svg'))


# reading in dest_pts takes a while
# just do it once here and then each plot can modify it as necessary
df = pd.DataFrame()
for dest_pt in dest_paths:
    gdf = gp.read_file(dest_pt)
    gdf = gdf.drop(columns=['geometry','date_start'])
    df = df.append(gdf)
df['time_int_f'] = df.time_int - df.time_int_s
df['time_int_days'] = df.time_int_f * 0.5 / 24.0


# distribution of settling times of individual particles
# remove particles that died or did not settle
df_1 = df[(df.dest_id > 0) & (df.mortstep == -1)]
fig, ax = plt.subplots()
ax.set(yscale='log')
sd = sns.distplot(df_1.time_int_days, ax=ax, kde=False)
sd.set(xlabel = 'timestep of connection (days)', ylabel='particle count')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'dist_timepart_201701.svg'))


# distribution of settling times of individual particles (no self-conn)
df_1 = df[df.dest_id != df.uID]
# remove particles that died or did not settle
df_1 = df_1[(df_1.dest_id > 0) & (df_1.mortstep == -1)]
fig, ax = plt.subplots()
ax.set(yscale='log')
sd = sns.distplot(df_1.time_int_days, ax=ax, kde=False)
sd.set(xlabel = 'timestep of connection (days)', ylabel='particle count (no self conn)')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'dist_timepart_201701_noselfconn.svg'))



# distribution of mortality times
# remove particles that did not die
df_1 = df[df.mortstep > -1]
fig, ax = plt.subplots()
sd = sns.distplot(df_1.time_int_days, ax=ax, kde=False)
sd.set(xlabel = 'timestep of mortality (days)', ylabel = 'particle count')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'dist_timemort_201701.svg'))



# % of all particles released that settle for each PLD
tot_part = len(df)
df_z = df[(df.dest_id != df.uID) & (df.dest_id != -1)]
# this is probably not the most efficient way, but it works
df_1 = df_z[(df_z.time_int - df_z.time_int_s) < (1 * 24 / 0.5)]
df_3 = df_z[((df_z.time_int - df_z.time_int_s) >= (1 * 24 / 0.5)) & ((df_z.time_int - df_z.time_int_s) < (3 * 24 / 0.5))]
df_7 = df_z[((df_z.time_int - df_z.time_int_s) >= (3 * 24 / 0.5)) & ((df_z.time_int - df_z.time_int_s) < (7 * 24 / 0.5))]
df_21 = df_z[((df_z.time_int - df_z.time_int_s) >= (7 * 24 / 0.5)) & ((df_z.time_int - df_z.time_int_s) < (21 * 24 / 0.5))]
df_60 = df_z[((df_z.time_int - df_z.time_int_s) >= (21 * 24 / 0.5)) & ((df_z.time_int - df_z.time_int_s) < (60 * 24 / 0.5))]
df_a = pd.DataFrame(data={'pld': ['0-1','1-3', '3-7', '7-21', '21-60']})
# settle: dest_id > 0, mortstep == -1
df_a['settlecount'] = [
    len(df_1[(df_1.mortstep == -1) & (df_1.dest_id >0)]) / tot_part,
    len(df_3[(df_3.mortstep == -1) & (df_3.dest_id >0)]) / tot_part,
    len(df_7[(df_7.mortstep == -1) & (df_7.dest_id >0)]) / tot_part,
    len(df_21[(df_21.mortstep == -1) & (df_21.dest_id >0)]) / tot_part,
    len(df_60[(df_60.mortstep == -1) & (df_60.dest_id >0)]) / tot_part]
df_a['settleround'] = np.round(df_a.settlecount, 5)
fig, ax = plt.subplots()
ax = sns.barplot('pld', 'settlecount', data=df_a, palette='Blues_d')
ax.set(ylabel='proportion of particles settled (no self conns)')
for p in zip(ax.get_xticks(), df_a.settleround):
    ax.text(p[0], p[1]+0.0005, p[1], color='b') 
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'bar_partsettle_201701_noselfconn.svg'))