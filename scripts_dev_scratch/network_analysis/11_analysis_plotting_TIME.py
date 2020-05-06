# Build off single scenario analysis and look at differences in connectivity through time

# This script also outputs an average of averages for connections and an average for dPC for nodes


import os
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import log10, floor, sqrt



###########################
# configuration
###########################

dirs = [
    'seagrass_20200228_SS201701',
    'seagrass_20200309_SS201705',
    'seagrass_20200309_SS201708',
    'seagrass_20200310_SS201101',
    'seagrass_20200310_SS201105',
    'seagrass_20200310_SS201108',
    'seagrass_20200327_SS201401',
    'seagrass_20200327_SS201405',
    'seagrass_20200327_SS201408',
    ]

root = r'D:\Hakai\script_runs\seagrass'
shp_merged = 'shp_merged'
conn_avg = 'connectivity_average.shp'
conn_ind = 'connectivity_pld{}.shp'
conn_conefor = r'patch_centroids_metrics_commavg.shp'
plds = ['01', '03', '07', '21', '60']
pts = 'dest_biology_pts_sg{}.shp'
subfolders = 9
subfolder = 'seagrass_{}'
output_ind = 'output_figs'
output_all = os.path.join(root, 'output_figs_SALISHSEA_ALL')
overall_indices = r'conefor\conefor_connectivity_{}\overall_indices.txt'
max_area = 620.023012 # total area (after log transform and scale)


# create directories and paths
if not os.path.exists(output_all):
    os.mkdir(output_all)

# universal mapping
sns.set() # switch to seaborn aesthetics
sns.set_style('white')
sns.set_context('notebook')


########################

# NO NEED TO COMPILE THE DF AGAIN UNLESS YOU CHANGE THE DATA
# it takes forever to run
# If you just want to create a new plot, then just load the csv that gets saved below

# % of particles released that settle and die for each scenario (2 lines)
period = []
for dir in dirs:
    dest_paths = []
    for sub in range(1, subfolders + 1):
        dest_pt = os.path.join(root, dir, subfolder.format(sub), 'outputs\\shp', pts.format(sub))
        dest_paths.append(dest_pt)
    df = gp.GeoDataFrame()
    for dp in dest_paths:
        gdf = gp.read_file(dp)
        df = df.append(gdf)
    date = df.iloc[0].date_start
    tot_part = len(df)
    mort = (len(df[df.mortstep > -1]) / tot_part) * 100
    settled_other = (len(df[(df.mortstep == -1) & (df.dest_id >0) & (df.dest_id != df.uID)]) / tot_part) * 100
    settled_self = (len(df[(df.mortstep == -1) & (df.dest_id >0) & (df.dest_id == df.uID)]) / tot_part) * 100
    period.append([date, mort, settled_other, settled_self])

period_df = pd.DataFrame(period, columns=['date', 'mort', 'settled_other', 'settled_self'])
period_df['date'] = period_df.date.str[:7]

period_df.to_csv(os.path.join(output_all, 'settlemort.csv'), index=False)

fig, ax = plt.subplots()
sns.pointplot(x='date', y='settled_other', data=period_df, ci=None, label='settled_other')
# can use matplotlib ax.plot() if I want a legend
sns.pointplot(x='date', y='settled_self', data=period_df, ci=None, label='settled_self')
sns.pointplot(x='date', y='mort', data=period_df, ci=None, label='mort')
#plt.legend()
#plt.show()
fig.savefig(os.path.join(output_all, 'settlemort.svg'))



#############################


# connection frequency and variance

df = gp.GeoDataFrame()
for dir in dirs:
    conn = (os.path.join(root, dir, shp_merged, conn_avg))
    conn = gp.read_file(conn)
    df = df.append(conn)
df['date'] = df.date_start.str[:7]
df['dateYR'] = df.date.str[:4]
df['dateSEA'] = df.date.str[5:8]

def customstd(x):
    var = np.var(x, ddof=0)
    # pandas defaults to using a degrees of freedom of 1, so to get 0 I had to put this in a custom function
    # population vs. sample variance http://www.differencebetween.net/science/mathematics-statistics/difference-between-sample-variance-population-variance/
    std = sqrt(var)
    return std
ldirs = len(dirs)
def customstd9(x):
    mean = sum(x)/ldirs
    var = (sum((x-mean)**2))/ldirs
    std = sqrt(var)
    return std
def conn_mean(x):
    s = x.sum()
    m = s/ldirs
    return m
df_freqstd = df.groupby(['from_id', 'to_id']).agg(
    freq = ('from_id', 'count'),
    probavgm = ('prob_avg', conn_mean),
    prob_stdf0 = ('prob_avg', customstd),
    prob_std9 = ('prob_avg', customstd9),
    geometry = ('geometry', 'first'),
    date = ('date', 'first'),
    dateYR = ('dateYR', 'first'),
    dateSEA = ('dateSEA', 'first'),
    timeintavg = ('time_int', 'mean'),
    timeintmin = ('time_int', 'min'),
    timeintmax = ('time_int', 'max')
    ).reset_index()

gdf = gp.GeoDataFrame(df_freqstd)
gdf.crs = df.crs
gdf.to_file(filename=os.path.join(output_all, 'connectivity_average_ALL.shp'), driver='ESRI Shapefile')


df_noselfconn = df_freqstd[df_freqstd.from_id != df_freqstd.to_id]

# distribution of frequency of connections for connectivity_average
# self connections removed
fig, ax = plt.subplots()
sd = sns.distplot(df_noselfconn.freq, kde=False)
sd.set(xlabel = 'frequency of connection (self-conns removed)')
#plt.show()
fig.savefig(os.path.join(output_all, 'freq_dist_noselfconn.svg'))

# distribution of std of connectivity strength
fig, ax = plt.subplots()
ax.set(xscale='log')
min = df_noselfconn.prob_std9.min()
min_round = 10**(int(floor(log10(abs(min)))))
bins = np.logspace(np.log10(min_round),np.log10(0.5), 10)
sd = sns.distplot(df_noselfconn.prob_std9, bins=bins, kde=False)
sd.set(xlabel = 'standard deviation of connection (self-conns removed)')
plt.show()
fig.savefig(os.path.join(output_all, 'std_dist_noselfconn.svg'))


#########################
# Conefor values through time

# first creat overall gdf of all connectivity averages
# this is what I can use for mapping
# can also be used for average community membership
gdf = gp.GeoDataFrame()
for dir in dirs:
    df = gp.read_file(os.path.join(root, dir, shp_merged, conn_conefor))
    gdf = gdf.append(df)
gdf['date'] = gdf.date_start.str[:7]
gdf['dateYR'] = gdf.date.str[:4]
gdf['dateSEA'] = gdf.date.str[5:8]

gdf_avg = gdf.groupby(['uID']).agg(
    dPC = ('dPC', 'mean'),
    dPCintra = ('dPCintra', 'mean'),
    dPCflux = ('dPCflux', 'mean'),
    dPCconnect = ('dPCconnect', 'mean'),
    dBC_PC = ('dBC_PC', 'mean'),
    dPCv = ('dPC', 'var'),
    dPCintrav = ('dPCintra', 'var'),
    dPCfluxv = ('dPCflux', 'var'),
    dPCconnecv = ('dPCconnect', 'var'),
    dBC_PCv = ('dBC_PC', 'var'),
    dPCstd = ('dPC', 'std'),
    dPCintrast = ('dPCintra', 'std'),
    dPCfluxstd = ('dPCflux', 'std'),
    dPCconnecs = ('dPCconnect', 'std'),
    dBC_PCstd = ('dBC_PC', 'std'),
    #comid_mode = ('comidns', pd.Series.mode),
    #comid_uniq = ('comidns', 'nunique'),  # IDs change each time, so this doesn't work.
    geometry = ('geometry', 'first'),
    date = ('date', 'first'),
    dateYR = ('dateYR', 'first'),
    dateSEA = ('dateSEA', 'first')
    ).reset_index()
gdf_avg['dPCinter'] = gdf_avg.dPCflux + gdf_avg.dPCconnect
gdf_avg = gp.GeoDataFrame(gdf_avg)
gdf_avg.crs = gdf.crs
gdf_avg.to_file(filename=os.path.join(output_all, 'patch_centroids_metrics_ALL.shp'), driver='ESRI Shapefile')

# distribution of std of dPC
fig, ax = plt.subplots()
#ax.set(xscale='log')
#min = gdf_avg.dPCstd.min()
#min_round = 10**(int(floor(log10(abs(min)))))
#bins = np.logspace(np.log10(min_round),np.log10(0.5), 10)
sd = sns.distplot(gdf_avg.dPCstd, kde=False)
sd.set(xlabel = 'standard deviation of dPC')
plt.show()
fig.savefig(os.path.join(output_all, 'std_dist_dPC.svg'))

# read in PCnum txt for conn averages
df_indices = pd.DataFrame()
for dir in dirs:
    indices = os.path.join(root, dir, overall_indices.format('average'))
    df_ind = pd.read_csv(indices, sep='\t', names=['conefor_index', 'value'])
    df_ind['date'] = dir[20:]
    df_indices = df_indices.append(df_ind)
df_indices = df_indices.pivot(index='date', columns='conefor_index', values='value').reset_index()
df_indices['dateYR'] = df_indices.date.str[:4]
df_indices['dateSEA'] = df_indices.date.str[5:8]
df_indices['PCinter(%)'] = df_indices['PCdirect(%)'] + df_indices['PCstep(%)']


# time series of PC values (period to period, and another with years overlaid)
fig, ax = plt.subplots()
sns.pointplot(x='date', y='PCnum', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pcnum_ALL.svg'))

fig, ax = plt.subplots()
sns.pointplot(x='dateSEA', y='PCnum', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pcnum_SEASON.svg'))

fig, ax = plt.subplots()
ax.plot(df_indices.date, df_indices['PCintra(%)'], label='PCintra')
ax.plot(df_indices.date, df_indices['PCdirect(%)'], label='PCflux')
ax.plot(df_indices.date, df_indices['PCstep(%)'], label='PCconnect')
plt.legend()
plt.show()
fig.savefig(os.path.join(output_all, 'pcparts_ALL.svg'))

# change in parts (Intra won't change)
fig, (ax, ax2) = plt.subplots(2,1, sharex=True)
ax.plot(df_indices.date, df_indices['PCdirect(%)'], label='PCflux', marker='o')
ax2.plot(df_indices.date, df_indices['PCstep(%)'], label='PCstep', marker='o')
ax.set_ylim(8, 9)
ax2.set_ylim(0.15, 0.26)
# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.tick_top()
ax2.xaxis.tick_bottom()
# create diagonal breaks
d = .01
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
ax.legend()
ax2.legend()
ax2.set_xlabel('date')
ax.set_ylabel('% PCnum')
plt.show()
fig.savefig(os.path.join(output_all, 'pcinter_format.svg'))


fig, ax = plt.subplots()
sns.pointplot(x='dateSEA', y='PCdirect(%)', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pcdirect_SEASON.svg'))
fig, ax = plt.subplots()
sns.pointplot(x='dateSEA', y='PCstep(%)', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pcstep_SEASON.svg'))
fig, ax = plt.subplots()
sns.pointplot(x='dateSEA', y='PCinter(%)', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pcinter_SEASON.svg'))





# average change of PC values for PLDs
df_indices = pd.DataFrame()
for dir in dirs:
    for pld in plds:
        indices = os.path.join(root, dir, overall_indices.format('pld'+pld))
        df_ind = pd.read_csv(indices, sep='\t', names=['conefor_index', 'value'])
        df_ind['datepld'] = dir[20:] + pld  # this is a hokey work around, but whatever
        df_indices = df_indices.append(df_ind)
df_indices = df_indices.pivot(index='datepld', columns='conefor_index', values='value').reset_index()
df_indices['date'] = df_indices.datepld.str[:6]
df_indices['dateYR'] = df_indices.date.str[:4]
df_indices['dateSEA'] = df_indices.date.str[4:7]
df_indices['pld'] = df_indices.datepld.str[6:9]
df_indices['PCinter(%)'] = df_indices['PCdirect(%)'] + df_indices['PCstep(%)']

fig, ax = plt.subplots()
sns.pointplot(x='dateSEA', y='PCnum', hue='pld', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pld_pcnum_SEA.svg'))

fig, ax = plt.subplots()
sns.pointplot(x='pld', y='PCnum', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pld_pcnum.svg'))
fig, ax = plt.subplots()
sns.pointplot(x='pld', y='PCdirect(%)', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pld_pcdirect.svg'))
fig, ax = plt.subplots()
sns.pointplot(x='pld', y='PCstep(%)', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pld_pcstep.svg'))
fig, ax = plt.subplots()
sns.pointplot(x='pld', y='PCintra(%)', data=df_indices)
plt.show()
fig.savefig(os.path.join(output_all, 'pld_pcintra.svg'))

