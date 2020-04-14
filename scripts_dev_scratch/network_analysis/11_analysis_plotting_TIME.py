# Build off single scenario analysis and look at differences in connectivity through time


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
plds = ['01', '03', '07', '21', '60']
pts = 'dest_biology_pts_sg{}.shp'
subfolders = 9
subfolder = 'seagrass_{}'
output_ind = 'output_figs'
output_all = os.path.join(root, 'output_figs_SALISHSEA_ALL')


# create directories and paths
if not os.path.exists(output_all):
    os.mkdir(output_all)

# universal mapping
sns.set() # switch to seaborn aesthetics
sns.set_style('white')
sns.set_context('notebook')


########################


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

period_df.to_file(os.path.join(output_all, 'settlemort.csv'), index=False)

fig, ax = plt.subplots()
sns.pointplot(x='date', y='settled_other', data=period_df, ci=None, label='settled_other')
# can use matplotlib ax.plot() if I want a legend
sns.pointplot(x='date', y='settled_self', data=period_df, ci=None, label='settled_self')
sns.pointplot(x='date', y='mort', data=period_df, ci=None, label='mort')
#plt.legend()
plt.show()
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

# distribution of variance of connectivity strength
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
# time series of PC values (period to period, and another with years overlaid)
# average change of PC values for PLDs (see if they are always increasing with PLD)



#########################
# Change in community membership