# plot connection strength vs. distance (with distance being calculated in ArcGIS in a different script)
# plot distribution of distances where no connection was made
# environment: community_detection (I'm getting lazy)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp

# euclidean distances for all combinations of nodes
distances = r"C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\distance_analysis\distance_analysis_mapping\euc_lines_ALL.csv"
# overall averaged established connections
conns = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs_cluster\seagrass\output_figs_SALISHSEA_ALL\connectivity_average_ALL.shp'

# For some reason, there were a few meadows that did not have their ocean distances calculated in the previous script, but they did not throw an error so I did not know it at the time.
# Identify the uIDs of the meadows that did not have their distances calculated. There should be 5.
# It looks like they weren't calculated because they were land locked by the rasterization, but still in water.
# However, 2 of them were not landlocked and I have no idea why they don't calculate. Therefore, I'm not going to take the time to run them again. I have enough points to establish the general relationship.
# There are 5 in total, randomly distributed in space.

# calc frequency and look for singletons
dist_df = pd.read_csv(distances)
dist_df_freq = dist_df.origin_id.value_counts()
# print out and check in map manually so that I can see why they aren't 

# uIDs not considered:
uIDs_not = [720, 192, 569, 762, 867]
# NOTE: 720 was not calculated at all, so it won't have a self connection

# drop these values and self connections from the dataframe
# oh wait, by just dropping self connections, I will get rid of these
dist_df = dist_df[dist_df.DestID != dist_df.origin_id]

# access my overall averaged connection dataset
# drop self connections
# drop any lines that are to/from the ones I couldn't calculate
conns_df = gpd.read_file(conns)
conns_df = conns_df.drop(['geometry', 'dateSEA', 'dateYR', 'date'], 1)
conns_df = conns_df[conns_df.from_id != conns_df.to_id]
conns_df = conns_df[~conns_df.from_id.isin(uIDs_not)]
conns_df = conns_df[~conns_df.to_id.isin(uIDs_not)]
conns_df = pd.DataFrame(conns_df) # geopandas to pandas

# use time_int to categorize by PLD
conns_df.loc[conns_df.timeintavg < 2880, 'pld'] = 60
conns_df.loc[conns_df.timeintavg < 1008, 'pld'] = 21
conns_df.loc[conns_df.timeintavg < 336, 'pld'] = 7
conns_df.loc[conns_df.timeintavg < 144, 'pld'] = 3
conns_df.loc[conns_df.timeintavg < 48, 'pld'] = 1

# pandas merge, keep all records from distance dataframe
df_merge = dist_df.merge(conns_df, how='left', left_on=['origin_id', 'DestID'], right_on=['from_id', 'to_id'])

# for any distance combinations that do not have a connection strength, fill as 0
df_merge.probavgm = df_merge.probavgm.fillna(0)
# convert to km
df_merge['distkm'] = (df_merge.Shape_Leng)/1000.0

# create dfs with and without zeros and also a big one with duplicates of connections so that I can compare with and without zero values in one plot with seaborn
df_merge['withzeros'] = 'yes'
df_nozero = df_merge[df_merge.probavgm > 0]
df_nozero.withzeros = 'no'
df_concat = pd.concat([df_merge, df_nozero]).reset_index(drop=True)







############
# PLOTTING
############

# # TESTING
# # PLOT: no zeros
# sns.set()
# sns.set_style('white')
# sns.set_context('paper')
# f = sns.lmplot(x="distkm", y="probavgm", data=df_nozero, scatter=True, fit_reg=True, logx=True, scatter_kws={"s": 1, 'alpha':0.3})
# # PLOT: with zeros, two lines to compare with and without zero values, do not show points
# sns.set()
# sns.set_style('white')
# sns.set_context('paper')
# f = sns.lmplot(x="distkm", y="probavgm", data=df_concat, hue='withzeros', scatter=False, fit_reg=True, logx=True)
# # NO BUENO, the curves go below zero. This isn't a linear relationship. I'll need to fit my own curve instead of relying on the fitting from Seaborn.



# FITTING WITH MY OWN EQUATION
# Resource:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

# # I first tried this, thinking that I would define an asymptote:
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
# # in this equation:
# # a is the starting y value when x is zero
# # -b is how quickly the slope changes
# # c is the asymptote
# but I can't actually say that there is an asymptote present. That doesn't really make sense, especially if I log transform my data.
# it is more of a power curve (exponential curve)
# This is what Treml 2012 did. "Data fit to a negative exponential curve, with best fit of y = 0.664x**(-1.032), Rsquared = 0.22."
# However, I found that I can't get a good fit at all unless I first log transform my data. It's simply not possible without it.
# At first I did not want to do this because I wanted to compare when I do and do not include connections with a strength of zero (which can't be log transformed).
# However, for the sake of getting a good fit, I will drop that. So...

#df_nozero['problog'] = np.log10(df_nozero.probavgm)
df_nozero['probperc'] = df_nozero.probavgm * 100
df_nozero['probperclog'] = np.log10(df_nozero.probperc)
def func(x, a, b):
    return a * x**(b)
popt, pcov = curve_fit(func, df_nozero.distkm, df_nozero.probperclog)
# "Use non-linear least squares to fit a function, f, to data."
print(popt) # to see a,b
#print(pcov)

# get 95% confidence interval
# this ended up not being as straight forward as I thought it would be.
# This is the most straighforward approach and is an answer from 2020:
# https://prodevsblog.com/questions/118243/confidence-interval-for-exponential-curve-fit/
a, b = unc.correlated_values(popt, pcov)
px = np.linspace(0,200, 300)
py = a * px**(b)
nom = unp.nominal_values(py)
std = unp.std_devs(py)

# PLOT 1
# plot data (run this all together so that the line plots on top)
###########
sns.set()
sns.set_style('white')
sns.set_context('paper')
#fig, ax = plt.subplots()
#f = sns.regplot(x="distkm", y="problog", data=df_nozero, scatter=True, fit_reg=False, scatter_kws={"s": 1, 'alpha':0.3}, ax=ax) # plot points
f = sns.lmplot(
    x="distkm", 
    y="probperclog", 
    data=df_nozero, 
    hue='pld',
    hue_order=[60,21,7,3,1], 
    scatter=True, 
    fit_reg=False, 
    scatter_kws={"s": 1, 'alpha':1},
    legend=True,
    legend_out=False,
    ) # plot points
plt.plot(px, nom, 'dimgray') # plot fitted curve
#plt.plot(px, nom - 2 * std) # if you want to plot just the bounding lines of the CI
#plt.plot(px, nom + 2 * std)
## or plot it as a fill:
#plt.fill_between(px, nom - 2 * std, nom + 2 * std, color='gray', alpha=0.2) # plot CI
# However, I won't plot the CI. it doesn't really show up, and is it relevant on a log scale?

# LEGEND...
# I need to use lmplot instead of regplot because it allows me to use hue, but
# it is a facetgrid type of plot (higher order?), so the access to the legend is
# different (there is not ax= attribute in lmplot like in regplot).
# I need to order my points so that 60 draws on bottom, which means putting it
# first in the list, and then the legend orders this way.
# There is no easy way to reorder the legend (there is if this was a regplot
# though). So, I need to manually create a legend. Oh well.
pal = sns.color_palette() # I'm just using the default
pal_hex = pal.as_hex()[:5]
pal_hex.reverse()
handles = []
labels = ['1', '3', '7', '21', '60']
import matplotlib.lines as mlines
for h, l in zip(pal_hex, labels):
    blue_line = mlines.Line2D([], [], color=h, linestyle='None', marker='o', markersize=2, label=l)
    handles.append(blue_line)
plt.legend(title='PD (days)', frameon=False, handles=handles)

f.set(xlim=(0,200))
f.set(xlabel='Distance (km)', ylabel=r'log$_{10}$ Connection probability (%)')
f.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\publications_figures\chap1\fig08_conn_v_dist_log.svg')
###########

# get r squared
residuals = df_nozero.probperclog- func(df_nozero.distkm, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((df_nozero.probperclog-np.mean(df_nozero.probperclog))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

# EQUATION
# y = -0.52(x^0.38)
# r2 = 0.44



# PLOT 2
# for reference, plot non transformed points:
sns.set()
sns.set_style('white')
sns.set_context('paper')
f = sns.lmplot(x="distkm", y="probavgm", data=df_nozero, scatter=True, scatter_kws={"s": 1, 'alpha':0.3}, fit_reg=False)
f.set(xlim=(0,200), ylim=(0,1))
f.set(xlabel='Distance (km)', ylabel=r'Connection Strength')
sns.despine(top=False, right=False, left=False, bottom=False)
f.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\publications_figures\chap1\fig09_conn_v_dist.svg')


# PLOT 3
# plot distribution of distances for zero prob connections
# get all rows where a connection wasn't made
df_zero = df_merge[df_merge.probavgm == 0]
sns.set()
sns.set_style('white')
sns.set_context('paper')
d = pd.Series(df_zero.distkm)
h = sns.distplot(d, kde=False)
h.set(xlabel='Distance (km)', ylabel=r'Count of unrealized connections')
figh = h.get_figure() # distplot is deprecated, so I need to save in this old way
figh.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\publications_figures\chap1\fig10_noconns_distribution.svg')

# investigative plot
# histogram with connections removed if they are at least formed in ONE direction (e.g. 1-2 or 2-1)
# then I can say: how well we can predict strength by distance if we can reliably predict the direction of flow
# identify opposite connections and remove them. Do a join back to connection dataset, but switch keys.
# do a left exclusive join (so, find where opposite established connections exist, then take the portion of the left dataframe that DOES NOT include these connections)
df_opposite = df_zero.merge(df_nozero, how='outer', left_on=['origin_id', 'DestID'], right_on=['to_id', 'from_id'], indicator=True).query('_merge=="left_only"')
# column names get screwed up in this join. Need to recalc distance in km
df_opposite['distkm_y'] = (df_opposite.Shape_Leng_x)/1000.0
sns.set()
sns.set_style('white')
sns.set_context('paper')
d = pd.Series(df_opposite.distkm_y)
h = sns.distplot(d, kde=False)
# as we can see, it doesn't change the distribution much at all. If perhaps if drastically reduce the shorter distance counts then maybe it would be worth saying something about, but it does not.