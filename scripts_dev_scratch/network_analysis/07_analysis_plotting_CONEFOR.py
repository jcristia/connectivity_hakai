# Question: Are certain meadows more important than others for maintaining different forms of connectivity?
# Significance: Show that there is a lot of variation across the seascape, and show that values don't necessarily scale with one another (e.g. being large doesn't mean you are a connector). These are the potential meadows to prioritize for conservation.

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
patchshp = 'patch_centroids_metrics_commavg.shp'
output_ind = 'output_figs'

# create out directory
if not os.path.exists(os.path.join(root, dir, output_ind)):
    os.mkdir(os.path.join(root, dir, output_ind))

patchshp = os.path.join(root, dir, shp_merged, patchshp)


##### for comparing individual PLD runs
plds = ['01', '03', '07', '21', '60']
overall_indices = r'conefor\conefor_connectivity_pld{}\overall_indices.txt'

max_area = 620.023012 # total area (after log transform and scale)


# universal mapping
sns.set() # switch to seaborn aesthetics
sns.set_style('white')
sns.set_context('notebook')




###########################
# distributions of PC
###########################

df = gp.read_file(patchshp)


# histogram dPC
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.dPC, ax=ax, kde=False)
sdist1.set(xlabel = 'dPC 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpc_hist_201701.svg'))


# histogram varPC
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.varPC, ax=ax, kde=False)
sdist1.set(xlabel = 'varPC 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_varpc_hist_201701.svg'))


# box plots
fig, ax = plt.subplots()
sdist1 = sns.boxplot(df.dPC)
sdist1.set(xlabel = 'dPC 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpc_box_201701.svg'))

# histogram of each dPC metric
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.dPCintra, ax=ax, kde=False)
sdist1.set(xlabel = 'dPCintra 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpcintra_hist_201701.svg'))
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.dPCflux, ax=ax, kde=False)
sdist1.set(xlabel = 'dPCflux 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpcflux_hist_201701.svg'))
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.dPCconnect, ax=ax, kde=False)
sdist1.set(xlabel = 'dPCconnect 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpcconnect_hist_201701.svg'))
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.dPCinter, ax=ax, kde=False)
sdist1.set(xlabel = 'dPCinter 2017-01')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpcinter_hist_201701.svg'))

# intra and inter
fig, ax = plt.subplots()
sns.distplot(df.dPCintra, ax=ax, kde=False, label='intra')
sns.distplot(df.dPCinter, ax=ax, kde=False, label= 'inter')
sdist1.set(xlabel = 'dPCintra and dPCinter 2017-01')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpcintrainter_hist_201701.svg'))


# how each metric scales with area
fig, ax = plt.subplots()
sreg01 = sns.regplot(x=np.log10(df['area']), y=df.dPC, lowess=True)
sreg01.set(xlabel='log10(area)', ylabel='dPC')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_area_dPC_201701.svg'))

fig, ax = plt.subplots()
sreg01 = sns.regplot(x=np.log10(df['area']), y=df.dPCintra, lowess=True)
sreg01.set(xlabel='log10(area)', ylabel='dPCintra')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_area_dPCintra_201701.svg'))

fig, ax = plt.subplots()
sreg01 = sns.regplot(x=np.log10(df['area']), y=df.dPCflux, lowess=True)
sreg01.set(xlabel='log10(area)', ylabel='dPCflux')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_area_dPCflux_201701.svg'))

fig, ax = plt.subplots()
sreg01 = sns.regplot(x=np.log10(df['area']), y=df.dPCconnect, lowess=True)
sreg01.set(xlabel='log10(area)', ylabel='dPCconnect')
plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'reg_area_dPCconnect_201701.svg'))




################
# plots to compare PC between PLDs
################

# read in overall indices, rows to columns, add in PLD column
df_indices = pd.DataFrame()
for pld in plds:
    indices = os.path.join(root, dir, overall_indices.format(pld))
    df_ind = pd.read_csv(indices, sep='\t', names=['conefor_index', 'value'])
    df_ind['pld'] = pld
    df_indices = df_indices.append(df_ind)
df_indices = df_indices.pivot(index='pld', columns='conefor_index', values='value').reset_index()

# change in PCnum
fig, ax = plt.subplots()
sns.pointplot(x='pld', y='PCnum', data=df_indices)
#plt.show()
# I think this shows that it is likely reaching a max possible connectivity
fig.savefig(os.path.join(root, dir, output_ind, 'plds_PCnum_201701.svg'))

# change in PCnum with max
fig, ax = plt.subplots()
sns.pointplot(x='pld', y='PCnum', data=df_indices)
plt.axhline(y=max_area, ls='--')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'plds_PCnum_wMax_201701.svg'))

# change in PCintra, PCflux, PCconnect
fig, ax = plt.subplots()
ax.plot(df_indices.pld, df_indices['PCintra(%)'], label='PCintra')
ax.plot(df_indices.pld, df_indices['PCdirect(%)'], label='PCflux')
ax.plot(df_indices.pld, df_indices['PCstep(%)'], label='PCconnect')
plt.legend()
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'plds_PCall_201701.svg'))

# change in PCflux, PCconnect
fig, (ax, ax2) = plt.subplots(2,1, sharex=True)
ax.plot(df_indices.pld, df_indices['PCdirect(%)'], label='PCflux', marker='o')
ax2.plot(df_indices.pld, df_indices['PCstep(%)'], label='PCstep', marker='o')
ax.set_ylim(7, 9)
ax2.set_ylim(0.15, 0.21)
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
ax2.set_xlabel('PLD')
ax.set_ylabel('% PCnum')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'plds_PCfluxconnect_201701.svg'))



# change in PCintra, PCinter
fig, (ax, ax2) = plt.subplots(2,1, sharex=True)
ax.plot(df_indices.pld, df_indices['PCintra(%)'], label='PCintra', marker='o')
ax2.plot(df_indices.pld, df_indices['PCstep(%)'] + df_indices['PCdirect(%)'], label='PCinter', marker='o')
ax.set_ylim(91, 93)
ax2.set_ylim(7, 9)
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
ax2.set_xlabel('PLD')
ax.set_ylabel('% PCnum')
#plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'plds_PCintrainter_201701.svg'))