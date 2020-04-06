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
patchshp = 'patch_centroids_metrics.shp'
output_ind = 'output_figs'

# create out directory
if not os.path.exists(os.path.join(root, dir, output_ind)):
    os.mkdir(os.path.join(root, dir, output_ind))

patchshp = os.path.join(root, dir, shp_merged, patchshp)

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
#ax.set(xscale='log')
#min = df.prob_avg.min()
#min_round = 10**(int(floor(log10(abs(min)))))
#bins = np.logspace(np.log10(min_round),np.log10(1.0), 50)
sdist1 = sns.distplot(df.dPC, ax=ax, kde=False)
sdist1.set(xlabel = 'dPC 2017-01')
plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_dpc_hist_201701.svg'))


# histogram varPC
fig, ax = plt.subplots()
sdist1 = sns.distplot(df.varPC, ax=ax, kde=False)
sdist1.set(xlabel = 'varPC 2017-01')
plt.show()
fig.savefig(os.path.join(root, dir, output_ind, 'conefor_varpc_hist_201701.svg'))


# box plots


# how each metric scales with area