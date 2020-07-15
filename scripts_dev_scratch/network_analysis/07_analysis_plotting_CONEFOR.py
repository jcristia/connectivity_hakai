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




###############################################
# PCnum vs. averaged PLD
# find max AL2 of these scenarios to compare to - perhaps just put as a text box
# plot individual points and put a regression line through, x axis will be categorical?

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

root = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_runs_cluster\seagrass'
plds = ['01', '03', '07', '21', '60']
overall_indices = r'conefor\conefor_connectivity_pld{}\overall_indices.txt'

df_indices = pd.DataFrame()
for pld in plds:
    for dir in dirs:
        indices = os.path.join(root, dir, overall_indices.format(pld))
        df_ind = pd.read_csv(indices, sep='\t', names=['conefor_index', 'value'])
        df_ind['pld'] = int(pld)
        df_ind = df_ind.pivot(index='pld', columns='conefor_index', values='value').reset_index()
        df_ind['datepld'] = dir[20:] + pld
        df_indices = df_indices.append(df_ind)

################## THIS IS THE ONE
df_indlog = df_indices
df_indlog['pldlog'] = np.log(df_indices.pld) # log pld
df_indlog['season'] = df_indlog.datepld.str[4:6].astype(int) # color by season
df_indlog['season'] = df_indlog.season.replace([1,5,8],['winter', 'spring', 'summer'])
sns.set()
sns.set_style('white')
sns.set_context('paper')
colors = ['#377eb8', '#4daf4a', '#ff7f00']
sns.set_palette(colors)
# I'm not going to present this as a regression. Instead I will just present the points, then draw a line connecting the means.
# would make more sense to use scatterplot than lmplot, but scatterplot doesn't have jitter
#sns.scatterplot(x='pldlog', y='PCnum', data=df_indlog, hue='season', legend="full", x_jitter=0.01)
f = sns.lmplot(x="pldlog", y="PCnum", data=df_indlog, fit_reg=False, hue='season', x_jitter=0.02, legend='full')
f._legend.remove()
sns.lineplot(x='pldlog', y='PCnum', data=df_indlog, ci=95, err_style='band', color='grey')
# band is a 95% confidence interval for the means
f.set(xlabel='ln PLD')
sns.despine(top=False, right=False)
f.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\publications_figures\chap1\fig04_PCnumLOG.svg')
# to maybe do in the future:
# draw the lineplot behind the points. However, lmplot doesn't use 'ax', so I can't figure out how to do this.
# instead of transforming the data, just plot on a custom ln axis
# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale
# https://stackoverflow.com/questions/43463431/custom-logarithmic-axis-scaling-in-matplotlib
########################



############# TESTING
sns.set()
sns.set_style('white')
sns.set_context('notebook')
# x as discrete point plot
sns.pointplot(x='pld', y='PCnum', data=df_indices)
# x as continuous, lineplot
sns.lineplot(x='pld', y='PCnum', data=df_indices, err_style="bars", ci=95, marker="o")

# regression
# show means and confidence intervals
fig, ax = plt.subplots(figsize=(13,10))
sreg2 = sns.regplot(x="pld", y="PCnum", data=df_indices, x_estimator=np.mean, ci=95, order=2, line_kws={'linestyle': '--'})
ax.set(xlim=(-5,65))

# regression logx
sns.set()
sns.set_style('white')
sns.set_context('paper')
sns.set_palette('gray')
fig, ax = plt.subplots()
sreg1 = sns.regplot(x="pld", y="PCnum", data=df_indices, logx=True, x_jitter=0.1)
ax.set(xlim=(-5,65))
ax.set_xlabel('PLD (days)')
#fig.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\publications_figures\chap1\fig04_PCnum.svg')
# How to explain in paper:
# estimate a linear regression of the form y ~ log(x), but plot the scatterplot and regression model in the input space
# no longer using that one
# there's nothing wrong with this one, it's just that:
# linearly doesn't make sense, even with logx. It's not a good fit and i know there is a curve.
# However with order=2, it curves down at the end, which even though the data does do that, I know that it is not the relationship I am looking for.
# It's most likely a logarithmic relationship, even if my data doesn't show that.
# Therefore, I would need to model something with an asymptote.
# Also, if you check the residuals, even with order=2, it is still not a good fit. This indicates there is some other relationship going on and I would need to model it differently.

sns.residplot(x="pld", y="PCnum", data=df_indices, order=2)

# log pld
df_indlog = df_indices
df_indlog['pldlog'] = np.log(df_indices.pld)
sreg1 = sns.regplot(x="pldlog", y="PCnum", data=df_indlog, lowess=True, x_jitter=0.02)
# lowess does local polynomial thing where it just looks at a local neighborhood. This is good for fitting a line to noisy data where you may not be able to detect the relationship.
# However, I think I will have a hard time defending this.

# do 1/x (this is similar to what would happen if there was an asymptote)
df_ind1 = df_indices
df_ind1['pld1'] = 1 / df_indices.pld
sreg1 = sns.regplot(x="pld1", y="PCnum", data=df_ind1, order=1, x_jitter=0.01)
sns.residplot(x="pld1", y="PCnum", data=df_ind1, order=1)
# this has a good fit, indicating to me that I need to figure out how to model a curve that reaches an asymptote

# try to fit my own curve
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
def f(x, a, b, n):
    return a * x ** n  / (x ** n + b)
y= df_indices.PCnum
x= df_indices.pld 
popt, pcov = curve_fit(f, x, y, p0=[445., 441., 2.])
plt.scatter(x, y)
plt.plot(x, f(x, *popt), 'r-')
plt.show()

# GLM
# https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab
# This still gave me a straight line
import numpy as np
from numpy.random import uniform, normal, poisson, binomial
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
x = df_indlog.pldlog.values
y = df_indlog.PCnum.values
exog, endog = sm.add_constant(x), y
# Poisson regression
mod = sm.GLM(endog, exog, family=sm.families.Poisson(link=sm.families.links.log()))
res = mod.fit()
display(res.summary())
y_pred = res.predict(exog)
idx = x.argsort()
x_ord, y_pred_ord = x[idx], y_pred[idx]
plt.plot(x_ord, y_pred_ord, color='m')
plt.scatter(x, y,  s=20, alpha=0.8)
plt.xlabel("X")
plt.ylabel("Y")