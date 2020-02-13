import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import seaborn as sns


##################################################

# compare how long each simulation took
# Run time doesn't get saved with the nc file.
# Also, run time may not be super accurate anyways. My machine was pretty slow when I had chrome open while running.
# 0.01 ~ 35 min
# 0.0005 ~ 84 min
# 0.001 ~ missed it
# 0.004 ~ 28 min
# 0.005 ~ 25 min

##################################################

# Compare particle tracks over 3 days with different window resolutions.


# base matplotlib approach
# code from BMM
# https://nbviewer.jupyter.org/urls/bitbucket.org/salishsea/analysis-ben/raw/tip/notebooks/OpenDrift/OceanParcels_workflow.ipynb# code from BMM

# Comparison of first particle
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title(f'Seagrass Jan 01 2016')
for res, color in zip([0.01, 0.005, 0.004, 0.001, 0.0005], ['c', 'r', 'k', 'g', 'm']):
    res_str = str(res).split('.')[1]
    ds = xr.open_dataset(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\working_simulation\outputs\seagrass12particles_' + res_str + '.nc')
    ax.plot(ds.lon[0, :], ds.lat[0, :], 'o-', color=color, fillstyle='none', label=f'opendrift dx={res}deg',)
#ax.set_xlim([ds.lon[0].min().item() - 0.01, ds.lon[0].max().item() + 0.01])
#ax.set_ylim([ds.lat[0].min().item() - 0.01, ds.lat[0].max().item() + 0.01])
ax.legend()
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()
fig.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\OpenDrift_interpolation_1particle.pdf', bbox_inches='tight')

# Comparison of all 12 particles
fig, axs = plt.subplots(4,3)
axs = axs.reshape(12)
fig.subplots_adjust(hspace=1)
fig.suptitle('Seagrass Jan 01 2016')
for ax, drifter in zip(axs, range(12)):
    ax.set_title(f'Drifter {drifter}')
    for res, color in zip([0.01, 0.005, 0.004, 0.001, 0.0005], ['c', 'r', 'k', 'g', 'm']):
        res_str = str(res).split('.')[1]
        ds = xr.open_dataset(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\working_simulation\outputs\seagrass12particles_' + res_str + '.nc')
        ax.plot(
            ds.lon[drifter, :], ds.lat[drifter, :], label=f'dx={res}deg')
        ax.plot(ds.lon[drifter, 0].item(), ds.lat[drifter, 0].item(), marker='*', ms=10)
ax.legend()
plt.show()
fig.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\OpenDrift_interpolation_12particles.pdf', bbox_inches='tight')


# Seaborn approach
# 1 particle
ds = xr.open_dataset(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\working_simulation\outputs\seagrass12particles_01.nc')
sns.set() # uses seaborn defaults instead of matplotlib
sns.set_context("notebook")
sb_ln = sns.lineplot(x=ds.lon[0, :], y=ds.lat[0, :], sort=False, lw=1, label=f'opendrift dx=0.01deg')
sb_ln.set(xlabel = "Longitude", ylabel = "Latitude")
sb_ln.legend()
plt.show()
fig = sb_ln.get_figure()
fig.savefig(r"C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\OpenDrift_interpolation_1particle_sns.png")


# doing 12 particle approach with seaborn works best if the data is all in 1 dataset and I don't need to do a for loop. I could do a for loop to put the data together, but it is not worth it at this point.



#############################################

# compare particle difference to a baseline of 0.005 (assuming it is the most accurate since it is closest to the real resolution)
# code also from BMM
# note: his code had an error in it when it goes to plot and indexes time
# also, his code may not match up with the plots displayed. Where he normalizes it (second time he calculates d), I don't think he is actually doing this.
# Also, refer to Nancy S presentation I got from Hauke. She calculates the Molcard skill score which I think is where the normalization comes from. This considers how far apart particles are, but also how far they are from their origin.

dts = ['01', '005', '004', '001', '0005']
fig, ax = plt.subplots(figsize=(10, 10))
ds = xr.open_dataset(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\working_simulation\outputs\seagrass12particles_005.nc')
time, lon, lat = [ds[key] for key in ['time', 'lon', 'lat']]

for dt in dts:
    ds = xr.open_dataset(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\working_simulation\outputs\seagrass12particles_' + dt + '.nc')
    d = np.sqrt((ds.lon[:, 1:] - lon[:, 1:])**2 + (ds.lat[:, 1:] - lat[:, 1:])**2)
    #d = d / np.sqrt((ds.lon[:, 1:] - ds.lon[:, 0])**2 + (ds.lat[:, 1:] - ds.lat[:, 0])**2)
    ax.plot(time[1:], d.mean(axis=0), label=f'{dt} deg')
ax.legend()
ax.set_title(f'Average separation distance for all particles when compared to 0.005 window resolution')
plt.show()
fig.savefig(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\sensitivity_testing\window_res\OpenDrift_interpolation_12particles_distancediff.pdf', bbox_inches='tight')


#############################################