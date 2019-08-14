import netCDF4 as nc

filename = r'D:\Hakai\NEP36-N26_5d_20140524_20161228_grid_U_20160712-20160721.nc'
#filename = r'D:\Hakai\fvcom_results\calvert03_tides_verified_EDITED.nc'
#filename = r'C:\Users\jcristia\Downloads\opendrift_test.tar\opendrift_test\SalishSea_1h_20171110_20171110_opendrift.nc'
#filename = r'C:\Miniconda3\envs\opendrift_p2\Lib\site-packages\opendrift-master\tests\test_data\2Feb2016_Nordic_sigma_3d\Nordic-4km_SLEVELS_avg_00_subset2Feb2016.nc'
dataset = nc.Dataset(filename, "r+")
print dataset
variables = dataset.variables.keys()
for variable in variables:
    print variable
for dim in dataset.dimensions.values():
    print dim
for var in dataset.variables.values():
    print var

time = dataset.variables["time_counter"][:]
depth = dataset.variables["depthu"][:] # this seems to be the center of a slice
depth_b = dataset.variables["depthu_bounds"][:] # these are the bounds of that slice
lon = dataset.variables["nav_lon"]
lat = dataset.variables["nav_lat"]
lon[0] # notice how there are zeros at the end of every slice

from opendrift.models.oceandrift import OceanDrift
o = OceanDrift(loglevel=20)

from opendrift.readers import reader_ROMS_native
reader_roms = reader_ROMS_native.Reader(r'C:\Miniconda3\envs\opendrift_p2\Lib\site-packages\opendrift-master\tests\test_data\2Feb2016_Nordic_sigma_3d\Nordic-4km_SLEVELS_avg_00_subset2Feb2016.nc')
print reader_roms

from opendrift.readers import reader_netCDF_CF_unstructured
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\fvcom_results\calvert03_tides_verified_EDITED.nc')
print reader_hakai

from opendrift.readers import reader_NEMO_pacific_JC
reader_nemo_pacific_4 = reader_NEMO_pacific_JC.Reader(r'D:\Hakai\NEP36-N26_5d_20140524_20161228_grid_U_20160712-20160721.nc')
print reader_nemo_pacific_4


