# test how long it takes to seed from all patches

# temporary fix for when running on anything other than mank01:
#import sys
#sys.path.append("/Linux/src/opendrift-master")

from opendrift.models.oceandrift import OceanDrift
o = OceanDrift(loglevel=0)

from opendrift.readers import reader_netCDF_CF_unstructured
#reader_hakai = reader_netCDF_CF_unstructured.Reader(r'/home/jcristia/models/cal03brcl_21_0003_EDITED.nc')
reader_hakai = reader_netCDF_CF_unstructured.Reader(r'D:\Hakai\models\fvcom_results\cal03brcl_21_0003_EDITED.nc')

from datetime import datetime
from datetime import timedelta
time_step = timedelta(hours=4)
num_steps = 1
for i in range(num_steps):
    o.seed_from_shapefile(r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\seagrass\seagrass_all\seagrass_all_17FINAL.shp', number=5000000, time=reader_hakai.start_time + i*time_step)

o.elements_scheduled

# YUP, it takes too long. It start out fast but then gets very slow. I probably don't want to exceed the 400,000 per release. Therefore, I will need to split the seagrass and do separate runs and stitch together the shapefiles after.