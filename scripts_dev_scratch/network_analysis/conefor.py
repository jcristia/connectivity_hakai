# use conefor directed command line to calculate dPC

import os
import arcpy
import numpy as np
import subprocess

arcpy.env.workspace = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output\conefor_test\metrics.gdb'

# Conefor executable:
coneforExe = r'C:\Users\jcristia\Desktop\CONEFOR\conefor_directed\conefor.exe'

# inputs / outputs
seagrass_og = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output\conefor_test\metrics.gdb\seagrass_all_17FINAL_SEL'
#seagrass_og = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\spatial\seagrass\seagrass_all\seagrass.gdb\seagrass_all_17FINAL'
out_folder = r'C:\Users\jcristia\Documents\GIS\MSc_Projects\Hakai\scripts_dev_scratch\network_analysis\output\conefor_test'
conn_fc = r'connectivity_metrics_connections_SEL'
nodefile = r'nodes_conefor.txt'
connections_file = r'conns_conefor.txt'


#######################
# convert feature classes to correct table format

na_n = arcpy.da.FeatureClassToNumPyArray(seagrass_og, ['uID', 'area'])
np.savetxt(os.path.join(out_folder, nodefile), na_n, '%d\t%f') # format as integer with a tab delimiter and a float

na_c = arcpy.da.FeatureClassToNumPyArray(conn_fc, ['from_id', 'to_id', 'prob'])
np.savetxt(os.path.join(out_folder, connections_file), na_c, '%d\t%d\t%f')


#######################


# run Conefor PC calculations
# # NOTE: if in the future I want to batch process mutliple text files, refer back to the conefore examples. There is a way to easily do this.
command = '''{} -nodeFile {} -conFile {} -t prob -PC -BCPC -pcHeur 0.0'''.format(coneforExe, os.path.join(out_folder, nodefile), os.path.join(out_folder, connections_file))
print command
# can change pcHeur to a minimum threshold connecton to consider

#os.system(command) # this works but doesn't print anything to screen

#out = subprocess.check_output(command)
#print out
# NOTE: the above code works, however it does not print the output continuously. Since the process takes a really long time, it may be nice to have this printed to screen.

# Therefore, I should just accept that I will do this step manually from the command line.
# so just 'print command' to get the text then paste this into the command line
# the directed version of conefor only has a windows version right now so I can't run it on the cluster. I could set it up to run it in the computer lab though so that it doesn't take up my computer for 5 days.

# the output texts files are saved in whichever directory I am in on the command line. Therefore, print out_folder and cd directory to it before running the above command.
print '''cd {}'''.format(out_folder)

########################

# put output back into feature class

