import os
# Package path
PACKPATH = 'absolute_path_to_wgpack_directory'
# data path
DATAPATH = os.path.join(PACKPATH,'wgpack/data')
# path to rdradcp.m
RDRPATH = os.path.join(PACKPATH,'wgpack/mfiles')
# set Seachest data path
SCPATH = 'absolute_path_to_seachest_mount'
seachest_data_dir = os.path.join(SCPATH,'cordc-data/PROJECTS/WAVEGLIDER')
# set LRI DataService data path
DSPATH = 'absolute_path_to_lri-wgms-data-service_directory'