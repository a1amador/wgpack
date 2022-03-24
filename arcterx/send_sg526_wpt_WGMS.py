'''
This script updates Sea Glider 526 latest position on WGMS. SG526 positions are obtained from Seachest
'''

import sys,os
import pandas as pd
import netCDF4 as netcdf

# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.nav_autonomy import sendWPT
from wgpack.config import seachest_data_dir
from wgpack.timeconv import epoch2datetime64

# ----------------------------------------------------------------------------------------------------------------------
# LOAD Sea Glider data from Seachest
# ----------------------------------------------------------------------------------------------------------------------
SGdatadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/sg526')
# filename = 'sg526_ARCTERX_1.0m_up_and_down_profile.nc'
filename = 'sg526_ARCTERX_timeseries.nc'
SGfnam    = os.path.join(SGdatadir,filename)
SG_data = netcdf.Dataset(SGfnam)
# Sea Glider time
ttSG = pd.to_datetime(epoch2datetime64(SG_data['end_time'][:]))
ttSG = str(ttSG[-1])
# Sea Glider posits
SG_enlon = SG_data['end_longitude'][-1]
SG_enlat = SG_data['end_latitude'][-1]

# ----------------------------------------------------------------------------------------------------------------------
# Send Slocum Glider posits as WPT on WGMS
# ----------------------------------------------------------------------------------------------------------------------
vnam = 'sv3-251'
WaypointName  = 'SG526 ' + ttSG + ' UTC'
# WaypointName  = 'Slocum'
WaypointNumber= '55'
lat = str(SG_enlat)
lon = str(SG_enlon)
# send waypoint
sendWPT(vnam,WaypointName,WaypointNumber,lat,lon)
