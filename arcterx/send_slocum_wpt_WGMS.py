'''
This script updates Slocum Glider latest position on WGMS. Slocum positions are obtained from
https://glidervm3.ceoas.oregonstate.edu/ARCTERX/
'''

import sys,os
import pandas as pd

# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.nav_autonomy import sendWPT

# ----------------------------------------------------------------------------------------------------------------------
# Read in Slocum Glider posits from https://glidervm3.ceoas.oregonstate.edu/ARCTERX/
# ----------------------------------------------------------------------------------------------------------------------
# Create an URL object
url = 'https://glidervm3.ceoas.oregonstate.edu/ARCTERX/'
# Returns list of all tables on page
Assetdf = pd.read_html(url)[0]
Slocdf = Assetdf[Assetdf['Asset']=='Slocum']

# ----------------------------------------------------------------------------------------------------------------------
# Send Slocum Glider posits as WPT on WGMS
# ----------------------------------------------------------------------------------------------------------------------
vnam = 'sv3-253'
# WaypointName  = 'Slocum ' + Slocdf['Time (UTC)'].values[-1] + ' UTC'
WaypointName  = 'Slocum'
WaypointNumber= '55'
lat = str(Slocdf['Latitude'].values[-1])
lon = str(Slocdf['Longitude'].values[-1])
# send waypoint
sendWPT(vnam,WaypointName,WaypointNumber,lat,lon)
