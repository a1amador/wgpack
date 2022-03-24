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
# url = 'https://glidervm3.ceoas.oregonstate.edu/ARCTERX/'
# # Returns list of all tables on page
# Assetdf = pd.read_html(url)[0]
# Slocdf = Assetdf[Assetdf['Asset']=='Slocum']

# Create an URL object
url = 'https://glidervm3.ceoas.oregonstate.edu/ARCTERX/Sync/Shore/Slocum/pos.csv'
Slocdf = pd.read_csv(url)
# set time stamp column as datetimeindex
Slocdf = Slocdf.set_index(pd.DatetimeIndex(Slocdf['t'].values))
Slocdf.sort_index(inplace=True)


# ----------------------------------------------------------------------------------------------------------------------
# Send Slocum Glider posits and destination waypoint as WPT on WGMS
# ----------------------------------------------------------------------------------------------------------------------
# Slocum Glider posits
vnam = 'sv3-253'
WaypointName  = 'Slocum posit ' + Slocdf['t'][-1] + ' UTC'
# WaypointName  = 'Slocum'
WaypointNumber= '55'
lat = str(Slocdf['lat'][-1])
lon = str(Slocdf['lon'][-1])
# send waypoint
sendWPT(vnam,WaypointName,WaypointNumber,lat,lon)

# Slocum Glider destination waypoint
vnam = 'sv3-253'
WaypointName  = 'Slocum target wpt'
WaypointNumber= '56'
lat = str(Slocdf['wptLat'][-1])
lon = str(Slocdf['wptLon'][-1])
# send waypoint
sendWPT(vnam,WaypointName,WaypointNumber,lat,lon)
