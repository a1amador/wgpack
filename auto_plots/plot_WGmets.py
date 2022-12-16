import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats

# local imports
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.dportal import veh_list,readDP_weather
from wgpack.metoc import read_metbuoy_SBD
from wgpack.ndbc import get_ndbc_latest,get_ndbc_nearest
from wgpack.autoplot import mets_bulk_autoplot_timeseries,mets_bulk_autoplot
from wgpack.config import seachest_data_dir
from wgpack.server_helpers import sftp_put_cordcdev,sftp_remove_cordcdev

# --------------------------------------------------------
args = sys.argv
# Select vehicle
# vnam = 'sv3-125'
# Select channels
# channels = 'G012WTLC56K'    #calcofi_2020
# channels = 'C0158P2JJTT'    #app-tests
# Select time window
# tw = 'None'
# tw = '7'
# Select project
# prj = 'palau'
vnam    = args[1]           # vehicle name
vid     = veh_list[vnam]    # vehicle id (Data Portal)
channels= args[2]           # Slack channel
tw      = args[3]           # time window to display specified in days (ends at present time)
tw = None if tw == 'None' else tw
prj     = args[4]           # project folder name (e.g., 'calcofi', 'tfo', etc.)

print('vehicle:',vnam)
print('project:',prj)
# --------------------------------------------------------
# Update wind data - need to be on VPN (sio-terrill pool) and SeaChest needs to be mounted on local machine
REMOTEdir = os.path.join(seachest_data_dir,vnam,'current/nrt/mmb')
# REMOTEdir = '/Users/a1amador/data/test'
f = []
for (dirpath, dirnames, filenames) in os.walk(REMOTEdir):
    f.extend(filenames)
    break
f.sort()
print(f)

# Read-in WXT mets
WXTdf = read_metbuoy_SBD(os.path.join(REMOTEdir,f[0]))
for fi in f[1:]:
    WXTdf = pd.concat([WXTdf, read_metbuoy_SBD(os.path.join(REMOTEdir,fi))])
# sort by date
WXTdf.sort_index(inplace=True)

# Crop dataframe according to the time window to be considered
now = datetime.utcnow()
if tw is not None:
    # use specified time window
    # tst = WXTdf.index.values[-1] - pd.Timedelta(days=float(tw))
    tst = now - pd.Timedelta(days=float(tw))
elif WXTdf.index.values[-1]-pd.Timedelta(days=7)>WXTdf.index.values[0]:
    # use last 7 days
    # tst = WXTdf.index.values[-1] - pd.Timedelta(days=7)
    tst = now - pd.Timedelta(days=7)
else:
    # use splash date
    tst = WXTdf.index.values[0] - pd.Timedelta(days=0)
# crop dataframe accordingly
WXTdf = WXTdf[WXTdf.index>=tst]
# remove outliers:
THRESH_OUTLIER = 3
WXTdf['WindSpeed'][np.abs(stats.zscore(WXTdf['WindSpeed'])>THRESH_OUTLIER)]=np.nan
WXTdf['WindSpeed10'][np.abs(stats.zscore(WXTdf['WindSpeed'])>THRESH_OUTLIER)]=np.nan


# --------------------------------------------------------
# Read in Wave Glider Airmar data from Data Portal
try:
    airdf = readDP_weather(vid=vid,
                           start_date=tst.strftime("%Y-%m-%dT%H:%M:00.000Z"),
                           end_date=now.strftime("%Y-%m-%dT%H:%M:00.000Z"))
    # remove outliers
    airdf['WindSpeed'][np.abs(stats.zscore(airdf['WindSpeed']) > THRESH_OUTLIER)] = np.nan
    # airdf = readDP_weather(vid=vid, start_date=tst.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    # airdf = airdf.replace(9999, np.nan)
except:
    print("Something went wrong when retrieving Airmar data from LRI's Data Portal")
    airdf = pd.DataFrame(columns=['latitude', 'longitude', 'time', 'accessionTime',
                                  'maxWindSpeed', 'nWindSamples', 'pressure','stdDevWindDir',
                                  'stdDevWindSpeed', 'temperature', 'WindSpeed', 'WindDirection'])


# # --------------------------------------------------------
# # TODO: Query the NDBC in real Time - Last 45 Days
# # The latest observation file is available at:
# # http://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt
# # The real time data for all of their buoys can be found at:
# # http://www.ndbc.noaa.gov/data/realtime2/
# # References
# # buoypy:
# # https://github.com/nickc1/buoypy
# # Info about all of NOAA data can be found at:
# # https://www.ndbc.noaa.gov/docs/ndbc_web_data_guide.pdf
# # What all the values mean:
# # https://www.ndbc.noaa.gov/measdes.shtml
#
# # Get 6 hour average of WG coordinates
# tres = 6 # hours
# df_WGcoord = WXTdf[['latitude', 'longitude']].resample(str(tres) + 'H').mean().dropna(how='all')
# if df_WGcoord.empty:
#     print('WXT DataFrame is empty!')
#     df_WGcoord = airdf[['latitude', 'longitude']].resample(str(tres) + 'H').mean().dropna(how='all')
# dthrsh = 30000 # (m)
# bidstr_lst = []
# Bdf_lst= []
# bid_1 = []
# Ldf = get_ndbc_latest()
# # TODO: consider using https://www.ndbc.noaa.gov/data/stations/station_table.txt for Ldf
# for index, row in df_WGcoord.iterrows():
#     WGcoord = (row['latitude'],row['longitude'])
#     # Find closest station
#     bid, dist = get_ndbc_nearest(WGcoord, Ldf, type='mets')[0]
#     # TODO: something
#     if dist<dthrsh and bid!=bid_1:
#         # store NDBC station id
#         bidstr_lst.append(bid)
#         # Get the latest observations for NDBC station
#         # TODO: sometimes this throws an error
#         # try:
#         # except:
#         B = bp.realtime(bid)
#         Bdf = B.txt()
#         # 12 hour data
#         ta = index - pd.Timedelta(hours=tres/2)
#         tb = index + pd.Timedelta(hours=tres/2)
#         Bdf_crop = Bdf[(Bdf.index >= ta) & (Bdf.index <= tb)]
#         # Add coordinates
#         Bdf_crop['LON'] = pd.Series(float(Ldf.loc[bid].LON), index=Bdf_crop.index)
#         Bdf_crop['LAT'] = pd.Series(float(Ldf.loc[bid].LAT), index=Bdf_crop.index)
#         Bdf_lst.append(Bdf_crop)
#         # update bid_1
#         bid_1 = bid
#     elif dist<dthrsh:
#         # store NDBC station id
#         bidstr_lst.append(bid)
#         # 12 hour data
#         ta = index - pd.Timedelta(hours=tres / 2)
#         tb = index + pd.Timedelta(hours=tres / 2)
#         Bdf_crop = Bdf[(Bdf.index >= ta) & (Bdf.index <= tb)]
#         # TODO: fix this warning
#         # Add coordinates
#         Bdf_crop['LON'] = pd.Series(float(Ldf.loc[bid].LON), index=Bdf_crop.index)
#         Bdf_crop['LAT'] = pd.Series(float(Ldf.loc[bid].LAT), index=Bdf_crop.index)
#         Bdf_lst.append(Bdf_crop)

# --------------------------------------------------------
# Plot
# --------------------------------------------------------


# if not Bdf_lst:
#     fig = mets_bulk_autoplot_timeseries(WXTdf,airdf,figshow=True)
# else:
#     fig = mets_bulk_autoplot(WXTdf,airdf,Bdf_lst,bidstr_lst,figshow=True)
fig = mets_bulk_autoplot_timeseries(WXTdf,airdf,figshow=True)
# tmstmp = date.today().strftime('%Y-%m-%d')
tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
fig.suptitle(vnam + ': mets ' + tmstmp, fontsize=16)

# add grid lines
ax = fig.get_axes()
for axi in ax:
    axi.grid()


# --------------------------------------------------------
# Save figure
figdir = os.path.join(seachest_data_dir,vnam,'autoplots')
figname = 'mets_' + vnam + '.jpg'
fig.savefig(os.path.join(figdir,figname), dpi=100, bbox_inches='tight')

# --------------------------------------------------------
# TODO: Consider defining these paths in config file (settings.py)
LOCALpath_CD = os.path.join(figdir,figname)
if prj=='westpac':
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, vnam,figname)
elif prj=='norse' or prj=='palau':
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, figname)
else:
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, 'waveglider', figname)
SFTPAttr = sftp_put_cordcdev(LOCALpath_CD,REMOTEpath_CD)
print('done uploading output to CORDCdev')

# # --------------------------------------------------------
# # TODO: Upload file to Slack at 16 or 23 UTC (9 AM or 4 PM PDT)
# if np.logical_or(datetime.utcnow().hour==16,datetime.utcnow().hour==23):
#     # Inputs:
#     filepath = os.path.join(figdir,figname)
#     title = vnam + " mets"
#     message = vnam + " mets update for " + tmstmp
#     # Report to Slack
#     send_slack_file(channels,filepath,title,message)
#
#     # Check if data is being updated
#     if (now-WXTdf.index[-1]).total_seconds()/3600>2:
#         message = "Latest CORDC mets for " + vnam + " are " \
#                   + str(round((now-WXTdf.index[-1]).total_seconds()/3600,2)) + " hours old"
#         send_slack_message(channels, message)
#
#     if (now-airdf.index[-1]).total_seconds()/3600>2:
#         message = "Latest Airmar mets for " + vnam + " are " \
#                   + str(round((now-airdf.index[-1]).total_seconds()/3600,2)) + " hours old"
#         send_slack_message(channels, message)