import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# --------------------------------------------------------
# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
from wgpack.dportal import veh_list
from wgpack.server_helpers import sftp_put_cordcdev,sftp_remove_cordcdev

# # from auto_plot_settings import prj_folder,seachest_data_dir
# # Set local paths and import local packages
# if prj_folder not in sys.path:
#     sys.path.insert(0, prj_folder)
# from WGcode.WGhelpers import veh_list, sftp_put_cordcdev,readDP_AMPSPowerStatusSummary
# from WGcode.slackhelpers import send_slack_file,send_slack_message

# Data service path
ds_folder = os.path.join(str(Path.home()),'src/lri-wgms-data-service')
if ds_folder not in sys.path:
    sys.path.insert(0, ds_folder)
from DataService import DataService

# --------------------------------------------------------
args = sys.argv
# Select vehicle
# vnam = 'sv3-253'
# vnam = 'magnus'
# vnam = 'sv3-251'
# vnam = 'sv3-125'
# Select channels
# channels = 'G012WTLC56K'    #calcofi_2020
# channels = 'C0158P2JJTT'    #app-tests
# Select time window
# tw = 'None'
# tw = '7'
# Select project
# prj = 'calcofi'
# prj = 'tfo'
vnam    = args[1]           # vehicle name
vid     = veh_list[vnam]    # vehicle id (Data Portal)
tw      = args[2]           # time window to display specified in days (ends at present time)
tw = None if tw == 'None' else tw
prj     = args[3]           # project folder name (e.g., 'calcofi', 'tfo', etc.)
# vnam,vid,channels,tw,prj =  'sv3-253','131608817','C0158P2JJTT','7','tfo'
# vnam,channels,tw,prj =  'sv3-022','C0158P2JJTT','2','palau'

print('vehicle:',vnam)
print('project:',prj)
# --------------------------------------------------------
# Set start date according to the time window to be considered
now = datetime.utcnow()
if tw is not None:
    # use specified time window
    tst = now - pd.Timedelta(days=float(tw))
else:
    # use last 7 days
    tst = now - pd.Timedelta(days=7)
# # Read in Power status from Data Portal
# pwrDPdf = readDP_AMPSPowerStatusSummary(vid=vid, start_date=tst.strftime("%Y-%m-%dT%H:%M:%S.000Z"))

# Read in Power status from Data Service
try:
    # instantiate data-service object
    ds = DataService()
    # To get report names
    # ds.report_list
    # Get report
    start_date = tst.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_date = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    out = ds.get_report_data('Amps Power Summary Report', start_date, end_date, [vnam])
    # Convert to pandas dataframe
    # c3DSdf = pd.json_normalize(out['report_data'][0]['vehicle_data'])
    pwrDPdf = pd.json_normalize(out['report_data'][0]['vehicle_data']['recordData'])

    # set timeStamp column as datetimeindex
    pwrDPdf = pwrDPdf.set_index(pd.DatetimeIndex(pwrDPdf['gliderTimeStamp'].values))
    pwrDPdf.drop(columns=['gliderTimeStamp'], inplace=True)
    # sort index
    pwrDPdf.sort_index(inplace=True)
    flg_plt = True
except:
    print("Something went wrong when retrieving power data from LRI's Data Service")
    flg_plt = False

# --------------------------------------------------------
# Plot
# --------------------------------------------------------
labsz = 8
fntsz = 10
tL = [tst, now]
fig, ax = plt.subplots(2,1,figsize=(12, 6),sharex=True)
# totalBatteryPower
ax[0].plot(pwrDPdf['totalBatteryPower']/1000,'-r')
ax[0].set_ylabel('Total Power Remaining [Wh]', fontsize=fntsz)
# Power in/out
ax[1].plot(tL,[0,0],'--',color='gray')
l2,=ax[1].plot(pwrDPdf['solarPowerGenerated']/1000,'-',color='gold',label='Solar Power Generated')
l3,=ax[1].plot(pwrDPdf['outputPortPower']/1000,'-b',label='Power Consumed')
ax[1].legend(fontsize=fntsz)
# ax[1].legend((l1,l2,l3),('Battery Charging Power','Solar Power Generated','Power Consumed'), fontsize=fntsz)
ax[1].set_ylabel('Power [W]', fontsize=fntsz)
ax[1].set_xlabel('Time [UTC]', fontsize=fntsz)
ax[1].set_xlim(tL[0],tL[-1])

# tmstmp = date.today().strftime('%Y-%m-%d')
tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
fig.suptitle(vnam + ': power status ' + tmstmp, fontsize=16)

# Define the date format
date_form = DateFormatter("%m/%d/%y %H")
ax[1].xaxis.set_major_formatter(date_form)

# add grid lines
ax = fig.get_axes()
for axi in ax:
    axi.grid()
fig.show()

# --------------------------------------------------------
# Save figure
figdir = os.path.join(seachest_data_dir,vnam,'autoplots')
figname = 'power_' + vnam + '.png'
fig.savefig(os.path.join(figdir,figname), dpi=100, bbox_inches='tight')

# --------------------------------------------------------
# Upload to CORDCdev - need to be on VPN (sio-terrill pool) to access CORDCdev
# TODO: Consider defining these paths in config file (settings.py)
LOCALpath_CD = os.path.join(figdir,figname)
if prj=='westpac':
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, vnam,figname)
elif prj=='norse' or prj=='palau':
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, figname)
else:
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, 'waveglider', figname)
SFTPAttr = sftp_put_cordcdev(LOCALpath_CD,REMOTEpath_CD)
print('done uploading power plots to CORDCdev')

# # --------------------------------------------------------
# # Upload file to Slack at 16 or 23 UTC (9 AM or 4 PM PDT)
# if np.logical_or(datetime.utcnow().hour==16,datetime.utcnow().hour==23):
#     # Inputs:
#     filepath = os.path.join(figdir,figname)
#     title = vnam + " power"
#     message = vnam + " power update for " + tmstmp
#     # Report to Slack
#     send_slack_file(channels,filepath,title,message)