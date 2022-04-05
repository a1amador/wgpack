import os, sys
import wget
import numpy as np
import pandas as pd
import netCDF4 as netcdf
from datetime import datetime
from pathlib import Path
from geopy.distance import distance
import matplotlib.pyplot as plt

# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
from wgpack.dportal import veh_list, readDP_CTD, OxHz2DO
from wgpack.nav import get_bearing
from wgpack.arcterx_helpers import readDP_CTD_interp,compute_sogcogDS

# ----------------------------------------------------------------------------------------------------------------------
# Set local paths and import local packages
# ----------------------------------------------------------------------------------------------------------------------
calcofi_path = os.path.join(os.path.abspath(os.path.join('..')),'calcofi')
if calcofi_path not in sys.path:
    sys.path.insert(0, calcofi_path)
from WGcode.WGhelpers import sftp_put_cordcdev,sftp_remove_cordcdev
from WGcode.WGhelpers import veh_list, sftp_put_cordcdev,sftp_remove_cordcdev

# Data service path
ds_folder = os.path.join(str(Path.home()),'src/lri-wgms-data-service')
if ds_folder not in sys.path:
    sys.path.insert(0, ds_folder)
from DataService import DataService

# knots to m/s
kt2mps = 0.514444

# ----------------------------------------------------------------------------------------------------------------------
# Load pasteljet colormap
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib
import h5py
cmapdir = os.path.join(module_path,'mfiles')
matfile = 'pasteljet.mat'
# read-in mat file
hf = h5py.File(os.path.join(cmapdir,matfile), 'r')
pjet = np.array(hf['mycolormap'][:]).T
pjet_cmap = matplotlib.colors.ListedColormap(pjet)

# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------
vnam,tw,prj =  'sv3-251','1','westpac'
# tw = None if tw == 'None' else tw
vid = veh_list[vnam]
print('vehicle:',vnam)
print('project:',prj)

# ----------------------------------------------------------------------------------------------------------------------
# Set start date according to the time window to be considered
# ----------------------------------------------------------------------------------------------------------------------
now = datetime.utcnow()
if tw is not None:
    # use specified time window
    tst = now - pd.Timedelta(days=float(tw))
else:
    # use last 7 days
    tst = now - pd.Timedelta(days=7)
    # use prescribed splash date
    # tst = datetime(2022, 3, 9, 8, 0, 0, 0)

ten = now
# load sven's data
sven_ctd_df = readDP_CTD_interp(vnam,tst,ten)
# # load ole's data
# sven_ctd_df = readDP_CTD_interp(vnam,tst,ten)

# ----------------------------------------------------------------------------------------------------------------------
# Download SSH anomally
# ----------------------------------------------------------------------------------------------------------------------
# Download data?
dwld_flg=True
par_path = 'https://glidervm3.ceoas.oregonstate.edu/ARCTERX/Sync/Shore/KFC'
datadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/SSH')

try:
    str_tmstp = now.strftime('%Y%m%d')
    fnam_ssh = 'dataset-duacs-nrt-global-merged-allsat-phy-l4.'+ str_tmstp +'.nc'
    data_url = os.path.join(par_path,fnam_ssh)
    dfnam    = os.path.join(datadir,fnam_ssh)
    # if file exist, remove it directly
    if dwld_flg:
        if os.path.exists(dfnam):
            os.remove(dfnam)
        print('Beginning file download with wget module...')
        wget.download(data_url, out=dfnam)
        print('Downloaded ' + fnam_ssh)
except:
    str_tmstp = (now - pd.Timedelta(days=1)).strftime('%Y%m%d')
    fnam_ssh = 'dataset-duacs-nrt-global-merged-allsat-phy-l4.'+ str_tmstp +'.nc'
    data_url = os.path.join(par_path,fnam_ssh)
    dfnam    = os.path.join(datadir,fnam_ssh)
    # if file exist, remove it directly
    if dwld_flg:
        if os.path.exists(dfnam):
            os.remove(dfnam)
        print('Beginning file download with wget module...')
        wget.download(data_url, out=dfnam)
        print('Downloaded ' + fnam_ssh)

# Get SSH anomally
SSHnc = netcdf.Dataset(dfnam)
SSHnc.variables.keys()

# ----------------------------------------------------------------------------------------------------------------------
# Get drifter data
# ----------------------------------------------------------------------------------------------------------------------
par_path = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/drifters_leg1')
fnam_drift = 'drifter.csv'
driftdf = pd.read_csv(os.path.join(par_path,fnam_drift),index_col=0,parse_dates=True)

# ----------------------------------------------------------------------------------------------------------------------
# Plot map
# ----------------------------------------------------------------------------------------------------------------------
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, COLORS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# configure general plot parameters
labsz   = 12
fntsz   = 14
sz      = 30
tL      = [tst, now]
lev     = 10

# widths  = [1, 1]
# heights = [1, 0.5, 0.5, 0.5]
# gs_kw = dict(width_ratios=widths, height_ratios=heights)

# --------------------------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------------------------
# configure general cartopy parameters
LAND = NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',
                           facecolor=COLORS['land'])
fig, ax = plt.subplots(1,1,figsize=(12,12), subplot_kw=dict(projection=ccrs.PlateCarree()))
# plt.subplot_mosaic([['ax1', 'ax2'], ['ax3', 'ax4'], ['ax5', 'ax6'], ['ax7', 'ax8']], gridspec_kw=gs_kw,
#                               figsize=(12, 12), subplot_kw=dict(projection=ccrs.PlateCarree()))

# Set map limits
dx = 0.5
dy = 0.5
lonmin = sven_ctd_df['longitude'][-1] - dx / 2
lonmax = sven_ctd_df['longitude'][-1] + dx / 2
latmin = sven_ctd_df['latitude'][-1] - dy / 2
latmax = sven_ctd_df['latitude'][-1] + dy / 2
ax.set_extent([lonmin, lonmax, latmin, latmax])
# used to crop SSH
iilon = np.logical_and(SSHnc['longitude'][:].data>lonmin-0.2,SSHnc['longitude'][:].data<lonmax+0.2)
iilat = np.logical_and(SSHnc['latitude'][:].data>latmin-0.2,SSHnc['latitude'][:].data<latmax+0.2)



# --------------------------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------------------------
# configure labels and grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
tistr = 'Sea level anomally [cm], drifter positions [cyan], and WG temperature [deg C]'
# add land and coastlines
ax.add_feature(LAND)
ax.coastlines(resolution='10m')

# plot Wave Glider location
cf = ax.scatter(sven_ctd_df['longitude'].values, sven_ctd_df['latitude'].values,
                        s=sz, c=sven_ctd_df['T'].values, cmap=pjet_cmap)
ax.scatter(sven_ctd_df['longitude'].values[-1], sven_ctd_df['latitude'].values[-1],
                    s=sz*4, c=sven_ctd_df['T'][-1], marker='d', cmap=pjet_cmap,
                    vmin=sven_ctd_df['T'].min(), vmax=sven_ctd_df['T'].max(), edgecolors='k')

# Sea Surface height anomally
CS = ax.contour(SSHnc['longitude'][iilon].data,
                SSHnc['latitude'][iilat].data,
                SSHnc['sla'][iilat,iilon].data*100,
                lev, colors='gray')
# CS = ax.contour(X, Y, Z)
ax.clabel(CS, CS.levels, inline=True, fontsize=fntsz)

# for did in np.unique(driftdf['Platform-ID']):
#     iidid = driftdf['Platform-ID']==did
#     drift_tmp = driftdf[iidid]
#     drift_tmp = drift_tmp[drift_tmp.index>tst]
#     ax.plot(drift_tmp[' GPS-Longitude(deg)'], drift_tmp[' GPS-Latitude(deg)'],'.c',ms=1)
#     ax.plot(drift_tmp[' GPS-Longitude(deg)'][-1], drift_tmp[' GPS-Latitude(deg)'][-1],'oc')

# tick params
ax.tick_params(labelsize=fntsz)
# colorbar and labels
cbtstr = 'deg C\n'
cb = fig.colorbar(cf, ax=ax, shrink=0.825)
cb.ax.set_title(cbtstr)
ax.set_title(tistr)


# --------------------------------------------------------
# Save figure
# Set local paths and import local packages
from pathlib import Path

loc_folder = os.path.join(str(Path.home()), 'src/calcofi/WGautonomy/auto_plots')
figdir = os.path.join(loc_folder, 'figz', vnam)
figname = 'Map_' + vnam + '.png'
fig.savefig(os.path.join(figdir, figname), dpi=100, bbox_inches='tight')

# ----------------------------------------------------------------------------------------------------------------------
# Upload to CORDCdev - need to be on VPN (sio-terrill pool) to access CORDCdev
# ----------------------------------------------------------------------------------------------------------------------
# TODO: Consider defining these paths in config file (settings.py)
LOCALpath_CD = os.path.join(figdir, figname)
if prj == 'westpac':
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, vnam, figname)
elif prj == 'norse' or prj == 'palau':
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, figname)
else:
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, 'waveglider', figname)
SFTPAttr = sftp_put_cordcdev(LOCALpath_CD, REMOTEpath_CD)
print('done uploading CTD plots to CORDCdev')