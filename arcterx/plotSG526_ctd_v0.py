import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
import netCDF4 as netcdf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import seawater as sw
from seawater.library import T90conv


# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
from wgpack.timeconv import epoch2datetime64

# ----------------------------------------------------------------------------------------------------------------------
# Set local paths and import local packages
calcofi_path = os.path.join(os.path.abspath(os.path.join('..')),'calcofi')
if calcofi_path not in sys.path:
    sys.path.insert(0, calcofi_path)
from WGcode.WGhelpers import sftp_put_cordcdev,sftp_remove_cordcdev
verbose = False

# ----------------------------------------------------------------------------------------------------------------------
# Set start date according to the time window to be considered
# tw = '7'
tw=None
now = datetime.utcnow()
if tw is not None:
    # use specified time window
    tst = now - pd.Timedelta(days=float(tw))
else:
    # use last 7 days
    # tst = now - pd.Timedelta(days=7)
    # use prescribed splash date
    tst = datetime(2022, 3, 9, 8, 0, 0, 0)

# ----------------------------------------------------------------------------------------------------------------------
# Remote
SGdatadir = os.path.join(os.path.dirname(seachest_data_dir), 'ARCTERX2022/sg526')

# filename = 'sg526_ARCTERX_1.0m_up_and_down_profile.nc'
filename = 'sg526_ARCTERX_timeseries.nc'
SGfnam = os.path.join(SGdatadir, filename)
SG_data = netcdf.Dataset(SGfnam)
if verbose:
    for v in SG_data.variables.keys():
        print(v)

# ----------------------------------------------------------------------------------------------------------------------
# Remote
SGdatadir = os.path.join(os.path.dirname(seachest_data_dir), 'ARCTERX2022/sg526')

filename = 'sg526_ARCTERX_1.0m_up_and_down_profile.nc'
# filename = 'sg526_ARCTERX_timeseries.nc'
SGfnam = os.path.join(SGdatadir, filename)
SG_data_b = netcdf.Dataset(SGfnam)

if verbose:
    for v in SG_data_b.variables.keys():
        print(v)


# ----------------------------------------------------------------------------------------------------------------------
# Extract data
# ----------------------------------------------------------------------------------------------------------------------
# Sea Glider time
ttSG = pd.to_datetime(epoch2datetime64(SG_data['end_time'][:]))
# SG time index associated with start time
iiaSG = np.abs(ttSG - tst).argmin()
# crop data
SG_stlon = SG_data['start_longitude'][iiaSG:]
SG_stlat = SG_data['start_latitude'][iiaSG:]
SG_enlon = SG_data['end_longitude'][iiaSG:]
SG_enlat = SG_data['end_latitude'][iiaSG:]
ttSG = ttSG[iiaSG:]
# find sea-glider CTD data
# crop by time
ttSGctd = pd.to_datetime(epoch2datetime64(SG_data['ctd_time'][:]))
# SG time index associated with start time
iiaSGctd = np.abs(ttSGctd - tst).argmin()
# Data
SG_T = SG_data['temperature_raw'][iiaSGctd:]
SG_S = SG_data['salinity_raw'][iiaSGctd:]
SG_d = SG_data['ctd_depth'][iiaSGctd:]
ttSGctd = ttSGctd[iiaSGctd:]
# day of year
doySGctd = ttSGctd.dayofyear

# binned data
ttSGb = pd.to_datetime(epoch2datetime64(SG_data_b['time'][:]))
SG_Tb = SG_data_b['temperature'][:].data
SG_Sb = SG_data_b['salinity'][:].data
SG_Pb = SG_data_b['pressure'][:].data
# compute density
T90 = T90conv(SG_Tb)
SG_densityb = sw.pden(SG_Sb, T90, SG_Pb, pr=0)

# ----------------------------------------------------------------------------------------------------------------------
# bin vertical profiles of salinity and temperature
# ----------------------------------------------------------------------------------------------------------------------
nbins = 101
bin_d = np.linspace(0,300,nbins)

# bin temperature
bin_T, binedges_T, binnumber_T = stats.binned_statistic(SG_d, SG_T, statistic='median', bins=bin_d)
# bin salinity
bin_S, binedges_S, binnumber_S = stats.binned_statistic(SG_d, SG_S, statistic='median', bins=bin_d)


# ----------------------------------------------------------------------------------------------------------------------
# plot vertical profiles of salinity and temperature
# ----------------------------------------------------------------------------------------------------------------------
# fntsz = 14
# sz = 6
#
# fig, ax = plt.subplots(1,2,figsize=(8,8))
#
# ax[0].scatter(SG_T,SG_d,c=doySGctd,s=sz,cmap='jet')
# ax[0].plot(bin_T,bin_d[:-1],'--w',linewidth=2)
# ax[0].set_xlabel('Temperature [deg C]',fontsize=fntsz)
# ax[0].set_ylabel('Depth [m]',fontsize=fntsz)
# ax[0].grid(':')
# ax[0].invert_yaxis()
#
# cf = ax[1].scatter(SG_S,SG_d,c=doySGctd,s=sz,cmap='jet')
# ax[1].plot(bin_S,bin_d[:-1],'--w',linewidth=2)
# ax[1].invert_yaxis()
# ax[1].set_xlabel('Salinity [psu]',fontsize=fntsz)
# ax[1].set_xlim(33,35.5)
# ax[1].grid(':')
#
# cbtstr = 'doy\n'
# cb = fig.colorbar(cf, ax=ax[1], shrink=0.95)
# cb.ax.set_title(cbtstr)

# ----------------------------------------------------------------------------------------------------------------------
# pcolor plots
# ----------------------------------------------------------------------------------------------------------------------

fntsz = 14
labsz = 12
import cmocean
from matplotlib.dates import DateFormatter


# Pcolor plot
# fig, ax = plt.subplots(3,1,figsize=(10,7.5),sharex=True,sharey=True)
fig, ax = plt.subplots(3,1,figsize=(20,14),sharex=True,sharey=True)

X,Y = np.meshgrid(ttSGb, SG_data_b['depth'][:].data)
# Temperature
tistr ='temperature'
Z = SG_data_b['temperature'][:].data
# ax[0].contourf(X, Y, Z.T,levels=10, cmap=plt.cm.jet)
cf = ax[0].pcolor(X, Y, Z.T, cmap=plt.cm.jet)
ax[0].set_ylabel('depth [m]',fontsize=fntsz)
# ax[0].invert_yaxis()
ax[0].set_title(tistr)
# colorbar and labels
cbtstr = 'deg C'
cb = fig.colorbar(cf, ax=ax[0], shrink=0.95)
cb.ax.set_title(cbtstr,fontsize=fntsz)


# Salinity
tistr ='salinity'
Z = SG_data_b['salinity'][:].data
cf = ax[1].pcolor(X, Y, Z.T)
# ax[1].invert_yaxis()
ax[1].set_ylabel('depth [m]',fontsize=fntsz)
ax[1].set_title(tistr)
# colorbar and labels
cbtstr = 'psu'
cb = fig.colorbar(cf, ax=ax[1], shrink=0.95)
cb.ax.set_title(cbtstr,fontsize=fntsz)
# ax[1].set_title(tistr)


# Density
tistr ='density'
Z = SG_densityb
cf = ax[2].pcolor(X, Y, Z.T,cmap=cmocean.cm.dense)
# cf = ax[2].pcolormesh(X, Y, Z.T, cmap=cmocean.cm.dense,shading='gouraud')
# cf = ax[2].pcolormesh(Xn, Yn, Zn.T, cmap=cmocean.cm.dense)
ax[2].invert_yaxis()
ax[2].set_ylabel('depth [m]',fontsize=fntsz)
# colorbar and labels
cbtstr = 'kg/m$^3$'
cb = fig.colorbar(cf, ax=ax[2], shrink=0.95)
cb.ax.set_title(cbtstr)
ax[2].set_title(tistr,fontsize=fntsz)

# Define the date format
date_form = DateFormatter("%m-%d")
ax[2].xaxis.set_major_formatter(date_form)

# tick params
ax[0].tick_params(labelsize=labsz)
ax[1].tick_params(labelsize=labsz)
ax[2].tick_params(labelsize=labsz)

# # save figure
# savefig = False
# dirFile = '/Users/a1amador/Documents/SIO/Projects/ARCTERX/figz'
# fnam = 'SG_TSDpcolor.jpg'
# if savefig:
#     print(dirFile)
#     fig.savefig(os.path.join(dirFile,fnam),dpi=300, bbox_inches='tight')

# --------------------------------------------------------
# Save figure
# Set local paths and import local packages
from pathlib import Path
loc_folder = os.path.join(str(Path.home()), 'src/calcofi/WGautonomy/auto_plots')
figdir = os.path.join(loc_folder, 'figz', 'arcterx')
figname = 'TSDen_SG526.png'
fig.savefig(os.path.join(figdir, figname), dpi=100, bbox_inches='tight')
# --------------------------------------------------------
# Upload to CORDCdev - need to be on VPN (sio-terrill pool) to access CORDCdev
# TODO: Consider defining these paths in config file (settings.py)
LOCALpath_CD = os.path.join(figdir, figname)
REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', 'westpac', 'arcterx', figname)
SFTPAttr = sftp_put_cordcdev(LOCALpath_CD,REMOTEpath_CD)
print('done uploading Sea Glider plots to CORDCdev')

