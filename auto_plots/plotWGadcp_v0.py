import sys,os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import RDRPATH,seachest_data_dir
from wgpack.adcp import readADCP_raw,motion_correct_ADCP_gps_h5py
from wgpack.timeconv import datetime2matlabdn
from wgpack.nav import get_bearing
from wgpack.helperfun import nan_interpolate

# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------
args = sys.argv
vnam    = args[1]           # vehicle name
# vid     = veh_list[vnam]  # vehicle id (Data Portal)
channels= args[2]           # Slack channel
tw      = args[3]           # time window to display specified in days (ends at present time)
tw = None if tw == 'None' else tw
prj     = args[4]           # project folder name (e.g., 'calcofi', 'tfo', etc.)
# vnam, channels, tw, prj = 'sv3-022', 'C0158P2JJTT', '7', 'palau'

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
    # tst = now - pd.Timedelta(days=7)
    # use prescribed splash date
    tst = datetime(2022, 1, 21, 19, 55, 17, 469297)

# ----------------------------------------------------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------------------------------------------------
# file path and name for input .PD0 file
# Update wind data - need to be on VPN (sio-terrill pool) and SeaChest needs to be mounted on local machine
REMOTEdir = os.path.join(seachest_data_dir,vnam,'current/nrt/adcp')
f = []
for (dirpath, dirnames, filenames) in os.walk(REMOTEdir):
    f.extend(filenames)
    break
f.sort()

# # -------------------------------------------------------------------------------------
# # Process raw ADPC binaries (all files)
# # -------------------------------------------------------------------------------------
# for fnam in f:
#     adcp_filepath_in = os.path.join(REMOTEdir, fnam)
#     print(adcp_filepath_in)
#     # file path and name for output .mat file
#     fnam_out, file_extension = os.path.splitext(adcp_filepath_in)
#     if not os.path.exists(fnam_out+'.mat'):
#         # read-in ADCP binaries
#         adcpr = readADCP_raw(adcp_filepath_in, RDRPATH, adcp_filepath_out=fnam_out, eng_exit=True)
#     elif os.path.getmtime(fnam_out+'.mat')<os.path.getmtime(fnam_out+'.PD0'):
#         print('Updating')
#         print(fnam_out+'.mat')
#         # read-in ADCP binaries
#         adcpr = readADCP_raw(adcp_filepath_in, RDRPATH, adcp_filepath_out=fnam_out, eng_exit=True)

# -------------------------------------------------------------------------------------
# Process raw ADPC binaries (daily files)
# -------------------------------------------------------------------------------------
for fnam in f:
    adcp_filepath_in = os.path.join(REMOTEdir, fnam)
    print(adcp_filepath_in)
    # file path and name for output .mat file
    fnam_out, file_extension = os.path.splitext(adcp_filepath_in)
    # extract dates
    spl_str = fnam.split('_')
    if np.logical_and(not os.path.exists(fnam_out+'.mat'),len(spl_str)>3):
        # read-in ADCP binaries
        adcpr = readADCP_raw(adcp_filepath_in, RDRPATH, adcp_filepath_out=fnam_out, eng_exit=True)
    elif os.path.getmtime(fnam_out+'.mat')<os.path.getmtime(fnam_out+'.PD0') and len(spl_str)>3:
        print('Updating')
        print(fnam_out+'.mat')
        # read-in ADCP binaries
        adcpr = readADCP_raw(adcp_filepath_in, RDRPATH, adcp_filepath_out=fnam_out, eng_exit=True)

# -------------------------------------------------------------------------------------
# Correct ADCP velocities for Wave Glider motion using GPS-derived velocities
# -------------------------------------------------------------------------------------
# Inputs:
dt_gps = 120   # Time-averaging interval for GPS-derived velocities (s)
dt_avg = 120*2 # Time-averaging interval for motion-corrected ADCP velocities (s)
errvel_THRESH = 0.1 # error velocity threshold (values greater than threshold get masked with nans)
mag_dec = None

# # -------------------------------------------------------------------------------------
# # Process: motion correction (all files)
# # -------------------------------------------------------------------------------------
# find and sort processed ADCP files
# fmat = []
# for filenames in os.listdir(REMOTEdir):
#     if filenames.endswith('.mat'):
#         # Consider only .mat files
#         fmat.append(filenames)
# fmat.sort()
# print(fmat)

# -------------------------------------------------------------------------------------
# Process: motion correction (daily files)
# -------------------------------------------------------------------------------------
# find and sort processed ADCP files
fmat = []
for filenames in os.listdir(REMOTEdir):
    if filenames.endswith('.mat'):
        # Consider only .mat files
        # extract dates
        spl_str = filenames.split('_')
        if len(spl_str)>3:
            fdate = datetime.strptime(spl_str[-1][:-4], '%Y%m%d')
            if fdate>tst:
                # Consider only daily .mat files after start date
                fmat.append(filenames)
fmat.sort()

# Collect ADCP data and motion correct velocities
time_adcp_lst =[]
Nvel_lst,Evel_lst,errvel_lst=[],[],[]
Nvelf_lst,Evelf_lst=[],[]
pitch_lst,roll_lst,heading_lst,heading_float_lst = [],[],[],[]
nav_elongitude_lst,nav_elatitude_lst = [],[]
for matfile in fmat:
    try:
        print(matfile)
        # read-in processed adcp data
        hf = h5py.File(os.path.join(REMOTEdir,matfile), 'r')
        adcpr = hf['adcpr']
        # apply motion correction
        adcpm = motion_correct_ADCP_gps_h5py(adcpr, dt_gps, dt_avg, mag_dec=mag_dec)
        # time
        time_adcp_lst.append(adcpm['time'])
        # ADCP velocities
        Nvel_lst.append(adcpm['Nvel'])
        Evel_lst.append(adcpm['Evel'])
        errvel_lst.append(adcpm['err_vel'])
        # ADCP velocities (low-pass filtered)
        Nvelf_lst.append(adcpm['Nvelf'])
        Evelf_lst.append(adcpm['Evelf'])
        kk = len(adcpm['time'])
        # nav variables
        pitch_lst.append(np.array(adcpr['pitch'][:kk]).flatten())
        roll_lst.append(np.array(adcpr['roll'][:kk]).flatten())
        heading_lst.append(np.array(adcpr['heading'][:kk]).flatten())
        heading_float_lst.append(adcpm['heading_float'])
        nav_elongitude_lst.append(np.array(adcpr['nav_elongitude'][:kk]).flatten())
        nav_elatitude_lst.append(np.array(adcpr['nav_elatitude'][:kk]).flatten())
    except:
        print('error in motion correction algorithm')

# concatenate lists
time_adcp = np.concatenate(time_adcp_lst, axis=0 )
Nvel = np.concatenate(Nvel_lst, axis=1 )
Evel = np.concatenate(Evel_lst, axis=1 )
Nvelf = np.concatenate(Nvelf_lst, axis=1 )
Evelf = np.concatenate(Evelf_lst, axis=1 )
errvel = np.concatenate(errvel_lst, axis=1 )
pitch_adcp = np.concatenate(pitch_lst, axis=0)
roll_adcp = np.concatenate(roll_lst, axis=0)
heading_adcp = np.concatenate(heading_lst, axis=0)
heading_float = np.concatenate(heading_float_lst, axis=0)
longitude_adcp = np.concatenate(nav_elongitude_lst, axis=0)
latitude_adcp = np.concatenate(nav_elatitude_lst, axis=0)
ranges = adcpm['ranges']

# rotate pitch and roll onto float reference frame
xducer_misalign = adcpr['config']['xducer_misalign'][0][0]
EArad = np.deg2rad(-xducer_misalign)
pitch_float = pitch_adcp*np.cos(EArad) - roll_adcp*np.sin(EArad)
roll_float = pitch_adcp*np.cos(EArad) + roll_adcp*np.sin(EArad)

# Q/C ADCP velocities
Nvel[errvel>errvel_THRESH]=np.nan
Evel[errvel>errvel_THRESH]=np.nan

# -------------------------------------------------------------------------------------
# Clean-up output
# -------------------------------------------------------------------------------------
ranges_WG = ranges
# crop times
tt_WG = pd.to_datetime(time_adcp)
keep_bool = tt_WG>=tst
tt_WG = tt_WG[keep_bool]
# sort times
ind = np.argsort(tt_WG)
tt_WG = tt_WG[ind]
# find unique indices
uii = np.unique(tt_WG, return_index=True)[-1]
tt_WG = tt_WG[uii]
# motion-corrected velocities
U_WG = Evel[:,keep_bool]
V_WG = Nvel[:,keep_bool]
U_WG = U_WG[:,ind]
V_WG = V_WG[:,ind]
U_WG = U_WG[:,uii]
V_WG = V_WG[:,uii]
# motion-corrected and filtered velocities
Uf_WG = Evelf[:,keep_bool]
Vf_WG = Nvelf[:,keep_bool]
Uf_WG = Uf_WG[:,ind]
Vf_WG = Vf_WG[:,ind]
Uf_WG = Uf_WG[:,uii]
Vf_WG = Vf_WG[:,uii]
# crop
lon = longitude_adcp[keep_bool]
lat = latitude_adcp[keep_bool]
pitch = pitch_adcp[keep_bool]
roll = roll_adcp[keep_bool]
heading = heading_float[keep_bool]
errvel_WG = errvel[:,keep_bool]
# and sort
lon = lon[ind]
lat = lat[ind]
pitch = pitch[ind]
roll = roll[ind]
heading = heading[ind]
errvel_WG = errvel_WG[:,ind]
# remove repeated values
lon = lon[uii]
lat = lat[uii]
pitch = pitch[uii]
roll = roll[uii]
heading = heading[uii]
errvel_WG = errvel_WG[:,uii]

# # ----------------------------------------------------------------------------------------------------------------------
# # compute averages for continuous records
# # ----------------------------------------------------------------------------------------------------------------------
# iic = np.append(-1,np.where(np.array(np.diff(tt_WG),dtype='timedelta64[m]')>np.array(15,dtype='timedelta64[m]'))[0])
#
# ttbar_WG,Ubar_WG,Vbar_WG = [],[],[]
# for kk, jj in enumerate(iic[:-1]):
#     ii = jj+1
#     ll = iic[kk+1]
#     ttbar_WG.append(pd.to_datetime(tt_WG[ii:ll]).mean())
#     Ubar_WG.append(np.nanmean(U_WG[:,ii:ll],axis=1))
#     Vbar_WG.append(np.nanmean(V_WG[:,ii:ll],axis=1))
# # convert to array
# ttbar_WG = pd.to_datetime(ttbar_WG)
# Ubar_WG = np.array(Ubar_WG).T
# Vbar_WG = np.array(Vbar_WG).T

# ----------------------------------------------------------------------------------------------------------------------
# Block average ADCP velocities and vehicle dynamics
# ----------------------------------------------------------------------------------------------------------------------
import statistics
from geopy.distance import distance

# sampling interval
dts = statistics.mode(np.diff(tt_WG) / np.timedelta64(1, 's'))

# set ensemble time base and interval
DT = np.timedelta64(10, 'm')
ta, tb = tt_WG[0], tt_WG[-1]
ttbar_WG = np.arange(ta, tb + DT, DT)

# Maximum fraction of allowed NaNs per ensemble
nan_adcp_THRESH = 0.6
# minimum number of samples per ensemble
fracN = 0.40
Nmin = int(((DT / np.timedelta64(1, 's')) / dts) * fracN)

lon_bar, lat_bar = [], []
N_adcp, Ubar_WG, Vbar_WG = [], [], []
sog_lonlat, cog_lonlat = [], []
for i, t in enumerate(ttbar_WG):
    # ADCP data
    ii_adcp = np.where(np.logical_and(tt_WG >= t - DT / 2, tt_WG <= t + DT / 2))[0]
    N_adcp.append(len(ii_adcp))
    if len(ii_adcp) > Nmin:
        # ADCP coordinates
        lon_bar.append(np.nanmean(lon[ii_adcp]))
        lat_bar.append(np.nanmean(lat[ii_adcp]))
        # ADCP velocities
        uvel = U_WG[:, ii_adcp].copy()
        vvel = V_WG[:, ii_adcp].copy()
        # TODO: mask turning
        # time-average ADCP velocity profiles
        U = np.nanmean(uvel, axis=1)
        V = np.nanmean(vvel, axis=1)
        # Q/C ADCP velocity profiles
        U_nans = np.sum(np.isnan(uvel), axis=1) / len(ii_adcp) > nan_adcp_THRESH
        V_nans = np.sum(np.isnan(vvel), axis=1) / len(ii_adcp) > nan_adcp_THRESH
        U[U_nans] = np.nan
        V[V_nans] = np.nan
        # store values
        Ubar_WG.append(U)
        Vbar_WG.append(V)
        # Calculate cog and sog from WG coordinates
        p1 = (lat[ii_adcp][0], lon[ii_adcp][0])
        p2 = (lat[ii_adcp][-1], lon[ii_adcp][-1])
        delt = (tt_WG[ii_adcp][-1] - tt_WG[ii_adcp][0]) / np.timedelta64(1, 's')
        sogtmp = distance(p1, p2).m / delt if delt > 0 else np.nan
        sog_lonlat.append(sogtmp)
        cog_lonlat.append(get_bearing(p1, p2))
    else:
        # mask if not enough data is available
        lon_bar.append(np.nan)
        lat_bar.append(np.nan)
        Ubar_WG.append(np.ones_like(ranges) * np.nan)
        Vbar_WG.append(np.ones_like(ranges) * np.nan)
        sog_lonlat.append(np.nan)
        cog_lonlat.append(np.nan)

# 0) convert to array
# ttbar_WG = pd.to_datetime(ttbar_WG)
lon_bar = np.array(lon_bar)
lat_bar = np.array(lat_bar)
sog_lonlat = np.array(sog_lonlat)
cog_lonlat = np.array(cog_lonlat)
Ubar_WG = np.array(Ubar_WG).T
Vbar_WG = np.array(Vbar_WG).T

# 1) mask contaminated bins due to glider sub acoustic interference
sub_depth1 = 8  # m (primary interference)
sub_depth2 = 16 # m (secondary interference)
sub_depth3 = 24 # m (tertiary interference)
jjsub1 = np.abs(ranges-sub_depth1).argmin()
jjsub0 = jjsub1-1
jjsub2 = np.abs(ranges-sub_depth2).argmin()
jjsubx = jjsub2+1
jjsub3 = np.abs(ranges-sub_depth3).argmin()
# Mask velocities for bins at sub depth
Ubar_WG[jjsub0,:] = np.nan
Ubar_WG[jjsub1,:] = np.nan
Ubar_WG[jjsubx,:] = np.nan
Ubar_WG[jjsub2,:] = np.nan
Vbar_WG[jjsub0,:] = np.nan
Vbar_WG[jjsub1,:] = np.nan
Vbar_WG[jjsubx,:] = np.nan
Vbar_WG[jjsub2,:] = np.nan

# 2) Interpolate vertically across nans
n = 2
Ubar_WGi,Vbar_WGi=[],[]
for uvel,vvel in zip(Ubar_WG.T,Vbar_WG.T):
    Ubar_WGi.append(nan_interpolate(uvel,n))
    Vbar_WGi.append(nan_interpolate(vvel,n))
Ubar_WGi = np.array(Ubar_WGi).T
Vbar_WGi = np.array(Vbar_WGi).T
print('Interpolated motion-corrected data')

# # ----------------------------------------------------------------------------------------------------------------------
# # Plot results (filtered velocities)
# # ----------------------------------------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# from matplotlib.dates import DateFormatter
# import matplotlib.gridspec as gridspec
# import numpy as np
#
# # create masked colormap
# cm_msk = plt.get_cmap('seismic').copy()
# cm_msk.set_bad(color='lightgray')
#
# # Plot WG water velocities
# yL = [4,80]
# uL = 0.5
# vL = 0.5
# fntsz = 16
# labsz = 14
# ylab_str = 'depth [m]'
#
# # Do not show vehicle heading
# heights = [1,1]
# nrows = 2
# ncols = 1
# gs_kw = dict(height_ratios=heights)
# fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
#                             sharex=True,
#                             sharey=False,
#                             gridspec_kw=gs_kw,
#                             figsize=(14, 11))
#
#
# # WG: East Vel
# x,y = np.meshgrid(tt_WG,ranges_WG)
# c = ax[0].pcolormesh(x,y,Uf_WG,vmin=-uL, vmax=uL,cmap = cm_msk)
# ax[0].set_ylim(yL)
# ax[0].set_ylabel(ylab_str,fontsize=fntsz)
# ax[0].set_title('WG: East Velocities',fontsize=fntsz)
# ax[0].invert_yaxis()
#
# # WG: North Vel
# x,y = np.meshgrid(tt_WG,ranges_WG)
# ax[1].pcolormesh(x,y,Vf_WG,vmin=-vL, vmax=vL, cmap = cm_msk)
# ax[1].set_ylim(yL)
# ax[1].set_ylabel(ylab_str,fontsize=fntsz)
# ax[1].set_title('WG: North Velocities',fontsize=fntsz)
# ax[1].invert_yaxis()
#
# # rotate and align the tick labels so they look better
# ax[0].tick_params(labelsize=labsz)
# ax[1].tick_params(labelsize=labsz)
# ax[1].set_xlim(tt_WG[0],now)
# # ax[1].set_xlim(tst,now)
#
# # Define the date format
# date_form = DateFormatter("%m-%d-%y %H")
# ax[1].xaxis.set_major_formatter(date_form)
# fig.autofmt_xdate()
#
# # add a colorobar
# gs = gridspec.GridSpec(ncols=3, nrows=nrows, height_ratios=gs_kw['height_ratios'], right=0.95,figure=fig)
# axc = fig.add_subplot(gs[0,-1])
# axc.set_visible(False)
# cbar = fig.colorbar(c, ax=axc,orientation='vertical')
# cax = cbar.ax
# # Add label on top of colorbar.
# cbar.ax.set_xlabel("[m/s]\n",fontsize=labsz)
# cbar.ax.xaxis.set_label_position('top')
# cbar.ax.tick_params(labelsize=labsz)
#
#
# tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
# fig.suptitle(vnam + ': ADCP output ' + tmstmp, fontsize=16)
# fig.show()

# ----------------------------------------------------------------------------------------------------------------------
# Plot results (block-averaged velocities)
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
import numpy as np

# create masked colormap
cm_msk = plt.get_cmap('seismic').copy()
cm_msk.set_bad(color='lightgray')

# Plot WG water velocities
yL = [4,80]
uL = 0.5
vL = 0.5
fntsz = 16
labsz = 14
ylab_str = 'depth [m]'

# Do not show vehicle heading
heights = [1,1,0.5]
nrows = 3
ncols = 1
gs_kw = dict(height_ratios=heights)
fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                            sharex=True,
                            sharey=False,
                            gridspec_kw=gs_kw,
                            figsize=(14, 11))


# WG: East Vel
x,y = np.meshgrid(ttbar_WG,ranges_WG)
c = ax[0].pcolormesh(x,y,Ubar_WGi,vmin=-uL, vmax=uL,cmap = cm_msk)
ax[0].set_ylim(yL)
ax[0].set_ylabel(ylab_str,fontsize=fntsz)
ax[0].set_title('East Velocities',fontsize=fntsz)
ax[0].invert_yaxis()

# WG: North Vel
x,y = np.meshgrid(ttbar_WG,ranges_WG)
ax[1].pcolormesh(x,y,Vbar_WGi,vmin=-vL, vmax=vL, cmap = cm_msk)
ax[1].set_ylim(yL)
ax[1].set_title('North Velocities',fontsize=fntsz)
ax[1].set_ylabel(ylab_str,fontsize=fntsz)
ax[1].invert_yaxis()

# Time gaps
ax[2].plot(tt_WG[:-1],np.diff(tt_WG)/np.timedelta64(1, 'm'))
ax[2].set_ylabel('$\Delta t$ [minutes]',fontsize=fntsz)
ax[2].grid(linestyle=':')

# rotate and align the tick labels so they look better
ax[0].tick_params(labelsize=labsz)
ax[1].tick_params(labelsize=labsz)
ax[2].tick_params(labelsize=labsz)
ax[1].set_xlim(ttbar_WG[0],now)

# Define the date format
date_form = DateFormatter("%m-%d-%y %H")
ax[1].xaxis.set_major_formatter(date_form)
fig.autofmt_xdate()

# add a colorobar
gs = gridspec.GridSpec(ncols=3, nrows=nrows, height_ratios=gs_kw['height_ratios'], right=0.95,figure=fig)
axc = fig.add_subplot(gs[0,-1])
axc.set_visible(False)
cbar = fig.colorbar(c, ax=axc,orientation='vertical')
cax = cbar.ax
# Add label on top of colorbar.
cbar.ax.set_xlabel("[m/s]\n",fontsize=labsz)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=labsz)

# Save figure
tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
fig.suptitle(vnam + ': ADCP output ' + tmstmp, fontsize=16)
fig.show()

# --------------------------------------------------------
# Save figure
# Set local paths and import local packages
from pathlib import Path
loc_folder = os.path.join(str(Path.home()),'src/calcofi/WGautonomy/auto_plots')
figdir = os.path.join(loc_folder,'figz',vnam)
figname = 'adcp_' + vnam + '.png'
fig.savefig(os.path.join(figdir,figname), dpi=100, bbox_inches='tight')

# --------------------------------------------------------
# Save figure
# Set local paths and import local packages
from pathlib import Path
loc_folder = os.path.join(str(Path.home()),'src/calcofi/WGautonomy/auto_plots')
figdir = os.path.join(loc_folder,'figz',vnam)
figname = 'adcp_' + vnam + '.png'
fig.savefig(os.path.join(figdir,figname), dpi=100, bbox_inches='tight')

# --------------------------------------------------------
# Save processed data
from scipy.io import savemat

# convert Python datetime to Matlab datenum
mtime,mtime_bar=[],[]
for mt in tt_WG:
    mtime.append(datetime2matlabdn(mt))
for mtbar in pd.to_datetime(ttbar_WG):
    mtime_bar.append(datetime2matlabdn(mtbar))
mtime = np.array(mtime)
mtime_bar = np.array(mtime_bar)

# create dictionary
mdic = {"time": tt_WG.values.astype(float)/1E9,
        "time_bar": ttbar_WG.astype(float)/1E6,
        "mtime": mtime,
        "mtime_bar": mtime_bar,
        "lon": lon,
        "lat": lat,
        "lon_bar": lon_bar,
        "lat_bar": lat_bar,
        "ranges": ranges,
        "pitch_adcp" : pitch,
        "roll_adcp": roll,
        "heading_float": heading,
        "Evel" : U_WG,
        "Nvel": V_WG,
        "errvel": errvel_WG,
        # "Evelf": Uf_WG,
        # "Nvelf": Vf_WG,
        "Evel_bar": Ubar_WG,
        "Nvel_bar": Vbar_WG,
        'dt_gps': dt_gps,
        # 'dt_avg': dt_avg,
        'sog': sog_lonlat,
        'cog': cog_lonlat,
        'config': adcpr['config']
       }

# Save output as mat file
proc_dir = os.path.join(seachest_data_dir,vnam,'current/nrt/adcp_mc')
fnamout_mat = 'adcp_mc.mat'
savemat(os.path.join(proc_dir,fnamout_mat), mdic)
print('saved matlab ouput to '+os.path.join(proc_dir,fnamout_mat))
# grant read permissions to all users for output file
cmd = 'chmod a+r ' + os.path.join(proc_dir,fnamout_mat)
os.system(cmd)
print('granted read permission to all users for:\n' + os.path.join(proc_dir,fnamout_mat))

# --------------------------------------------------------
# Set local paths and import local packages
calcofi_path = os.path.join(os.path.abspath(os.path.join('..')),'calcofi')
if calcofi_path not in sys.path:
    sys.path.insert(0, calcofi_path)
from WGcode.WGhelpers import sftp_put_cordcdev

# Upload to CORDCdev - need to be on VPN (sio-terrill pool) to access CORDCdev
# TODO: Consider defining these paths in config file (settings.py)
LOCALpath_CD = os.path.join(figdir,figname)
if prj=='norse' or prj=='palau':
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, figname)
else:
    REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, 'waveglider', figname)
SFTPAttr = sftp_put_cordcdev(LOCALpath_CD,REMOTEpath_CD)
print('done uploading ADCP plots to CORDCdev')