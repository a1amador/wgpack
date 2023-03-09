import sys,os
import numpy as np
import pandas as pd
from datetime import datetime
from geopy.distance import distance
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
import pickle
import scipy.io

# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.rdradcp import rdradcp
from wgpack.adcp import motion_correct_ADCP_gps,Doppler_vel_ADCP,rdradcp_output_to_dictionary
from wgpack.config import seachest_data_dir
from wgpack.RDP import rdp
from wgpack.nav import get_bearing
from wgpack.server_helpers import sftp_put_cordcdev,sftp_remove_cordcdev
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
# vnam, channels, tw, prj = 'sv3-125', 'C0158P2JJTT', '5', 'palau'

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
    # use prescribed splash date
    # tst = datetime(2022, 12, 14, 0, 0, 0, 0)

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

# -------------------------------------------------------------------------------------
# Process raw ADPC binaries (daily files)
# -------------------------------------------------------------------------------------
cc=-1
for fnam in f:
    adcp_filepath_in = os.path.join(REMOTEdir, fnam)
    adcp_filepath_out_pkl = os.path.join(REMOTEdir + '_pkl',fnam.replace('PD0','pkl'))
    adcp_filepath_out_mat = os.path.join(REMOTEdir + '_mat', fnam.replace('PD0', 'mat'))
    # file path and name for output .mat file
    fnam_out, file_extension = os.path.splitext(adcp_filepath_in)
    # extract dates
    spl_str = fnam.split('_')
    fdate = datetime.strptime(spl_str[-1][:-4], '%Y%m%d')
    if fdate > tst:
        cc+=1
        print(adcp_filepath_in)
        print(fdate)
        if np.logical_and(not os.path.exists(adcp_filepath_out_pkl), len(spl_str) > 3) or\
            np.logical_and(os.path.getmtime(adcp_filepath_out_pkl) < os.path.getmtime(adcp_filepath_in), len(spl_str) > 3):
            print('updating ' + fnam.replace('PD0', 'pkl'))
            # read-in ADCP binaries
            adcpr = rdradcp(adcp_filepath_in, num_av=1, nens=-1, baseyear=2000, despike='no', log_fp=None,
                            verbose=False)
            # convert to dictionary
            adcpr_dict = rdradcp_output_to_dictionary(adcpr)
            # save dictionary to a file with pickle module
            with open(adcp_filepath_out_pkl, 'wb') as outfile:
                pickle.dump(adcpr_dict, outfile)
            # Save the dictionary to a .mat file
            scipy.io.savemat(adcp_filepath_out_mat, {"adcpr": adcpr_dict})
        elif os.path.exists(adcp_filepath_out_pkl):
            print('loading '+ fnam.replace('PD0','pkl'))
            # Load previously saved ADCP data
            with open(adcp_filepath_out_pkl, 'rb') as infile:
                adcpr_dict = pickle.load(infile)
        # ------------------------------------------------------------
        # Apply motion correction and concatenate daily files
        dt_gps = 120 # Motion correction time interval for GPS-derived velocities (s)
        adcpm_d = motion_correct_ADCP_gps(adcpr_dict, dt_gps, mag_dec=None, qc_flg=False,dtc=None,three_beam_flg=4)
        # concatenate
        if cc==0:
            adcpm = adcpm_d.copy()
        else:
            for key in adcpm.keys():
                print(key,adcpm[key].shape)
                if len(adcpm[key].shape)>1:
                    adcpm[key] = np.concatenate((adcpm[key], adcpm_d[key]), axis=1)
                else:
                    adcpm[key] = np.concatenate((adcpm[key], adcpm_d[key]))
adcpm['ranges'] = np.unique(adcpm['ranges'])
adcpm['time'] = pd.to_datetime(adcpm['time'])

# ----------------------------------------------------------------------
# Find turning points using the RDP algorithm
tolerance = 0.00025
# tolerance = 0.00005
points = tuple(zip(adcpm['longitude'], adcpm['latitude']))
simplified = np.array(rdp(list(points), tolerance))
# find times associated with turning
t_turn = []
for ss in simplified:
    # if both points match, then this is a turning point
    jj = np.sum(ss == points, axis=1) == 2
    t_turn.append(adcpm['time'][jj][0])

# ----------------------------------------------------------------------
# Create time-base for block averaging
# time window
ta = pd.to_datetime(tst)
tb = pd.to_datetime(now)
# time interval for interpolation
DT = np.timedelta64(10,'m')

print('----------------------------------------------------------------------------------------------------------------')
print('creating unix time stamp (tt_unix)')
ta_unix = (pd.to_datetime(ta) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
tb_unix = (pd.to_datetime(tb) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
tt_unix = np.arange(ta_unix,
                    tb_unix,
                    DT.item().total_seconds())
# pandas datetime
tt = pd.to_datetime(tt_unix.astype('datetime64[s]'))

# ----------------------------------------------------------------------
# Block average vehicle dynamics, and ADCP velocities

# Maximum fraction of allowed NaNs per ADCP burst
nan_adcp_THRESH = 0.5

# Minimum number of pings per block average
dt_p2p = np.array(np.median(np.diff(adcpm['time'])), dtype='timedelta64[ns]').item() / 1E9
THRESH_len = (DT / dt_p2p).astype('timedelta64[s]').astype(float) * 0.25

# -------------------------------------------------------------------------------------
# Process: block-averaging
# -------------------------------------------------------------------------------------
lon_adcp, lat_adcp = [], []
sog_lonlat_adcp, cog_lonlat_adcp = [], []
Nvel_adcp, Evel_adcp, UPvel_adcp = [], [], []
Nvel_std_adcp, Evel_std_adcp = [], []
heading_std_adcp, pitch_std_adcp, roll_std_adcp = [], [], []
heading_std_float, pitch_std_float, roll_std_float = [], [], []
heading_mean_float, pitch_mean_float, roll_mean_float = [], [], []
sog_gpse, sog_gpsn = [], []
turn_bool = []
N_adcp = []
for i, t in enumerate(tt):
    # ADCP data
    ii_adcp = np.where(np.logical_and(adcpm['time'] >= t - DT / 2, adcpm['time'] <= t + DT / 2))[0]
    N_adcp.append(len(ii_adcp))
    if len(ii_adcp) > THRESH_len:
        # ADCP coordinates
        lon_adcp.append(adcpm['longitude'][ii_adcp])
        lat_adcp.append(adcpm['latitude'][ii_adcp])
        # motion-corrected (interplated across sub)
        nvel = adcpm['Nvel'][:, ii_adcp].copy()
        evel = adcpm['Evel'][:, ii_adcp].copy()
        # wvel = adcpD['vert_vel'][:, ii_adcp].copy()
        # Calculate cog and sog from WG MWG coordinates
        p1 = (adcpm['latitude'][ii_adcp][0], adcpm['longitude'][ii_adcp][0])
        p2 = (adcpm['latitude'][ii_adcp][-1], adcpm['longitude'][ii_adcp][-1])
        delt = (adcpm['time'][ii_adcp][-1] - adcpm['time'][ii_adcp][0]) / np.timedelta64(1, 's')
        sogtmp = distance(p1, p2).m / delt if delt > 0 else np.nan
        sog_lonlat_adcp.append(sogtmp)
        cog_lonlat_adcp.append(get_bearing(p1, p2))
        sog_gpse.append(np.nanmean(adcpm['sog_gpse'][ii_adcp]))
        sog_gpsn.append(np.nanmean(adcpm['sog_gpsn'][ii_adcp]))
        # Identify turning:
        turn_bool.append(any(item in t_turn for item in adcpm['time'][ii_adcp]))

        # Mask velocity instances with turning (replace with nans)
        if any(item in t_turn for item in adcpm['time'][ii_adcp]):
            print('Found turning')
            # find that turning instance
            ii_turn = np.where([item in t_turn for item in adcpm['time'][ii_adcp]])[0][0]
            # find times near turning point
            t_turn_a = adcpm['time'][ii_adcp][ii_turn] - np.timedelta64(int(dt_gps / 2), 's')
            t_turn_b = adcpm['time'][ii_adcp][ii_turn] + np.timedelta64(int(dt_gps / 2), 's')
            # mask velocities for times around that turning point
            iinan_turn = np.logical_and(adcpm['time'][ii_adcp] > t_turn_a, adcpm['time'][ii_adcp] < t_turn_b)
            nvel[:, iinan_turn] = np.nan
            evel[:, iinan_turn] = np.nan
            # wvel[:, iinan_turn] = np.nan

        # time-average ADCP velocity profiles
        VN = np.nanmean(nvel, 1)
        VE = np.nanmean(evel, 1)
        # VUP = np.nanmean(wvel, 1)
        VN_std = np.nanstd(adcpm['Nvel'][:, ii_adcp], 1)
        VE_std = np.nanstd(adcpm['Evel'][:, ii_adcp], 1)
        # Q/C ADCP velocity profiles
        Nvel_nans = np.sum(np.isnan(adcpm['Nvel'][:, ii_adcp]), axis=1) / len(ii_adcp) > nan_adcp_THRESH
        Evel_nans = np.sum(np.isnan(adcpm['Evel'][:, ii_adcp]), axis=1) / len(ii_adcp) > nan_adcp_THRESH
        VN[Nvel_nans] = np.nan
        VE[Evel_nans] = np.nan
        # store values
        Nvel_adcp.append(VN)
        Evel_adcp.append(VE)
        # UPvel_adcp.append(VUP)
        Nvel_std_adcp.append(VN_std)
        Evel_std_adcp.append(VE_std)

        # ADCP attitude
        heading_std_adcp.append(
            stats.circstd(adcpm['heading_float'][ii_adcp], high=360, low=0, axis=None, nan_policy='omit'))
        # pitch_std_adcp.append(stats.circstd(adcpr.pitch[ii_adcp], high=360, low=0, axis=None, nan_policy='omit'))
        # roll_std_adcp.append(stats.circstd(adcpr.roll[ii_adcp], high=360, low=0, axis=None, nan_policy='omit'))

        # float dynamics
        # Apply circular statistics: mean
        heading_mean_float.append(
            stats.circmean(adcpm['heading_float'][ii_adcp], high=360, low=0, axis=None, nan_policy='omit'))
    #         pitch_mean_float.append(np.nanmean(pitch_float[ii_adcp]))
    #         roll_mean_float.append(np.nanmean(roll_float[ii_adcp]))
    #         # Apply circular statistics: standard deviation
    #         heading_std_float.append(stats.circstd(heading_float[ii_adcp],high=360, low=0, axis=None, nan_policy='omit'))
    #         pitch_std_float.append(stats.circstd(pitch_float[ii_adcp],high=360, low=0, axis=None, nan_policy='omit'))
    #         roll_std_float.append(stats.circstd(roll_float[ii_adcp],high=360, low=0, axis=None, nan_policy='omit'))

    else:
        # mask if not enough data is available
        # ADCP data
        turn_bool.append(True)
        sog_lonlat_adcp.append(np.nan)
        cog_lonlat_adcp.append(np.nan)
        sog_gpse.append(np.nan)
        sog_gpsn.append(np.nan)
        lon_adcp.append(np.nan)
        lat_adcp.append(np.nan)
        Nvel_adcp.append(np.ones_like(adcpm['ranges']) * np.nan)
        Evel_adcp.append(np.ones_like(adcpm['ranges']) * np.nan)
        UPvel_adcp.append(np.ones_like(adcpm['ranges']) * np.nan)
        Nvel_std_adcp.append(np.ones_like(adcpm['ranges']) * np.nan)
        Evel_std_adcp.append(np.ones_like(adcpm['ranges']) * np.nan)
        # ADCP attitude
        heading_std_adcp.append(np.nan)
        # pitch_std_adcp.append(np.nan)
        # roll_std_adcp.append(np.nan)
        # # vehicle motion statistics
        # heading_mean_float.append(np.nan)
        # pitch_mean_float.append(np.nan)
        # roll_mean_float.append(np.nan)
        # heading_std_float.append(np.nan)
        # pitch_std_float.append(np.nan)
        # roll_std_float.append(np.nan)

# No. of measurements
N_adcp = np.array(N_adcp)
# turning
turn_bool = np.array(turn_bool)
# vehicle sog and cog
sog_lonlat_adcp = np.array(sog_lonlat_adcp)
cog_lonlat_adcp = np.array(cog_lonlat_adcp)
# vehicle north and east velocities
sog_gpse = np.array(sog_gpse)
sog_gpsn = np.array(sog_gpsn)
# vehicle motion statistics (mean)
heading_mean_float = np.array(heading_mean_float)
pitch_mean_float = np.array(pitch_mean_float)
roll_mean_float = np.array(roll_mean_float)
# vehicle motion statistics (standard deviation)
heading_std_float = np.array(heading_std_float)
pitch_std_float = np.array(pitch_std_float)
roll_std_float = np.array(roll_std_float)
# ADCP attitude
heading_std_adcp = np.array(heading_std_adcp)
pitch_std_adcp = np.array(pitch_std_adcp)
roll_std_adcp = np.array(roll_std_adcp)
# ADCP velocities
Nvel_adcp = np.array(Nvel_adcp).T
Evel_adcp = np.array(Evel_adcp).T
UPvel_adcp = np.array(UPvel_adcp).T
Nvel_std_adcp = np.array(Nvel_std_adcp).T
Evel_std_adcp = np.array(Evel_std_adcp).T
# ADCP coordinates
lon_adcp = np.array(lon_adcp)
lat_adcp = np.array(lat_adcp)
lon_bar, lat_bar = [], []
for lon, lat in zip(lon_adcp, lat_adcp):
    lon_bar.append(np.nanmean(lon))
    lat_bar.append(np.nanmean(lat))

# mask contaminated bins due to glider sub acoustic interference
sub_depth1 = 8  # m (primary interference)
sub_depth2 = 16  # m (secondary interference)
sub_depth3 = 24  # m (tertiary interference)
jjsub1 = np.abs(adcpm['ranges'] - sub_depth1).argmin()
jjsub2 = np.abs(adcpm['ranges'] - sub_depth2).argmin()
jjsub3 = np.abs(adcpm['ranges'] - sub_depth3).argmin()
jjsub0 = jjsub1-1
jjsubx = jjsub2+1
# Evel
Evel_adcp[jjsub1,:]=np.nan
Evel_adcp[jjsub2,:]=np.nan
Evel_adcp[jjsub0,:]=np.nan
Evel_adcp[jjsubx,:]=np.nan
# Nvel
Nvel_adcp[jjsub1,:]=np.nan
Nvel_adcp[jjsub2,:]=np.nan
Nvel_adcp[jjsub0,:]=np.nan
Nvel_adcp[jjsubx,:]=np.nan


# 2) Interpolate vertically across nans
Uf_WG = Evel_adcp.copy()
Vf_WG = Nvel_adcp.copy()
n = 2
Ubar_WGi,Vbar_WGi=[],[]
for uvel,vvel in zip(Uf_WG.T,Vf_WG.T):
    Ubar_WGi.append(nan_interpolate(uvel,n))
    Vbar_WGi.append(nan_interpolate(vvel,n))
Ubar_WGi = np.array(Ubar_WGi).T
Vbar_WGi = np.array(Vbar_WGi).T
print('Interpolated motion-corrected data')

# ----------------------------------------------------------------------------------------------------------------------
# Plot results
# ----------------------------------------------------------------------------------------------------------------------
# create masked colormap
cm_msk = plt.get_cmap('bwr').copy()
cm_msk.set_bad(color='lightgray')
# Plot WG water velocities
yL = [4,80]
uL = 0.5
vL = 0.5
fntsz = 14
labsz = 12
ylab_str = 'depth [m]'
heights = [1,1]
nrows = 2
ncols = 1
gs_kw = dict(height_ratios=heights)
fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                            sharex=True,
                            sharey=False,
                            gridspec_kw=gs_kw,
                            figsize=(12, 8))

# WG: East Vel
x,y = np.meshgrid(tt,adcpm['ranges'])
c = ax[0].pcolormesh(x,y,Ubar_WGi,vmin=-uL, vmax=uL,cmap = cm_msk)
ax[0].set_ylim(yL)
ax[0].set_ylabel(ylab_str,fontsize=fntsz)
ax[0].set_title('East Velocities',fontsize=fntsz)
ax[0].invert_yaxis()

# WG: North Vel
x,y = np.meshgrid(tt,adcpm['ranges'])
ax[1].pcolormesh(x,y,Vbar_WGi,vmin=-vL, vmax=vL, cmap = cm_msk)
ax[1].set_ylim(yL)
ax[1].set_ylabel(ylab_str,fontsize=fntsz)
ax[1].set_title('North Velocities',fontsize=fntsz)
ax[1].invert_yaxis()


# rotate and align the tick labels so they look better
ax[0].tick_params(labelsize=labsz)
ax[1].tick_params(labelsize=labsz)

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

# figure title
tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
fig.suptitle(vnam + ': ADCP output ' + tmstmp, fontsize=16)
fig.show()

# ----------------------------------------------------------------------------------------------------------------------
# Save figure
# ----------------------------------------------------------------------------------------------------------------------
# Set local paths and import local packages
from pathlib import Path
loc_folder = os.path.join(str(Path.home()),'src/calcofi/WGautonomy/auto_plots')
# figdir = os.path.join(loc_folder,'figz',vnam)
figdir = os.path.join(seachest_data_dir,vnam,'autoplots')
figname = 'adcp_' + vnam + '.jpg'
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
print('done uploading adcp plots to CORDCdev')



# import matplotlib.pyplot as plt
# plt.plot(adcpm['longitude'],adcpm['latitude'])
# for lonlat in simplified:
#     plt.plot(lonlat[0], lonlat[1],'or')
# plt.show()