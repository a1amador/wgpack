import os, sys
import numpy as np
import pandas as pd
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

# ----------------------------------------------------------------------------------------------------------------------
# Set local paths and import local packages
# ----------------------------------------------------------------------------------------------------------------------
calcofi_path = os.path.join(os.path.abspath(os.path.join('..')),'calcofi')
if calcofi_path not in sys.path:
    sys.path.insert(0, calcofi_path)
from WGcode.WGhelpers import sftp_put_cordcdev,sftp_remove_cordcdev
# from WGcode.WGhelpers import veh_list, sftp_put_cordcdev,sftp_remove_cordcdev
# from WGcode.slackhelpers import send_slack_file

# Data service path
ds_folder = os.path.join(str(Path.home()),'src/lri-wgms-data-service')
if ds_folder not in sys.path:
    sys.path.insert(0, ds_folder)
from DataService import DataService

# knots to m/s
kt2mps = 0.514444

# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------
args = sys.argv
vnam    = args[1]           # vehicle name
vid     = veh_list[vnam]    # vehicle id (Data Portal)
channels= args[2]           # Slack channel
tw      = args[3]           # time window to display specified in days (ends at present time)
tw = None if tw == 'None' else tw
prj     = args[4]           # project folder name (e.g., 'calcofi', 'tfo', etc.)
# vnam,channels,tw,prj =  'sv3-253','C0158P2JJTT','None','westpac'
# tw = None if tw == 'None' else tw
# vid = veh_list[vnam]

print('vehicle:',vnam)
print('project:',prj)

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

# ----------------------------------------------------------------------------------------------------------------------
# Read in CTD output from Data Portal
# ----------------------------------------------------------------------------------------------------------------------
try:
    ctdDPdf = readDP_CTD(vid=vid, start_date=tst.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    flg_plt = True
    # Convert oxygenHz to dissolved oxygen
    # Measured values
    F = ctdDPdf['oxygenHz'].values
    T = ctdDPdf['temperature'].values
    P = ctdDPdf['pressure'].values
    S = ctdDPdf['salinity'].values
    dFdt = np.gradient(F, ctdDPdf['time'].values)  # Time derivative of SBE 43 output oxygen signal (volts/second)
    OxSol = ctdDPdf['oxygenSolubility'].values  # Oxygen saturation value after Garcia and Gordon (1992)
    DOdict = OxHz2DO(F, dFdt, T, P, S, OxSol, vnam)
    # Store in dataframe
    ctdDPdf['DO (ml/L)'] = DOdict['DOmlL']
    ctdDPdf['DO (muM/kg)'] = DOdict['DOmuMkg']
except RuntimeError as err:
    print(err)
    print("Something went wrong when retrieving CTD data from LRI's Data Portal")
    flg_plt = False


# ----------------------------------------------------------------------------------------------------------------------
# Read in vehicle location from Data Service
# ----------------------------------------------------------------------------------------------------------------------
# instantiate data-service object
ds = DataService()
# To get report names
# print(ds.report_list)

start_date = tst.strftime("%Y-%m-%dT%H:%M:%S.000Z")
end_date = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
out = ds.get_report_data('Telemetry 6 Report', start_date, end_date, [vnam])
# Convert to pandas dataframe
Telemdf = pd.json_normalize(out['report_data'][0]['vehicle_data']['recordData'])
# set timeStamp column as datetimeindex
Telemdf = Telemdf.set_index(pd.DatetimeIndex(Telemdf['gliderTimeStamp'].values))
Telemdf.drop(columns=['gliderTimeStamp'], inplace=True)
# sort index
Telemdf.sort_index(inplace=True)

# ----------------------------------------------------------------------------------------------------------------------
# Compute vehicle speed and direction (sog, cog)
# ----------------------------------------------------------------------------------------------------------------------
sog_lonlat,cog_lonlat=[],[]
cc=0
for index, row in Telemdf[:-1].iterrows():
    cc+=1
    p1 = (row['latitude'],row['longitude'])
    p2 = (Telemdf.iloc[cc]['latitude'],Telemdf.iloc[cc]['longitude'])
    delt = (Telemdf.index[cc]-index)/np.timedelta64(1, 's')
    sogtmp = distance(p1,p2).m/delt if delt>0 else np.nan
    sog_lonlat.append(sogtmp)
    cog_lonlat.append(get_bearing(p1,p2))

# not sure if these are trustworthy
speedOverGround = Telemdf['speedOverGround'].values*kt2mps
gliderSpeed = Telemdf['gliderSpeed'].values*kt2mps

# ----------------------------------------------------------------------------------------------------------------------
# Interpolate
# ----------------------------------------------------------------------------------------------------------------------
from scipy.interpolate import interp1d
# time base
tt_ctd = ctdDPdf['time'].values
tt_tel = Telemdf.index.astype(np.int64)/1E9
# Interpolation functions (CTD)
fi_T = interp1d(tt_ctd, ctdDPdf['temperature'].values, fill_value="extrapolate")
fi_S = interp1d(tt_ctd, ctdDPdf['salinity'].values, fill_value="extrapolate")
# interpolate CTD data
T = fi_T(tt_tel)
S = fi_S(tt_tel)
# mask extrapolated values
iiamsk = tt_tel>tt_ctd[0]
iibmsk = tt_tel<tt_ctd[-1]
T[~iiamsk] = np.nan
T[~iibmsk] = np.nan
S[~iiamsk] = np.nan
S[~iibmsk] = np.nan

# ----------------------------------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------------------------------
if flg_plt:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature, COLORS
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # configure general plot parameters
    labsz   = 12
    fntsz   = 14
    sz      = 30
    tL      = [tst, now]
    widths  = [1, 1]
    heights = [1, 0.5, 0.5, 0.5]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)

    # --------------------------------------------------------------------------------------------
    # Map
    # --------------------------------------------------------------------------------------------
    # configure general cartopy parameters
    LAND = NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',
                               facecolor=COLORS['land'])
    fig, axd = plt.subplot_mosaic([['ax1', 'ax2'], ['ax3', 'ax4'], ['ax5', 'ax6'], ['ax7', 'ax8']], gridspec_kw=gs_kw,
                                  figsize=(12, 12), subplot_kw=dict(projection=ccrs.PlateCarree()))

    # Set map limits
    dx = 2
    dy = 2
    lonmin = Telemdf['longitude'][-1] - dx / 2
    lonmax = Telemdf['longitude'][-1] + dx / 2
    latmin = Telemdf['latitude'][-1] - dy / 2
    latmax = Telemdf['latitude'][-1] + dy / 2
    axd['ax1'].set_extent([lonmin, lonmax, latmin, latmax])
    axd['ax2'].set_extent([lonmin, lonmax, latmin, latmax])

    # --------------------------------------------------------------------------------------------
    # Temperature
    # --------------------------------------------------------------------------------------------
    # configure labels and grid lines
    gl = axd['ax1'].gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    tistr = 'Temperature'
    # add land and coastlines
    axd['ax1'].add_feature(LAND)
    axd['ax1'].coastlines(resolution='10m')

    # plot Wave Glider location
    cf = axd['ax1'].scatter(Telemdf['longitude'].values, Telemdf['latitude'].values,
                            s=sz, c=T, cmap=pjet_cmap)
    axd['ax1'].scatter(Telemdf['longitude'].values[-1], Telemdf['latitude'].values[-1],
                       s=sz*4, c=T[-1], marker='d', cmap=pjet_cmap, vmin=T.min(), vmax=T.max(), edgecolors='k')

    # tick params
    axd['ax1'].tick_params(labelsize=fntsz)
    # colorbar and labels
    cbtstr = 'deg C\n'
    cb = fig.colorbar(cf, ax=axd['ax1'], shrink=0.95)
    cb.ax.set_title(cbtstr)
    axd['ax1'].set_title(tistr)

    # --------------------------------------------------------------------------------------------
    # Salinity
    # --------------------------------------------------------------------------------------------
    gl = axd['ax2'].gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    tistr = 'Salinity'
    # add land and coastlines
    axd['ax2'].add_feature(LAND)
    axd['ax2'].coastlines(resolution='10m')

    # plot Wave Glider location
    cf = axd['ax2'].scatter(Telemdf['longitude'].values, Telemdf['latitude'].values,
                            s=sz, c=S)
    axd['ax2'].scatter(Telemdf['longitude'].values[-1], Telemdf['latitude'].values[-1],
                       s=sz*4, c=S[-1], marker='d', vmin=S.min(), vmax=S.max(), edgecolors='k')

    # tick params
    axd['ax2'].tick_params(labelsize=fntsz)
    # colorbar and labels
    cbtstr = 'psu\n'
    cb = fig.colorbar(cf, ax=axd['ax2'], shrink=0.95)
    cb.ax.set_title(cbtstr)
    axd['ax2'].set_title(tistr)

    # --------------------------------------------------------------------------------------------
    # Add timeseries plots
    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------
    # Temperature
    # --------------------------------------------------------------------------------------------
    fig.delaxes(axd['ax3'])
    fig.delaxes(axd['ax4'])
    gs = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(ctdDPdf['temperature'], '-', color='b', label='WG')
    ax3.set_ylabel('Temperature [deg C]', fontsize=fntsz)
    ax3.grid(':')
    ax3.set_xlim(tL[0], tL[-1])

    # --------------------------------------------------------------------------------------------
    # Salinity
    # --------------------------------------------------------------------------------------------
    fig.delaxes(axd['ax5'])
    fig.delaxes(axd['ax6'])
    gs = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)

    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(ctdDPdf['salinity'], '-', color='b', label='WG')
    ax4.set_ylabel('Salinity [psu]', fontsize=fntsz)
    ax4.grid(':')
    ax4.set_xlim(tL[0], tL[-1])

    # --------------------------------------------------------------------------------------------
    # SOG
    # --------------------------------------------------------------------------------------------
    fig.delaxes(axd['ax7'])
    fig.delaxes(axd['ax8'])
    gs = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)

    ax5 = fig.add_subplot(gs[3, :])
    # ax5.plot(Telemdf['oceanCurrent']*kt2mps,'-r',label='STW')
    ax5.plot(Telemdf.index[:-1], sog_lonlat, '-k', label='SOG')
    ax5.set_ylabel('SOG [m s$^{-1}$]', fontsize=fntsz)
    ax5.grid(':')
    # ax5.legend()
    ax5.set_xlim(tL[0], tL[-1])

    # --------------------------------------------------------------------------------------------
    # sup title
    # --------------------------------------------------------------------------------------------
    tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
    fig.suptitle(vnam + ': CTD output ' + tmstmp, fontsize=16)

    # --------------------------------------------------------
    # Save figure
    # Set local paths and import local packages
    from pathlib import Path

    loc_folder = os.path.join(str(Path.home()), 'src/calcofi/WGautonomy/auto_plots')
    figdir = os.path.join(loc_folder, 'figz', vnam)
    figname = 'CTD_' + vnam + '.png'
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