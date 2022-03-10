import sys,os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import netCDF4 as netcdf
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from geopy.distance import distance


# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.nav import get_bearing
from wgpack.config import seachest_data_dir
datadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/forecasts')

# Data service path
ds_folder = os.path.join(str(Path.home()),'src/lri-wgms-data-service')
if ds_folder not in sys.path:
    sys.path.insert(0, ds_folder)
from DataService import DataService

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
# vnam,channels,tw,prj =  'sv3-251','C0158P2JJTT','4','westpac'

print('vehicle:',vnam)
print('project:',prj)
# ----------------------------------------------------------------------------------------------------------------------
# Set start date according to the time window to be considered
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
# Read in vehicle location from Data Service
# try:
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
kt2mps = 0.514444
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
# LOAD ROMS data
try:
    # filename
    filename = 'ROMS_Guam_' + datetime.utcnow().strftime('%Y%m%d') + '.ncd'
    dfnam    = os.path.join(datadir,filename)
    init_str = datetime.utcnow().strftime('%Y-%m-%d') + ' UTC'
    # Read-in data
    ROMS_data = netcdf.Dataset(dfnam)
except FileNotFoundError as err:
    print(err)
    # filename (previous day)
    filename = 'ROMS_Guam_' + (datetime.utcnow()- pd.Timedelta(days=1)).strftime('%Y%m%d') + '.ncd'
    dfnam    = os.path.join(datadir,filename)
    init_str = (datetime.utcnow()- pd.Timedelta(days=1)).strftime('%Y-%m-%d') + ' UTC'
    # Read-in data
    ROMS_data = netcdf.Dataset(dfnam)

# get surface layer velocities near the Wave Glider
# depth
sub_depth = 8
iida = ROMS_data['depth'][:]<(sub_depth+2)
# find nearest ROMS point to WG location
iilonWG = np.abs(ROMS_data['lon'][:]-Telemdf['longitude'].values[-1]).argmin()
iilatWG = np.abs(ROMS_data['lat'][:]-Telemdf['latitude'].values[-1]).argmin()
# Grid points to show
nii = 20 # number of grid points
iiL = min(len(ROMS_data['lon'][:]),len(ROMS_data['lat'][:]))
iia_lon = max(iilonWG-nii,0)
iib_lon = min(iilonWG+nii,iiL)
iia_lat = max(iilatWG-nii,0)
iib_lat = min(iilatWG+nii,iiL)
# find nearest time interval
print(ROMS_data['time'].units)
match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', ROMS_data['time'].units)
basetime = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S').date())
ttsim = []
for hh in ROMS_data['time'][:]:
    ttsim.append(basetime+pd.Timedelta(hours=hh))
ttsim = pd.to_datetime(ttsim)
iit = np.abs(ttsim-Telemdf.index[-1]).argmin()

# get data
u_sl = np.mean(ROMS_data['u'][iit, iida, iia_lat:iib_lat, iia_lon:iib_lon],axis=0)
v_sl = np.mean(ROMS_data['v'][iit, iida, iia_lat:iib_lat, iia_lon:iib_lon],axis=0)
lon_ROMS = ROMS_data['lon'][iia_lon:iib_lon]
lat_ROMS = ROMS_data['lat'][iia_lat:iib_lat]
# compute velocity magnitude
vel_mag = np.sqrt(u_sl**2+v_sl**2)

# ----------------------------------------------------------------------------------------------------------------------
# LOAD WRF data
try:
    # filename
    filename = 'WRF_Guam_' + datetime.utcnow().strftime('%Y%m%d') + '.ncd'
    dfnam    = os.path.join(datadir,filename)
    init_str = datetime.utcnow().strftime('%Y-%m-%d') + ' UTC'
    # Read-in data
    WRF_data = netcdf.Dataset(dfnam)
except FileNotFoundError as err:
    print(err)
    # filename (previous day)
    filename = 'WRF_Guam_' + (datetime.utcnow()- pd.Timedelta(days=1)).strftime('%Y%m%d') + '.ncd'
    dfnam    = os.path.join(datadir,filename)
    init_str = (datetime.utcnow()- pd.Timedelta(days=1)).strftime('%Y-%m-%d') + ' UTC'
    # Read-in data
    WRF_data = netcdf.Dataset(dfnam)

# get 10-m wind velocities near the Wave Glider
# find nearest WRF point to WG location
iilonWG = np.abs(WRF_data['lon'][:]-Telemdf['longitude'].values[-1]).argmin()
iilatWG = np.abs(WRF_data['lat'][:]-Telemdf['latitude'].values[-1]).argmin()
# Grid points to show
nii = 20 # number of grid points
iiL = min(len(WRF_data['lon'][:]),len(WRF_data['lat'][:]))
iia_lon = max(iilonWG-nii,0)
iib_lon = min(iilonWG+nii,iiL)
iia_lat = max(iilatWG-nii,0)
iib_lat = min(iilatWG+nii,iiL)
# find nearest time interval
print(WRF_data['time'].units)
match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', WRF_data['time'].units)
basetime = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S').date())
ttsim = []
for hh in WRF_data['time'][:]:
    ttsim.append(basetime+pd.Timedelta(hours=hh))
ttsim = pd.to_datetime(ttsim)
iit = np.abs(ttsim-Telemdf.index[-1]).argmin()

# get data
u10 = WRF_data['Uwind'][iit, iia_lat:iib_lat, iia_lon:iib_lon]
v10 = WRF_data['Vwind'][iit, iia_lat:iib_lat, iia_lon:iib_lon]
lon_WRF = WRF_data['lon'][iia_lon:iib_lon]
lat_WRF = WRF_data['lat'][iia_lat:iib_lat]
# # compute velocity magnitude
u10_mag = np.sqrt(u10**2+v10**2)

# ----------------------------------------------------------------------------------------------------------------------
# LOAD WW3 data
try:
    # filename
    filename = 'WaveWatch_III_Mariana_' + datetime.utcnow().strftime('%Y%m%d') + '.ncd'
    dfnam    = os.path.join(datadir,filename)
    init_str = datetime.utcnow().strftime('%Y-%m-%d') + ' UTC'
    # Read-in data
    WW3_data = netcdf.Dataset(dfnam)
except FileNotFoundError as err:
    print(err)
    # filename (previous day)
    filename = 'WaveWatch_III_Mariana_' + (datetime.utcnow()- pd.Timedelta(days=1)).strftime('%Y%m%d') + '.ncd'
    dfnam    = os.path.join(datadir,filename)
    init_str = (datetime.utcnow()- pd.Timedelta(days=1)).strftime('%Y-%m-%d') + ' UTC'
    # Read-in data
    WW3_data = netcdf.Dataset(dfnam)

# get bulk wave params near the Wave Glider
# find nearest WW3 point to WG location
iilonWG = np.abs(WW3_data['lon'][:]-Telemdf['longitude'].values[-1]).argmin()
iilatWG = np.abs(WW3_data['lat'][:]-Telemdf['latitude'].values[-1]).argmin()
# Grid points to show
nii = 9 # number of grid points
iiL = min(len(WW3_data['lon'][:]),len(WW3_data['lat'][:]))
iia_lon = max(iilonWG-nii,0)
iib_lon = min(iilonWG+nii,iiL)
iia_lat = max(iilatWG-nii,0)
iib_lat = min(iilatWG+nii,iiL)
# find nearest time interval
print(WW3_data['time'].units)
match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', WW3_data['time'].units)
basetime = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S').date())
ttsim = []
for hh in WW3_data['time'][:]:
    ttsim.append(basetime+pd.Timedelta(hours=hh))
ttsim = pd.to_datetime(ttsim)
iit = np.abs(ttsim-Telemdf.index[-1]).argmin()

# get data
# peak wave direction (time, z, lat, lon)
Dp = WW3_data['Tdir'][iit,0,iia_lat:iib_lat,iia_lon:iib_lon]
# compute wave direction components (based on Dp)
uw = -np.sin(Dp*np.pi/180)
vw = -np.cos(Dp*np.pi/180)
# significant wave height (time, z, lat, lon)
Hs = WW3_data['Thgt'][iit,0,iia_lat:iib_lat,iia_lon:iib_lon]
# peak wave period (time, z, lat, lon)
Tp = WW3_data['Tper'][iit,0,iia_lat:iib_lat,iia_lon:iib_lon]
# coordinates
lon_WW3 = WW3_data['lon'][iia_lon:iib_lon]
lat_WW3 = WW3_data['lat'][iia_lat:iib_lat]


# ----------------------------------------------------------------------------------------------------------------------
# Plot results
# ----------------------------------------------------------------------------------------------------------------------
# Bringing it all together
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, COLORS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# configure general plot parameters
labsz = 12
fntsz = 14
widths = [1, 1]
heights = [1, 1, 0.5]
gs_kw = dict(width_ratios=widths, height_ratios=heights)

# configure general cartopy parameters
LAND = NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',
                           facecolor=COLORS['land'])

fig, axd = plt.subplot_mosaic([['ax1', 'ax2'], ['ax3', 'ax4'], ['ax5', 'ax5']], gridspec_kw=gs_kw,
                              figsize=(14, 15), subplot_kw=dict(projection=ccrs.PlateCarree()))
# fig, ax = plt.subplots(3, 2, gridspec_kw=gs_kw, figsize=(15, 15))

# --------------------------------------------------------------------------------------------
# Hs
# --------------------------------------------------------------------------------------------
# config for Hs
isub = 1
levels = np.linspace(0,5,101)

axd['ax1'].set_extent([lon_WW3.min(), lon_WW3.max(), lat_WW3.min(), lat_WW3.max()])
axd['ax1'].add_feature(LAND)
axd['ax1'].coastlines(resolution='10m')
gl = axd['ax1'].gridlines(draw_labels=True)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# contour plot
cf = axd['ax1'].contourf(lon_WW3, lat_WW3, Hs, levels=levels, cmap='jet',
                 transform=ccrs.PlateCarree())

# plot Wave Glider location
axd['ax1'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
axd['ax1'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
axd['ax1'].legend(fontsize=fntsz,loc='upper left')
# tick params
axd['ax1'].tick_params(labelsize=fntsz)

# Quiver plot
legend_vel=1.0
Q = axd['ax1'].quiver(lon_WW3, lat_WW3, uw, vw, pivot='middle')

# colorbar and labels
cb = fig.colorbar(cf, ax=axd['ax1'], shrink=0.95,ticks=np.linspace(levels[0],levels[-1],11))
cb.ax.set_title('m')
axd['ax1'].set_title('WW3 significant wave height (Hs)');

# --------------------------------------------------------------------------------------------
# Tp
# --------------------------------------------------------------------------------------------
# config for Tp
isub = 1
levels = np.linspace(2,18,101)

axd['ax2'].set_extent([lon_WW3.min(), lon_WW3.max(), lat_WW3.min(), lat_WW3.max()])
axd['ax2'].add_feature(LAND)
axd['ax2'].coastlines(resolution='10m')
gl = axd['ax2'].gridlines(draw_labels=True)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# contour plot
cf = axd['ax2'].contourf(lon_WW3, lat_WW3, Tp, levels=levels, cmap='jet',
                 transform=ccrs.PlateCarree())

# plot Wave Glider location
axd['ax2'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
axd['ax2'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
axd['ax2'].legend(fontsize=fntsz,loc='upper left')
# tick params
axd['ax2'].tick_params(labelsize=fntsz)

# Quiver plot
legend_vel=1.0
Q = axd['ax2'].quiver(lon_WW3, lat_WW3, uw, vw, pivot='middle')

# colorbar and labels
cb = fig.colorbar(cf, ax=axd['ax2'], shrink=0.95,ticks=np.linspace(levels[0],levels[-1],17))
cb.ax.set_title('s')
axd['ax2'].set_title('WW3 peak period (Tp)');

# --------------------------------------------------------------------------------------------
# Wind speed
# --------------------------------------------------------------------------------------------
# config for u10
isub = 3
levels = np.linspace(0,10,101)

axd['ax3'].set_extent([lon_WRF.min(), lon_WRF.max(), lat_WRF.min(), lat_WRF.max()])
axd['ax3'].add_feature(LAND)
axd['ax3'].coastlines(resolution='10m')
gl = axd['ax3'].gridlines(draw_labels=True)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# contour plot
cf = axd['ax3'].contourf(lon_WRF, lat_WRF, u10_mag, levels=levels, cmap='jet',
                 transform=ccrs.PlateCarree())

# plot Wave Glider location
axd['ax3'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
axd['ax3'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
axd['ax3'].legend(fontsize=fntsz,loc='upper left')
# tick params
axd['ax3'].tick_params(labelsize=fntsz)

# Quiver plot
legend_vel=1.0
Q = axd['ax3'].quiver(lon_WRF[::isub], lat_WRF[::isub], u10[::isub,::isub], v10[::isub,::isub], pivot='middle')

# colorbar and labels
cb = fig.colorbar(cf, ax=axd['ax3'], shrink=0.95,ticks=np.linspace(levels[0],levels[-1],11))
cb.ax.set_title('m s$^{-1}$')
axd['ax3'].set_title('WRF 10-m winds');

# --------------------------------------------------------------------------------------------
# Currents
# --------------------------------------------------------------------------------------------
# config for u10
isub = 2
levels = np.linspace(0,0.5,101)

axd['ax4'].set_extent([lon_ROMS.min(), lon_ROMS.max(), lat_ROMS.min(), lat_ROMS.max()])
axd['ax4'].add_feature(LAND)
axd['ax4'].coastlines(resolution='10m')
gl = axd['ax4'].gridlines(draw_labels=True)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# contour plot
cf = axd['ax4'].contourf(lon_ROMS, lat_ROMS, vel_mag, levels=levels, cmap='jet',
                 transform=ccrs.PlateCarree())

# plot Wave Glider location
axd['ax4'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
axd['ax4'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
axd['ax4'].legend(fontsize=fntsz,loc='upper left')
# tick params
axd['ax4'].tick_params(labelsize=fntsz)

# Quiver plot
legend_vel=1.0
Q = axd['ax4'].quiver(lon_ROMS[::isub], lat_ROMS[::isub], u_sl[::isub,::isub], v_sl[::isub,::isub], pivot='middle')

# colorbar and labels
cb = fig.colorbar(cf, ax=axd['ax4'], shrink=0.95,ticks=np.linspace(levels[0],levels[-1],11))
cb.ax.set_title('m s$^{-1}$')
axd['ax4'].set_title('ROMS sea surface currents');

# --------------------------------------------------------------------------------------------
# Vehicle SOG
# --------------------------------------------------------------------------------------------
fig.delaxes(axd['ax5'])
gs = fig.add_gridspec(ncols=2, nrows = 3,width_ratios=widths, height_ratios=heights)
ax = fig.add_subplot(gs[-1, :])
ax.plot(Telemdf.index[:-1], sog_lonlat,'-k',label='observed')
ax.set_ylabel('SOG [m s$^{-1}$]',fontsize=fntsz)
ax.legend(fontsize=labsz)
ax.set_xlim(Telemdf.index[0],Telemdf.index[-1])
ax.grid(':')


# Add a big title at the top
tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
fig.suptitle(vnam + ': PacIOOS forecasts for ' + tmstmp + '\nInitialized ' +init_str, fontsize=fntsz+2, y=0.945)

# --------------------------------------------------------------------------------------------
# Save figure
# --------------------------------------------------------------------------------------------
# Set local paths and import local packages
from pathlib import Path
loc_folder = os.path.join(str(Path.home()),'src/calcofi/WGautonomy/auto_plots')
figdir = os.path.join(loc_folder,'figz',vnam)
figname = 'forecasts_' + vnam + '.png'
fig.savefig(os.path.join(figdir,figname), dpi=100, bbox_inches='tight')


# --------------------------------------------------------------------------------------------
# Upload to CORDCdev
# --------------------------------------------------------------------------------------------

# Set local paths and import local packages
calcofi_path = os.path.join(os.path.abspath(os.path.join('..')),'calcofi')
if calcofi_path not in sys.path:
    sys.path.insert(0, calcofi_path)
from WGcode.WGhelpers import sftp_put_cordcdev

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
print('done uploading PacIOOS forecast plots to CORDCdev')