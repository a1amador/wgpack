import sys,os
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from pathlib import Path
import netCDF4 as netcdf
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from geopy.distance import distance
import pyproj


module_path = os.path.abspath(os.path.join('..'))
# module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
# module_path = '/Users/a1amador/src'
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.nav import get_bearing
from wgpack.config import seachest_data_dir
from wgpack.timeconv import epoch2datetime64
datadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/forecasts/PacIOOS')

# # Data service path
# ds_folder = os.path.join(str(Path.home()),'src/lri-wgms-data-service')
# if ds_folder not in sys.path:
#     sys.path.insert(0, ds_folder)
# from DataService import DataService

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
# Load ROMS data
# ----------------------------------------------------------------------------------------------------------------------
hh = np.arange(0,24*14,3)
tst = datetime(2022, 4, 1, 0, 0, 0, 0)
for h in hh:
    # set date
    tt_date = tst + timedelta(hours=int(h))
    dates_str = tt_date.strftime('%Y%m%d')
    print(tt_date.strftime('%Y-%m-%d-%H'))
    # Load ROMS data
    try:
        # filename
        filename = 'ROMS_Guam_' + dates_str + '.ncd'
        dfnam    = os.path.join(datadir,filename)
        init_str = tt_date.strftime('%Y-%m-%d') + ' UTC'
        # Read-in data
        ROMS_data = netcdf.Dataset(dfnam)
    except FileNotFoundError as err:
        print(err)

    # get surface layer velocities near the Wave Glider
    # depth
    sub_depth = 8
    iida = ROMS_data['depth'][:] < (sub_depth + 2)

    # find nearest ROMS point to ROTA
    lonRot = 145.2041
    latRot = 14.1538
    # Set map limits
    dx = 0.95
    dy = 0.95
    lonmin = lonRot - dx / 2
    lonmax = lonRot + dx / 2
    latmin = latRot - dy / 2
    latmax = latRot + dy / 2
    # Grid points to show
    iia_lon = np.argmin(np.abs(ROMS_data['lon'][:] - lonmin))
    iib_lon = np.argmin(np.abs(ROMS_data['lon'][:] - lonmax))
    iia_lat = np.argmin(np.abs(ROMS_data['lat'][:] - latmin))
    iib_lat = np.argmin(np.abs(ROMS_data['lat'][:] - latmax))
    # # crop further if needed:
    # if np.logical_or(iib_lon-iia_lon<nii,iib_lat-iia_lat<nii):
    #     nii = min(iib_lon-iia_lon, iib_lat-iia_lat)
    #     iia_lon = max(iilonWG - nii, 0)
    #     iib_lon = min(iilonWG + nii, iiL)
    #     iia_lat = max(iilatWG - nii, 0)
    #     iib_lat = min(iilatWG + nii, iiL)

    # find nearest time interval
    print(ROMS_data['time'].units)
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', ROMS_data['time'].units)
    basetime = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S').date())
    ttsim = []
    for hh in ROMS_data['time'][:]:
        ttsim.append(basetime + pd.Timedelta(hours=hh))
    ttsim = pd.to_datetime(ttsim)
    ttsim
    iit = np.abs(ttsim - tt_date).argmin()

    # get data
    u_sl = np.mean(ROMS_data['u'][iit, iida, iia_lat:iib_lat, iia_lon:iib_lon], axis=0)
    v_sl = np.mean(ROMS_data['v'][iit, iida, iia_lat:iib_lat, iia_lon:iib_lon], axis=0)
    lon_ROMS = ROMS_data['lon'][iia_lon:iib_lon]
    lat_ROMS = ROMS_data['lat'][iia_lat:iib_lat]
    # project to UTM to convert to meters
    myproj = pyproj.Proj(proj='utm', zone=55, ellps='WGS84', units='m', preserve_units=False)
    xy_lst = [myproj(x, y) for x, y in zip(lon_ROMS[:], lat_ROMS[:])]
    xUTM = np.array([x[0] for x in xy_lst])
    yUTM = np.array([y[-1] for y in xy_lst])
    dxx, dyy = np.meshgrid(np.gradient(xUTM), np.gradient(yUTM))
    # compute velocity magnitude
    vel_mag = np.sqrt(u_sl ** 2 + v_sl ** 2)
    # compute relative vorticity
    rel_vor = np.gradient(v_sl, axis=-1) / dxx - np.gradient(u_sl, axis=0) / dyy
    # Earth's rotation rate
    Om = 2 * np.pi / (24 * 3600)
    # planetrary vorticity (Coriolis parameter)
    f = 2 * Om * np.sin(np.deg2rad(np.mean(lat_ROMS)))
    # Rossby no.
    rel_vor_norm = rel_vor / f

    # temperature
    temp = np.mean(ROMS_data['temp'][iit, iida, iia_lat:iib_lat, iia_lon:iib_lon], axis=0)
    # salinity
    salt = np.mean(ROMS_data['salt'][iit, iida, iia_lat:iib_lat, iia_lon:iib_lon], axis=0)

    # ----------------------------------------------------------------------------------------------------------------------
    # LOAD WRF data
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        # filename
        filename = 'WRF_Guam_' + dates_str + '.ncd'
        dfnam    = os.path.join(datadir,filename)
        init_str = tt_date.strftime('%Y-%m-%d') + ' UTC'
        # Read-in data
        WRF_data = netcdf.Dataset(dfnam)
    except FileNotFoundError as err:
        print(err)

    # Grid points to show (box around ROTA)
    iia_lon = np.argmin(np.abs(WRF_data['lon'][:] - lonmin))
    iib_lon = np.argmin(np.abs(WRF_data['lon'][:] - lonmax))
    iia_lat = np.argmin(np.abs(WRF_data['lat'][:] - latmin))
    iib_lat = np.argmin(np.abs(WRF_data['lat'][:] - latmax))

    print(WRF_data['time'].units)
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', WRF_data['time'].units)
    basetime = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S').date())
    ttsim = []
    for hh in WRF_data['time'][:]:
        ttsim.append(basetime + pd.Timedelta(hours=hh))
    ttsim = pd.to_datetime(ttsim)
    iit = np.abs(ttsim - tt_date).argmin()

    # get data
    u10 = WRF_data['Uwind'][iit, iia_lat:iib_lat, iia_lon:iib_lon]
    v10 = WRF_data['Vwind'][iit, iia_lat:iib_lat, iia_lon:iib_lon]
    lon_WRF = WRF_data['lon'][iia_lon:iib_lon]
    lat_WRF = WRF_data['lat'][iia_lat:iib_lat]
    # compute wind velocity magnitude
    u10_mag = np.sqrt(u10 ** 2 + v10 ** 2)

    # ----------------------------------------------------------------------------------------------------------------------
    # LOAD WW3 data
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        # filename
        filename = 'WaveWatch_III_Mariana_' + dates_str + '.ncd'
        dfnam = os.path.join(datadir, filename)
        init_str = tt_date.strftime('%Y-%m-%d') + ' UTC'
        # Read-in data
        WW3_data = netcdf.Dataset(dfnam)
    except FileNotFoundError as err:
        print(err)

    # get bulk wave params near the Wave Glider
    # Grid points to show (box around ROTA)
    iia_lon = np.argmin(np.abs(WW3_data['lon'][:] - lonmin))
    iib_lon = np.argmin(np.abs(WW3_data['lon'][:] - lonmax))
    iia_lat = np.argmin(np.abs(WW3_data['lat'][:] - latmin))
    iib_lat = np.argmin(np.abs(WW3_data['lat'][:] - latmax))

    # find nearest time interval
    print(WW3_data['time'].units)
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', WW3_data['time'].units)
    basetime = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S').date())
    ttsim = []
    for hh in WW3_data['time'][:]:
        ttsim.append(basetime + pd.Timedelta(hours=hh))
    ttsim = pd.to_datetime(ttsim)
    iit = np.abs(ttsim - tt_date).argmin()

    # get data
    # peak wave direction (time, z, lat, lon)
    Dp = WW3_data['Tdir'][iit, 0, iia_lat:iib_lat, iia_lon:iib_lon]
    # compute wave direction components (based on Dp)
    uw = -np.sin(Dp * np.pi / 180)
    vw = -np.cos(Dp * np.pi / 180)
    # significant wave height (time, z, lat, lon)
    Hs = WW3_data['Thgt'][iit, 0, iia_lat:iib_lat, iia_lon:iib_lon]
    # peak wave period (time, z, lat, lon)
    Tp = WW3_data['Tper'][iit, 0, iia_lat:iib_lat, iia_lon:iib_lon]
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
    import cmocean

    # configure general plot parameters
    rel_vor_flg = True
    labsz = 12
    fntsz = 14
    widths = [1, 1]
    heights = [1, 1, 1]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)

    # configure general cartopy parameters
    LAND = NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',
                               facecolor=COLORS['land'])

    fig, axd = plt.subplot_mosaic([['ax1', 'ax2'], ['ax3', 'ax4'], ['ax5', 'ax6']], gridspec_kw=gs_kw,
                                  figsize=(14, 18), subplot_kw=dict(projection=ccrs.PlateCarree()))
    # --------------------------------------------------------------------------------------------
    # Hs
    # --------------------------------------------------------------------------------------------
    # config for Hs
    isub = 2
    levels = np.linspace(0, 5, 101)

    axd['ax1'].set_extent([lon_WW3.min(), lon_WW3.max(), lat_WW3.min(), lat_WW3.max()])
    gl = axd['ax1'].gridlines(draw_labels=True, linewidth=0)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # contour plot
    cf = axd['ax1'].contourf(lon_WW3, lat_WW3, Hs, levels=levels, cmap='jet', extend="both",
                             transform=ccrs.PlateCarree())

    # add land and coastlines
    axd['ax1'].add_feature(LAND)
    axd['ax1'].coastlines(resolution='10m')

    # # plot Wave Glider location
    # axd['ax1'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
    # axd['ax1'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
    # # plot Sea Glider location
    # axd['ax1'].plot(SG_stlon, SG_stlat,'.m')
    # axd['ax1'].plot(SG_enlon, SG_enlat,':m')
    # axd['ax1'].plot(SG_enlon[-1], SG_enlat[-1],'mo',mec='k',ms=12, label='SG526')
    # # Legend
    # axd['ax1'].legend(fontsize=fntsz,loc='upper left')

    # tick params
    axd['ax1'].tick_params(labelsize=fntsz)

    # Quiver plot
    legend_vel = 1.0
    isub = 2
    Q = axd['ax1'].quiver(lon_WW3[::isub], lat_WW3[::isub], uw[::isub, ::isub], vw[::isub, ::isub], pivot='middle')

    # colorbar and labels
    cb = fig.colorbar(cf, ax=axd['ax1'], shrink=0.95, ticks=np.linspace(levels[0], levels[-1], 11))
    cb.ax.set_title('m\n')
    axd['ax1'].set_title('WW3 significant wave height (Hs)');

    # --------------------------------------------------------------------------------------------
    # Tp
    # --------------------------------------------------------------------------------------------
    # config for Tp
    isub = 2
    levels = np.linspace(6, 14, 101)

    axd['ax2'].set_extent([lon_WW3.min(), lon_WW3.max(), lat_WW3.min(), lat_WW3.max()])
    gl = axd['ax2'].gridlines(draw_labels=True, linewidth=0)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # contour plot
    cf = axd['ax2'].contourf(lon_WW3, lat_WW3, Tp, levels=levels, cmap='jet', extend="both",
                             transform=ccrs.PlateCarree())

    # add land and coastlines
    axd['ax2'].add_feature(LAND)
    axd['ax2'].coastlines(resolution='10m')

    # # plot Wave Glider location
    # axd['ax2'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
    # axd['ax2'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
    # # plot Sea Glider location
    # axd['ax2'].plot(SG_stlon, SG_stlat,'.m')
    # axd['ax2'].plot(SG_enlon, SG_enlat,':m')
    # axd['ax2'].plot(SG_enlon[-1], SG_enlat[-1],'mo',mec='k',ms=12, label='SG526')
    # # Legend
    # axd['ax2'].legend(fontsize=fntsz,loc='upper left')
    # tick params
    axd['ax2'].tick_params(labelsize=fntsz)

    # Quiver plot
    legend_vel = 1.0
    Q = axd['ax2'].quiver(lon_WW3[::isub], lat_WW3[::isub], uw[::isub, ::isub], vw[::isub, ::isub], pivot='middle')

    # colorbar and labels
    cb = fig.colorbar(cf, ax=axd['ax2'], shrink=0.95, ticks=np.linspace(levels[0], levels[-1], 17))
    cb.ax.set_title('s\n')
    axd['ax2'].set_title('WW3 peak period (Tp)');

    # --------------------------------------------------------------------------------------------
    # Wind speed
    # --------------------------------------------------------------------------------------------
    # config for u10
    isub = 3
    levels = np.linspace(0, 15, 101)

    axd['ax3'].set_extent([lon_WRF.min(), lon_WRF.max(), lat_WRF.min(), lat_WRF.max()])
    gl = axd['ax3'].gridlines(draw_labels=True, linewidth=0)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # contour plot
    cf = axd['ax3'].contourf(lon_WRF, lat_WRF, u10_mag, levels=levels, cmap='jet', extend="both",
                             transform=ccrs.PlateCarree())

    # add land and coastlines
    axd['ax3'].add_feature(LAND)
    axd['ax3'].coastlines(resolution='10m')

    # # plot Wave Glider location
    # axd['ax3'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
    # axd['ax3'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
    # # plot Sea Glider location
    # axd['ax3'].plot(SG_stlon, SG_stlat,'.m')
    # axd['ax3'].plot(SG_enlon, SG_enlat,':m')
    # axd['ax3'].plot(SG_enlon[-1], SG_enlat[-1],'mo',mec='k',ms=12, label='SG526')
    # # Legend
    # axd['ax3'].legend(fontsize=fntsz,loc='upper left')
    # tick params
    axd['ax3'].tick_params(labelsize=fntsz)

    # Quiver plot
    legend_vel = 1.0
    Q = axd['ax3'].quiver(lon_WRF[::isub], lat_WRF[::isub], u10[::isub, ::isub], v10[::isub, ::isub], pivot='middle')

    # colorbar and labels
    cb = fig.colorbar(cf, ax=axd['ax3'], shrink=0.95, ticks=np.linspace(levels[0], levels[-1], 11))
    cb.ax.set_title('m s$^{-1}$\n')
    axd['ax3'].set_title('WRF 10-m winds');

    # --------------------------------------------------------------------------------------------
    # Currents
    # --------------------------------------------------------------------------------------------
    # config for currents
    if rel_vor_flg:
        cL = [-1.7, 1.7]
        levels = np.linspace(cL[0], cL[-1], 101)
    else:
        levels = np.linspace(0, 0.5, 101)

    axd['ax4'].set_extent([lon_ROMS.min(), lon_ROMS.max(), lat_ROMS.min(), lat_ROMS.max()])
    gl = axd['ax4'].gridlines(draw_labels=True, linewidth=0)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # contour plot
    if rel_vor_flg:
        cf = axd['ax4'].contourf(lon_ROMS, lat_ROMS, rel_vor_norm, levels=levels, cmap='bwr', extend="both",
                                 transform=ccrs.PlateCarree())
    else:
        cf = axd['ax4'].contourf(lon_ROMS, lat_ROMS, vel_mag, levels=levels, cmap='jet', extend="both",
                                 transform=ccrs.PlateCarree())

    # add land and coastlines
    axd['ax4'].add_feature(LAND)
    axd['ax4'].coastlines(resolution='10m')

    # # plot Wave Glider location
    # axd['ax4'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
    # axd['ax4'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
    # # plot Sea Glider location
    # axd['ax4'].plot(SG_stlon, SG_stlat,'.m')
    # axd['ax4'].plot(SG_enlon, SG_enlat,':m')
    # axd['ax4'].plot(SG_enlon[-1], SG_enlat[-1],'mo',mec='k',ms=12, label='SG526')
    # # Legend
    # axd['ax4'].legend(fontsize=fntsz,loc='upper left')
    # tick params
    axd['ax4'].tick_params(labelsize=fntsz)

    # Quiver plot
    legend_vel = 1.0
    isub = 3
    Q = axd['ax4'].quiver(lon_ROMS[::isub], lat_ROMS[::isub], u_sl[::isub, ::isub], v_sl[::isub, ::isub],
                          pivot='middle')

    # colorbar and labels
    if rel_vor_flg:
        cb = fig.colorbar(cf, ax=axd['ax4'], shrink=0.95)
        cb.ax.set_title('$\zeta/f$\n')
        axd['ax4'].set_title('ROMS vorticity')
    else:
        cb = fig.colorbar(cf, ax=axd['ax4'], shrink=0.95, ticks=np.linspace(levels[0], levels[-1], 11))
        cb.ax.set_title('m s$^{-1}$\n')
        axd['ax4'].set_title('ROMS sea surface currents')

    # --------------------------------------------------------------------------------------------
    # Temperature
    # --------------------------------------------------------------------------------------------
    # config for currents
    cL = [27.5, 28.6]
    levels = np.linspace(cL[0], cL[-1], 101)

    axd['ax5'].set_extent([lon_ROMS.min(), lon_ROMS.max(), lat_ROMS.min(), lat_ROMS.max()])
    gl = axd['ax5'].gridlines(draw_labels=True, linewidth=0)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # contour plot
    cf = axd['ax5'].contourf(lon_ROMS, lat_ROMS, temp, levels=levels, cmap=pjet_cmap, extend="both",
                             transform=ccrs.PlateCarree())
    # # contour plot
    # if rel_vor_flg:
    #     cf = axd['ax4'].contourf(lon_ROMS, lat_ROMS, rel_vor, levels=levels, cmap='bwr', extend="both",
    #                              transform=ccrs.PlateCarree())
    # else:
    #     cf = axd['ax4'].contourf(lon_ROMS, lat_ROMS, vel_mag, levels=levels, cmap='jet', extend="both",
    #                              transform=ccrs.PlateCarree())

    # add land and coastlines
    axd['ax5'].add_feature(LAND)
    axd['ax5'].coastlines(resolution='10m')

    # # plot Wave Glider location
    # axd['ax4'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
    # axd['ax4'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
    # # plot Sea Glider location
    # axd['ax4'].plot(SG_stlon, SG_stlat,'.m')
    # axd['ax4'].plot(SG_enlon, SG_enlat,':m')
    # axd['ax4'].plot(SG_enlon[-1], SG_enlat[-1],'mo',mec='k',ms=12, label='SG526')
    # # Legend
    # axd['ax4'].legend(fontsize=fntsz,loc='upper left')
    # tick params
    axd['ax5'].tick_params(labelsize=fntsz)

    cb = fig.colorbar(cf, ax=axd['ax5'], shrink=0.95)
    cb.ax.set_title('deg C\n')
    axd['ax5'].set_title('ROMS SST')

    # --------------------------------------------------------------------------------------------
    # Salinity
    # --------------------------------------------------------------------------------------------
    # config for currents
    cL = [34.25, 34.7]
    levels = np.linspace(cL[0], cL[-1], 101)

    axd['ax6'].set_extent([lon_ROMS.min(), lon_ROMS.max(), lat_ROMS.min(), lat_ROMS.max()])
    gl = axd['ax6'].gridlines(draw_labels=True, linewidth=0)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # contour plot
    cf = axd['ax6'].contourf(lon_ROMS, lat_ROMS, salt, levels=levels, cmap=cmocean.cm.haline, extend="both",
                             transform=ccrs.PlateCarree())

    # add land and coastlines
    axd['ax6'].add_feature(LAND)
    axd['ax6'].coastlines(resolution='10m')

    # # plot Wave Glider location
    # axd['ax4'].plot(Telemdf['longitude'].values,Telemdf['latitude'].values,':k')
    # axd['ax4'].plot(Telemdf['longitude'].values[-1],Telemdf['latitude'].values[-1],'gd',mec='k',ms=12, label=vnam)
    # # plot Sea Glider location
    # axd['ax4'].plot(SG_stlon, SG_stlat,'.m')
    # axd['ax4'].plot(SG_enlon, SG_enlat,':m')
    # axd['ax4'].plot(SG_enlon[-1], SG_enlat[-1],'mo',mec='k',ms=12, label='SG526')
    # # Legend
    # axd['ax4'].legend(fontsize=fntsz,loc='upper left')
    # tick params
    axd['ax6'].tick_params(labelsize=fntsz)

    cb = fig.colorbar(cf, ax=axd['ax6'], shrink=0.95)
    cb.ax.set_title('psu \n')
    axd['ax6'].set_title('ROMS Salinity');

    # --------------------------------------------------------
    # Set title
    # --------------------------------------------------------
    tmstmp = tt_date.strftime('%Y-%m-%d, %H:%M UTC')
    fig.suptitle('PacIOOS forecasts for ' + tmstmp + '\nInitialized ' + init_str, fontsize=fntsz + 2, y=0.945)

    # --------------------------------------------------------
    # Save figure
    # Set local paths and import local packages
    from pathlib import Path

    figdir = os.path.join(str(Path.home()), 'Documents/SIO/Projects/ARCTERX/figz/ROMSfigz_v0')
    figname = tt_date.strftime('%Y%m%d%H') + '_ROMSout' + '.jpg'
    fig.savefig(os.path.join(figdir, figname), dpi=100, bbox_inches='tight')

