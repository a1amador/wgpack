import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
from wgpack.dportal import veh_list, readDP_CTD, OxHz2DO

# --------------------------------------------------------
# Set local paths and import local packages
calcofi_path = os.path.join(os.path.abspath(os.path.join('..')),'calcofi')
if calcofi_path not in sys.path:
    sys.path.insert(0, calcofi_path)
from WGcode.WGhelpers import sftp_put_cordcdev,sftp_remove_cordcdev
# from WGcode.WGhelpers import veh_list, sftp_put_cordcdev,sftp_remove_cordcdev
# from WGcode.slackhelpers import send_slack_file

# --------------------------------------------------------
args = sys.argv
# Select vehicle
# vnam = 'sv3-253'
# vnam = 'magnus'
# vnam = 'sv3-251'
# Select channels
# channels = 'G012WTLC56K'    #calcofi_2020
# channels = 'C0158P2JJTT'    #app-tests
# Select time window
# tw = 'None'
# Select project
# prj = 'calcofi'
# prj = 'tfo'
vnam    = args[1]           # vehicle name
vid     = veh_list[vnam]    # vehicle id (Data Portal)
channels= args[2]           # Slack channel
tw      = args[3]           # time window to display specified in days (ends at present time)
tw = None if tw == 'None' else tw
prj     = args[4]           # project folder name (e.g., 'calcofi', 'tfo', etc.)
# vnam,channels,tw,prj =  'sv3-251','C0158P2JJTT','None','westpac'
# tw = None if tw == 'None' else tw
# vid = veh_list[vnam]

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
    tst = datetime(2022, 3, 9, 8, 0, 0, 0)
# Read in CTD output from Data Portal
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
# Load and interpolate ROMS data onto WG location
# ----------------------------------------------------------------------------------------------------------------------
if prj == 'westpac':
    import netCDF4 as netcdf
    import re
    from scipy.interpolate import griddata
    datadir = os.path.join(os.path.dirname(seachest_data_dir), 'ARCTERX2022/forecasts')

    # ------------------------------------------------------------------------------------------------------------------
    # Read in vehicle location from Data Service
    # Data service path
    ds_folder = os.path.join(str(Path.home()), 'src/lri-wgms-data-service')
    if ds_folder not in sys.path:
        sys.path.insert(0, ds_folder)
    from DataService import DataService
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

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD ROMS data
    ROMS_data_lst=[]
    T_ROMSi,S_ROMSi,t_ROMSi=[],[],[]
    T_ROMSn,S_ROMSn=[],[]
    for ii in np.flip(range((now - tst+pd.Timedelta(days=float(1))).days)):
        print(ii)
        tsti = now - pd.Timedelta(days=float(ii))
        try:
            # filename
            filename = 'ROMS_Guam_' + tsti.strftime('%Y%m%d') + '.ncd'
            dfnam = os.path.join(datadir, filename)
            print(dfnam)
            init_str = tsti.strftime('%Y-%m-%d') + ' UTC'
            # Read-in data
            ROMS_data = netcdf.Dataset(dfnam)
            # ROMS_data_lst.append(ROMS_data)
            # extract coordinates
            iia_lat, iia_lon = 0, 0
            iib_lat = min(len(ROMS_data['lon'][:]), len(ROMS_data['lat'][:]))
            iib_lon = min(len(ROMS_data['lon'][:]), len(ROMS_data['lat'][:]))
            lon_ROMS = ROMS_data['lon'][iia_lon:iib_lon]
            lat_ROMS = ROMS_data['lat'][iia_lat:iib_lat]

            # find runs for day 1
            print(ROMS_data['time'].units)
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', ROMS_data['time'].units)
            basetime = pd.to_datetime(datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S').date())
            ttsim = []
            for hh in ROMS_data['time'][:]:
                ttsim.append(basetime + pd.Timedelta(hours=hh))
            ttsim = pd.to_datetime(ttsim)
            # find runs for day 1
            iit = 8
            print(ttsim[0])

            # get surface layer velocities near the CTD depth
            sub_depth = ctdDPdf['pressure'].mean()
            iida = np.abs(ROMS_data['depth'][:] - (sub_depth)).argmin()

            for jj,tt in enumerate(ttsim[:iit]):
                # find nearest ROMS point to WG in time and space
                iit_ctd = np.abs(ctdDPdf.index - tt).argmin()
                iit_tel = np.abs(Telemdf.index - ctdDPdf.index[iit_ctd]).argmin()
                lonWG = Telemdf['longitude'].values[iit_tel]
                latWG = Telemdf['latitude'].values[iit_tel]

                # Interpolate model output onto vehicle's location
                # Reference: https://stackoverflow.com/questions/42504987/python-interpolate-point-value-on-2d-grid
                LON, LAT = np.meshgrid(lon_ROMS, lat_ROMS)
                points = np.array((LON.flatten(), LAT.flatten())).T
                # for nearest interpolation (sanity check)
                iin_lat = np.abs(lat_ROMS - latWG).argmin()
                iin_lon = np.abs(lon_ROMS - lonWG).argmin()
                # temperature
                T_ROMS = ROMS_data['temp'][jj,iida,iia_lat:iib_lat, iia_lon:iib_lon].data
                T_ROMSi.append(griddata(points, T_ROMS.flatten(), (lonWG, latWG),method='linear'))
                T_ROMSn.append(T_ROMS[iin_lat, iin_lon]) # nearest interpolation
                # salinity
                S_ROMS = ROMS_data['salt'][jj,iida,iia_lat:iib_lat, iia_lon:iib_lon].data
                S_ROMSi.append(griddata(points, S_ROMS.flatten(), (lonWG, latWG),method='linear'))
                S_ROMSn.append(S_ROMS[iin_lat, iin_lon]) # nearest interpolation
                # associated time
                t_ROMSi.append(tt)

        except FileNotFoundError as err:
            print(err)
    # convert to numpy array
    T_ROMSi, S_ROMSi, t_ROMSi = np.array(T_ROMSi), np.array(S_ROMSi), np.array(t_ROMSi)
    T_ROMSn, S_ROMSn = np.array(T_ROMSn), np.array(S_ROMSn)

# ----------------------------------------------------------------------------------------------------------------------
# Load Sea Glider data
# ----------------------------------------------------------------------------------------------------------------------
if prj == 'westpac':
    import netCDF4 as netcdf
    from wgpack.timeconv import epoch2datetime64
    SGdatadir = os.path.join(os.path.dirname(seachest_data_dir), 'ARCTERX2022/sg526')
    # filename = 'sg526_ARCTERX_1.0m_up_and_down_profile.nc'
    filename = 'sg526_ARCTERX_timeseries.nc'
    SGfnam = os.path.join(SGdatadir, filename)
    SG_data = netcdf.Dataset(SGfnam)
    # Sea Glider time
    # ttSG = pd.to_datetime(epoch2datetime64(SG_data['start_time'][:]))
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
    SG_T = SG_data['temperature_raw'][iiaSGctd:]
    SG_S = SG_data['salinity_raw'][iiaSGctd:]
    SG_d = SG_data['ctd_depth'][iiaSGctd:]
    ttSGctd = ttSGctd[iiaSGctd:]
    # crop by depth
    # depth of Wave Glider CTD
    dWGctd = np.nanmean(ctdDPdf['pressure'].values)
    iidSGctd = np.logical_and(SG_d<dWGctd+1,SG_d>dWGctd-1)

# ----------------------------------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------------------------------
if flg_plt:
    labsz = 8
    fntsz = 10
    tL = [tst, now]
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    # # Pressure
    # ax[0].plot(ctdDPdf['pressure'], '-k')
    # ax[0].set_ylabel('Pressure [dbar]', fontsize=fntsz)
    # Temperature
    l0, = ax[0].plot(ctdDPdf['temperature'], '-', color='b',label='WG')
    ax[0].set_ylabel('Temperature [deg C]', fontsize=fntsz)

    # Salinity
    l1, = ax[1].plot(ctdDPdf['salinity'], '-b',label='WG')
    ax[1].set_ylabel('Salinity [psu]', fontsize=fntsz)

    # Dissolved Oxygen
    l2, = ax[2].plot(ctdDPdf['DO (ml/L)'], '-b', label='WG')
    ax[2].set_ylabel('DO [ml/L]', fontsize=fntsz)
    ax[2].set_xlabel('Time [UTC]', fontsize=fntsz)
    ax[2].set_xlim(tL[0], tL[-1])

    if prj == 'westpac':
        ax[0].plot(t_ROMSi, T_ROMSi, '.-', color='gray', label='ROMS')
        ax[0].plot(ttSGctd[iidSGctd], SG_T[iidSGctd], 'o', color='m', label='SG')
        # ax[0].plot(t_ROMSi, T_ROMSn, '--', color='m', label='ROMS nearest')
        ax[1].plot(t_ROMSi, S_ROMSi, '.-', color='gray', label='ROMS')
        ax[1].plot(ttSGctd[iidSGctd], SG_S[iidSGctd], 'o', color='m', label='SG')
        # ax[1].plot(t_ROMSi, S_ROMSn, '--', color='m', label='ROMS nearest')
        ax[0].legend(fontsize=fntsz)

    tmstmp = now.strftime('%Y-%m-%d, %H:%M UTC')
    fig.suptitle(vnam + ': CTD output ' + tmstmp, fontsize=16)

    # add grid lines
    ax = fig.get_axes()
    for axi in ax:
        axi.grid()
    fig.show()

    # --------------------------------------------------------
    # Save figure
    # Set local paths and import local packages
    from pathlib import Path
    loc_folder = os.path.join(str(Path.home()), 'src/calcofi/WGautonomy/auto_plots')
    figdir = os.path.join(loc_folder, 'figz', vnam)
    figname = 'CTD_' + vnam + '.png'
    fig.savefig(os.path.join(figdir, figname), dpi=100, bbox_inches='tight')

    # --------------------------------------------------------
    # Upload to CORDCdev - need to be on VPN (sio-terrill pool) to access CORDCdev
    # TODO: Consider defining these paths in config file (settings.py)
    LOCALpath_CD = os.path.join(figdir, figname)
    if prj == 'westpac':
        REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, vnam, figname)
    elif prj=='norse' or prj=='palau':
        REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, figname)
    else:
        REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, 'waveglider', figname)
    SFTPAttr = sftp_put_cordcdev(LOCALpath_CD,REMOTEpath_CD)
    print('done uploading CTD plots to CORDCdev')

    # # --------------------------------------------------------
    # # Upload file to Slack at 16 or 23 UTC (9 AM or 4 PM PDT)
    # if np.logical_or(datetime.utcnow().hour==16,datetime.utcnow().hour==23):
    #     # Inputs:
    #     filepath = LOCALpath_CD
    #     title = vnam + " CTD"
    #     message = vnam + " CTD update for " + tmstmp
    #     # Report to Slack
    #     send_slack_file(channels,filepath,title,message)
else:
    # Remove file from CORDCdev - need to be on VPN (sio-terrill pool) to access CORDCdev
    from pathlib import Path
    loc_folder = os.path.join(str(Path.home()), 'src/calcofi/WGautonomy/auto_plots')
    figdir = os.path.join(loc_folder, 'figz', vnam)
    figname = 'CTD_' + vnam + '.png'
    if prj == 'westpac':
        REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, vnam, figname)
    elif prj == 'norse' or prj == 'palau':
        REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, figname)
    else:
        REMOTEpath_CD = os.path.join('/var/www/sites/cordcdev/data', prj, 'waveglider', figname)
    # Remove figure from CORDCdev
    try:
        SFTPAttr = sftp_remove_cordcdev(REMOTEpath_CD)
        print('CTD plots removed from CORDCdev')
    except:
        print("CTD plots not available in CORDCdev")