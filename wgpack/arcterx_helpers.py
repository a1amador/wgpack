import sys, os
import pandas as pd
import numpy as np
from pathlib import Path
from geopy.distance import distance


module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.dportal import veh_list, readDP_CTD, OxHz2DO
from wgpack.nav import get_bearing

# Data service path
ds_folder = os.path.join(str(Path.home()),'src/lri-wgms-data-service')
if ds_folder not in sys.path:
    sys.path.insert(0, ds_folder)
from DataService import DataService

# knots to m/s
kt2mps = 0.514444

def readDP_CTD_interp(vnam,tst,ten):

    vid = veh_list[vnam]

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
    end_date = ten.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    out = ds.get_report_data('Telemetry 6 Report', start_date, end_date, [vnam])
    # Convert to pandas dataframe
    Telemdf = pd.json_normalize(out['report_data'][0]['vehicle_data']['recordData'])
    # set timeStamp column as datetimeindex
    Telemdf = Telemdf.set_index(pd.DatetimeIndex(Telemdf['gliderTimeStamp'].values))
    Telemdf.drop(columns=['gliderTimeStamp'], inplace=True)
    # sort index
    Telemdf.sort_index(inplace=True)

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

    # create dict
    ctd_idict = {
        'time': tt_tel,
        'longitude': Telemdf['longitude'].values,
        'latitude': Telemdf['latitude'].values,
        'T': T,
        'S': S
    }
    # create dataframe
    ctd_df = pd.DataFrame(ctd_idict)
    ctd_df.set_index('time',inplace=True)
    return ctd_df

def compute_sogcogDS(vnam, tst, ten):
    # ----------------------------------------------------------------------------------------------------------------------
    # Read in vehicle location from Data Service
    # ----------------------------------------------------------------------------------------------------------------------
    # instantiate data-service object
    ds = DataService()
    # To get report names
    # print(ds.report_list)

    start_date = tst.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_date = ten.strftime("%Y-%m-%dT%H:%M:%S.000Z")
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
    sog_lonlat, cog_lonlat = [], []
    cc = 0
    for index, row in Telemdf[:-1].iterrows():
        cc += 1
        p1 = (row['latitude'], row['longitude'])
        p2 = (Telemdf.iloc[cc]['latitude'], Telemdf.iloc[cc]['longitude'])
        delt = (Telemdf.index[cc] - index) / np.timedelta64(1, 's')
        sogtmp = distance(p1, p2).m / delt if delt > 0 else np.nan
        sog_lonlat.append(sogtmp)
        cog_lonlat.append(get_bearing(p1, p2))

    # not sure if these are trustworthy
    speedOverGround = Telemdf['speedOverGround'].values * kt2mps
    gliderSpeed = Telemdf['gliderSpeed'].values * kt2mps

    # create dict
    nav_dict = {
        'time': Telemdf.index,
        'longitude': Telemdf['longitude'].values,
        'latitude': Telemdf['latitude'].values,
        'sog': sog_lonlat,
        'cog': cog_lonlat
    }
    # create dataframe
    nav_df = pd.DataFrame(nav_dict)
    nav_df.set_index('time', inplace=True)
    return nav_df
