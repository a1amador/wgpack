import sys,os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Data service path
ds_folder = os.path.join(str(Path.home()), 'src/lri-wgms-data-service')
if ds_folder not in sys.path:
    sys.path.insert(0, ds_folder)
from DataService import DataService

def readDS_ADCP_C4T(vnam, start_date, end_date=datetime.datetime.utcnow()):
    '''
    This function reads in 'ADCP C4T Samples' from Data Service and packages data into a dictionary
    :param vnam:
    :param start_date:
    :param end_date:
    :return:
    '''
    # Read in 'ADCP C4T Samples' from Data Service
    ds = DataService()
    # To get report names
    # ds.report_list
    # Get report
    start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_date = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    out = ds.get_report_data('ADCP C4T Samples', start_date, end_date, [vnam])

    tt, lon, lat = [], [], []
    VE, VN, ranges = [], [], []
    for d in out['report_data'][0]['vehicle_data']:
        tt.append(d['timeStamp'])
        lon.append(float(d['longitude']))
        lat.append(float(d['latitude']))
        # extract velocities, depth, etc.
        Evel, Nvel, R = [], [], []
        for jj in range(1, d['bin Count'] + 1):
            strE = 'bin' + str(jj) + ' East'
            strN = 'bin' + str(jj) + ' North'
            strR = 'bin' + str(jj) + ' Up'
            # velocities
            Evel.append(d[strE])
            Nvel.append(d[strN])
            # depth
            R.append(d[strR])
        # repackage
        VE.append(Evel)
        VN.append(Nvel)
        ranges.append(R)

    # convert to arrays
    tt = pd.to_datetime(tt)
    lon = np.array(lon)
    lat = np.array(lat)
    VE = np.array(VE) / 100
    VN = np.array(VN) / 100
    ranges = np.array(ranges) / 100

    # remove drop-outs
    # ll = np.diff(ranges[:,0])==0
    ll = np.logical_and(np.diff(ranges[:, 0]) < 1, np.diff(ranges[:, 0]) > -1)
    ll = np.where(np.append(False, ll))[0]
    ranges = np.delete(ranges, ll, axis=0)
    VE = np.delete(VE, ll, axis=0)
    VN = np.delete(VN, ll, axis=0)
    tt = np.delete(tt, ll, axis=0)
    lon = np.delete(lon, ll, axis=0)
    lat = np.delete(lat, ll, axis=0)

    # reshape matrices
    ranges = np.concatenate([ranges[0], ranges[1]])
    try:
        VE = np.reshape(VE, (int(VE.shape[0] / 2), len(ranges))).T
        VN = np.reshape(VN, (int(VN.shape[0] / 2), len(ranges))).T
        tt = tt[::2]
        lon = lon[::2]
        lat = lat[::2]
    except:
        VE = np.reshape(VE[:-1, :], (int(VE.shape[0] / 2), len(ranges))).T
        VN = np.reshape(VN[:-1, :], (int(VN.shape[0] / 2), len(ranges))).T
        tt = tt[:-1:2]
        lon = lon[:-1:2]
        lat = lat[:-1:2]

    # mask out nans
    VE[VE==99.99]=np.nan
    VN[VN==99.99]=np.nan

    # create output dictionary
    C4T = {
        'time': tt,
        'longitude': lon,
        'latitude': lat,
        'ranges': ranges,
        'Evel': VE,
        'Nvel': VN,
    }
    return C4T

