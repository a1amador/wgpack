# CORDC-fabricated METOC package module
import numpy as np
import pandas as pd
import netCDF4 as nc
from .timeconv import epoch2datetime64

# knots to m/s
kt2mps = 0.514444

def read_mwbn(fname):
    '''
    This function reads in Miniature Wave Buoy (mwb) data - GPS-based directional wave measurements
    :param fname (str): full path + file name (.nc, NETCDF4_CLASSIC data model, file format HDF5)
    :return (dict)    : dictionary structure with relevant wave variables
    '''
    ds = nc.Dataset(fname)
    buoy = {}
    for key in ds.variables:
        buoy.update({key : ds[key][:].data})

    # convert epoch time into numpy datetime64
    buoy['time'] = epoch2datetime64(buoy['time'])
    # use this to convert to datetime (after converting to datetime64)
    # buoy['time'].astype(datetime.datetime)

    for key in ds.dimensions:
        if key.find('time_filt_fs1')==0:
            buoy['time_filt_fs1'] = epoch2datetime64(buoy['time_filt_fs1'])
        elif key.find('time_filt_fs2')==0:
            buoy['time_filt_fs2'] = epoch2datetime64(buoy['time_filt_fs2'])
        elif key.find('time_filt_fs3')==0:
            buoy['time_filt_fs3'] = epoch2datetime64(buoy['time_filt_fs3'])
        elif key.find('time_filt_fs4')==0:
            buoy['time_filt_fs4'] = epoch2datetime64(buoy['time_filt_fs4'])
        elif key.find('time_nofilt_fs1')==0:
            buoy['time_nofilt_fs1'] = epoch2datetime64(buoy['time_nofilt_fs1'])
        elif key.find('time_nofilt_fs2')==0:
            buoy['time_nofilt_fs2'] = epoch2datetime64(buoy['time_nofilt_fs2'])

    # for additional info
    # print(ds.id)
    # print(ds.title)
    # print(ds.summary)
    return buoy

def read_miniwavebuoy(fname):
    '''
    This function reads in Mini Wave Buoy (mwb) data - bulk wave parameters
    :param fname: full path + file name (.nc)
    :return: dataframe structure with relevant wave variables
    '''
    # Read in and concatenate mwb data
    mwb = read_mwbn(fname)
    t_mwb = mwb['time']
    Hs_mwb = mwb['Hs']
    Tp_mwb = mwb['Tp']
    Dp_mwb = mwb['Dp']
    cog_mwb = mwb['cog']
    lon_mwb = mwb['lon']
    lat_mwb = mwb['lat']

    # Create dictionary
    mwbdict = {
        'Date': t_mwb,
        'Hs': Hs_mwb,
        'Tp': Tp_mwb,
        'Dp': Dp_mwb,
        'cog': cog_mwb,
        'lat': lat_mwb,
        'lon': lon_mwb
    }
    # Create dataframe
    mwbdf = pd.DataFrame(mwbdict)
    # set time as index
    mwbdf.set_index('Date', inplace=True)
    return mwbdf

def read_miniwavebuoy_spec(fname):
    '''
    This function reads in Mini Wave Buoy (mwb) data - wave spectra
    :param fname: full path + file name (.nc)
    :return: dataframe structure with spectral data
    '''
    # Read in and concatenate mwb data
    mwb = read_mwbn(fname)
    tspec_mwb = mwb['time_nofilt_fs2']
    E_mwb = mwb['energy_nofilt_fs2']
    f_mwb = mwb['frequency_nofilt_fs2']
    # Create dataframe for MWB spectral data
    mwbspec = pd.DataFrame(data=E_mwb,
                           index=tspec_mwb,
                           columns=f_mwb)
    # set time as index and sort dates
    mwbspec.index.name = 'Date'
    mwbspec.sort_index(inplace=True)

    return mwbspec

def read_miniwavebuoy_Dmeanspec(fname):
    '''
    This function reads in Mini Wave Buoy (mwb) data - mean wave direction as a function of frequency
    :param fname: full path + file name (.nc)
    :return: dataframe structure with mean wave direction as a function of frequency and time
    '''
    # Read in and concatenate mwb data
    mwb = read_mwbn(fname)
    tspec_mwb = mwb['time_nofilt_fs2']
    Dmean_mwb = mwb['mean_dir_nofilt_fs2']
    f_mwb = mwb['frequency_nofilt_fs2']
    # Create dataframe for MWB spectral data
    mwbDmean = pd.DataFrame(data=Dmean_mwb,
                           index=tspec_mwb,
                           columns=f_mwb)
    # set time as index and sort dates
    mwbDmean.index.name = 'Date'
    mwbDmean.sort_index(inplace=True)

    return mwbDmean

def read_metbuoy(fname):
    '''
    This function reads in Metbuoy (mmb) data - WXT-based weather measurements
    :param fname (str)  : full path + file name (.dat)
    :return (dataframe) : dataframe structure with relevant weather variables
    '''

    # read in and concatenate WXT data
    WXTdat = np.loadtxt(fname, skiprows=58)

    # --------------------------------------------------------
    # time
    tep_WXT = WXTdat[:,0]       # %column_001: Epoch (unit=s)
    t_WXT = pd.to_datetime(tep_WXT, unit='s')
    # wind speed
    wspd_WXT = WXTdat[:,11]     # %column_012: Vector Averaged Wind Speed (Knots)
    wspd10_WXT = WXTdat[:,14]   # %column_015: Wind Speed at 10m (Knots)
    # wind direction
    wdir_WXT = WXTdat[:,10]     # %column_011: Vector Averaged Wind Direction (deg)
    # pressure
    Hg2mb=33.8639
    SLpressure_WXT = WXTdat[:,22] # %column_023: Sea Level Pressure (mb)
    pressure_WXT = WXTdat[:, 20]*Hg2mb  # %column_021: Barometric Pressure (in. Hg)
    # temperature
    temp_WXT = WXTdat[:,15]     # %column_016: Air Temperature (C)
    # relative humidity
    relhum_WXT = WXTdat[:, 16]  # %column_017: Relative Humidity (%)
    # Latitude and Longitude
    lat_WXT = WXTdat[:,7]       # %column_008: Latitude (deg N)
    lon_WXT = WXTdat[:,8]       # %column_009: Longitude (deg E)

    # Sort values
    p           = tep_WXT.argsort()
    tep_WXT     = tep_WXT[p]
    t_WXT       = t_WXT[p]
    wspd_WXT    = wspd_WXT[p]
    wspd10_WXT  = wspd10_WXT[p]
    wdir_WXT    = wdir_WXT[p]
    pressure_WXT= pressure_WXT[p]
    SLpressure_WXT = SLpressure_WXT[p]
    temp_WXT    = temp_WXT[p]
    relhum_WXT  = relhum_WXT[p]
    lat_WXT     = lat_WXT[p]
    lon_WXT     = lon_WXT[p]
    # Create dictionary
    WXTdict = {
        'Date'          : t_WXT,
        'WindSpeed'     : wspd_WXT,
        'WindSpeed10'   : wspd10_WXT,
        'WindDirection' : wdir_WXT,
        'SLpressure'    : SLpressure_WXT,
        'pressure'      : pressure_WXT,
        'temperature'   : temp_WXT,
        'RelativeHumidity': relhum_WXT,
        'latitude'      : lat_WXT,
        'longitude'     : lon_WXT
    }
    # Create dataframe
    WXTdf = pd.DataFrame(WXTdict)
    # set time as index
    WXTdf.set_index('Date',inplace=True)
    return WXTdf


def read_metbuoy_SBD(fname):
    '''
    This function reads in SBD Metbuoy (mmb) data - WXT-based weather measurements
    :param fname (str)  : full path + file name (.dat)
    :return (dataframe) : dataframe structure with relevant weather variables
    '''

    # read in and concatenate WXT data
    WXTdat = np.loadtxt(fname, skiprows=126)

    # --------------------------------------------------------
    # time
    tep = WXTdat[:, 0]  # %column_001: Epoch (unit=s)
    tt = pd.to_datetime(tep, unit='s')
    # Latitude and Longitude
    lat = WXTdat[:, 7]  # %column_008: Latitude (deg N)
    lon = WXTdat[:, 8]  # %column_009: Longitude (deg E)
    # true wind speed and directions
    wspdE = WXTdat[:, 10] * kt2mps  # %column_011: True wind mean eastward velocity (Knots)
    wspdN = WXTdat[:, 12] * kt2mps  # %column_013: True wind mean northward velocity (Knots)
    wspd = WXTdat[:, 14] * kt2mps  # %column_015: True wind mean speed (Knots)
    wdir = WXTdat[:, 15]  # %column_016: True wind mean direction (deg)
    wspd_min = WXTdat[:, 16] * kt2mps  # %column_017: True wind minimum speed (Knots)
    wspd_max = WXTdat[:, 17] * kt2mps  # %column_018: True wind maximum speed (Knots)
    wspd10 = WXTdat[:, 34] * kt2mps  # %column_035: True Wind speed at 10m (Knots)
    # pressure
    Hg2mb = 33.8639
    pressure_baro = WXTdat[:, 36] * Hg2mb  # %column_037: Barometric Pressure (in. Hg)
    pressure_SL = WXTdat[:, 41]  # %column_042: Sea Level Pressure (mb)
    # temperature
    Tair = WXTdat[:, 37]  # %column_038: Air Temperature (C)
    # relative humidity
    Rh = WXTdat[:, 38]  # %column_039: Relative Humidity (%)
    # platform speed and course over ground
    cog = WXTdat[:, 71]  # %column_072: Mean course over ground (deg)
    sog = WXTdat[:, 74] * kt2mps  # %column_075: Mean speed over ground (Knots)
    # sogE = WXTdat[:,76]*kt2mps # %column_077: Platform mean eastward velocity (Knots)
    # sogN = WXTdat[:,78]*kt2mps # %column_079: Platform mean northward velocity (Knots)
    # compass heading
    heading = WXTdat[:, 110] # %column_111: Compass mean heading (deg)


    # Create dictionary
    WXTdict = {
        'Date': tt,
        'latitude': lat,
        'longitude': lon,
        'wspdE': wspdE,
        'wspdN': wspdN,
        'wspd_min': wspd_min,
        'wspd_max': wspd_max,
        'WindSpeed': wspd,
        'WindSpeed10': wspd10,
        'WindDirection': wdir,
        'pressure_SL': pressure_SL,
        'pressure_baro': pressure_baro,
        'temperature': Tair,
        'RelativeHumidity': Rh,
        'cog': cog,
        'sog': sog,
        'heading': heading,
    }
    # Create dataframe
    WXTdf = pd.DataFrame(WXTdict)
    # set time as index
    WXTdf.set_index('Date', inplace=True)
    WXTdf.sort_index(inplace=True)
    return WXTdf