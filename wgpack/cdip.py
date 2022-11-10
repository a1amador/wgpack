# CDIP module
# 3rd party imports
import datetime
import numpy as np
import pandas as pd
import netCDF4 as nc
# local imports
from .timeconv import epoch2datetime64

def get_CDIP_displacement(stn, startdate, enddate=datetime.datetime.utcnow()):
    '''
    This function returns timeseries of directional displacement (x,y,z) using OPeNDAP service from CDIP THREDDS Server,
    where x = East/West, y = North/South, z = Up/Down
    qc_flag categories: [1,2,3,4,9] = [good,not_evaluated,questionable,bad,missing]
    references: http://cdip.ucsd.edu/themes/media/docs/documents/html_pages/dw_timeseries.html
    :param stn (str): station id
    :param startdate (str,timestamp,datetime): start date
    :param enddate (str,timestamp,datetime): end date
    :return (dataframe): Bulk wave parameters (x, y, z, qc_flag).
    '''
    import requests
    from bs4 import BeautifulSoup
    # time window
    ta = pd.to_datetime(startdate)
    tb = pd.to_datetime(enddate)

    # Read-in realtime data
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_xy.nc'
    ncspec = nc.Dataset(data_url)
    # check time coverage (realtime data)
    time_coverage_start = pd.to_datetime(ncspec.time_coverage_start).tz_localize(None)
    time_coverage_end = pd.to_datetime(ncspec.time_coverage_end).tz_localize(None)
    if ~np.logical_and(ta >= time_coverage_start, tb <= time_coverage_end):
        # check CDIP Archived Dataset URL
        # find full list of past deployments
        cat_url = "https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/"+stn+"p1/catalog.html"
        req = requests.get(cat_url)
        soup = BeautifulSoup(req.content, 'html.parser')
        substring = stn+'p1_d'
        deploy_lst = []
        for s in soup.find_all('a'):
            fullstring = s.text
            if substring in fullstring:
                # Read-in archived data
                data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/'+stn+'p1/'+fullstring
                ncspec = nc.Dataset(data_url)
                # check time coverage (archived data)
                time_coverage_start = pd.to_datetime(ncspec.time_coverage_start).tz_localize(None)
                time_coverage_end = pd.to_datetime(ncspec.time_coverage_end).tz_localize(None)
                if np.logical_and(ta >= time_coverage_start, tb <= time_coverage_end):
                    print(fullstring)
                    break

    # Turn off auto masking
    ncspec.set_auto_mask(False)

    # np.datetime64(datetime.datetime.utcfromtimestamp(ncspec.variables['xyzStartTime'][0]-ncspec.variables['xyzFilterDelay'][0]))
    filter_delay = ncspec.variables['xyzFilterDelay']
    start_time = ncspec.variables['xyzStartTime'][:] # Variable that gives start time for buoy data collection
    sample_rate = ncspec.variables['xyzSampleRate'][:] # Variable that gives rate (frequency, Hz) of sampling
    end_time = start_time + (len(ncspec.variables['xyzXDisplacement'])/sample_rate) # Calulate end time for buoy data collection

    # Find UNIX timestamps for user human-formatted start/end dates
    unix_start = (ta - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    unix_end = (tb - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # Create specialized array using UNIX Start and End times minus Filter Delay, and Sampling Period (1/sample_rate)
    # to calculate sub-second time values that correspond to Z-Displacement sampling values
    sample_time = np.arange((start_time - filter_delay[0]), end_time - filter_delay[0],(1/(sample_rate)))

    # Find corresponding start/end date index numbers in 'sample_time' array
    start_index = sample_time.searchsorted(unix_start)
    end_index = sample_time.searchsorted(unix_end)

    # Read Buoy Variables
    # Time
    tt = pd.to_datetime((sample_time[start_index:end_index]*1000).astype('datetime64[ms]'))
    # Three directional displacement variables (x, y, z)
    y = ncspec.variables['xyzXDisplacement'][start_index:end_index] # North
    x = ncspec.variables['xyzYDisplacement'][start_index:end_index] # East
    z = ncspec.variables['xyzZDisplacement'][start_index:end_index] # Vertical
    # QC Flag Categories: [1,2,3,4,9] = [good,not_evaluated,questionable,bad,missing]
    qc = ncspec.variables['xyzFlagPrimary'][start_index:end_index]

    # Filter out by quality control level
    # qc_level = 2 # Filter data with qc flags above this number
    # x = np.ma.masked_where(qc>qc_level,x)
    # y = np.ma.masked_where(qc>qc_level,y)
    # z = np.ma.masked_where(qc>qc_level,z)

    # Create dataframe
    d = {'Date': tt,
         'x': x, 'y': y, 'z': z, 'qc_flag': qc}
    df = pd.DataFrame(d)
    # set time as index
    df.set_index('Date', inplace=True)
    return df

def get_CDIP_wavesbulk(stn, startdate, enddate=datetime.datetime.utcnow()):
    '''
    This function returns bulk wave parameters using OPeNDAP service from CDIP THREDDS Server.
    references: http://cdip.ucsd.edu/themes/media/docs/documents/html_pages/compendium.html
    :param stn (str): station id
    :param startdate (str,timestamp,datetime): start date
    :param enddate (str,timestamp,datetime): end date
    :return (dataframe): Bulk wave parameters (Hsig, Tp, Ta, Dp).
    '''
    # time window
    ta = pd.to_datetime(startdate)
    tb = pd.to_datetime(enddate)

    # CDIP Archived Dataset URL
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    ncbulk = nc.Dataset(data_url)
    ckDate = epoch2datetime64(ncbulk.variables['waveTime'][:])
    if np.sum(np.logical_and(ckDate >= ta, ckDate <= tb)) == 0:
        # CDIP Realtime Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
        ncbulk = nc.Dataset(data_url)
    # Read Buoy Variables
    # Assign variable names to the relevant variables from NetCDF file:
    ncTime = ncbulk.variables['waveTime'][:]
    ncDate = epoch2datetime64(ncTime)
    # Extract the significant wave height, period and direction
    Hs = ncbulk.variables['waveHs'][:]
    Tp = ncbulk.variables['waveTp'][:]
    Ta = ncbulk.variables['waveTa'][:]
    Dp = ncbulk.variables['waveDp'][:]

    # Create dataframe
    d = {'Date': ncDate,
         'Hsig': Hs[:], 'Tp': Tp[:], 'Ta': Ta[:], 'Dp': Dp[:]}
    df = pd.DataFrame(d)
    # set time as index
    df.set_index('Date', inplace=True)
    # Crop time window
    kk = np.logical_and(df.index >= ta, df.index <= tb)
    df = df.loc[kk]
    return df

def get_CDIP_wavespec(stn, startdate, enddate=datetime.datetime.utcnow()):
    """
    This function returns wave energy density using OPeNDAP service from CDIP THREDDS Server.
    references: http://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html
    :param stn (str): station id
    :param startdate (str,timestamp,datetime): start date
    :param enddate (str,timestamp,datetime): end date
    :return (dataframe): wave energy density as a function of frequency.
    """
    # time window
    ta = pd.to_datetime(startdate)
    tb = pd.to_datetime(enddate)

    # CDIP Archived Dataset URL
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    ncspec = nc.Dataset(data_url)
    ckDate = epoch2datetime64(ncspec.variables['waveTime'][:])
    if np.sum(np.logical_and(ckDate >= ta, ckDate <= tb)) == 0:
        # CDIP Realtime Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
        ncspec = nc.Dataset(data_url)

    # Read Buoy Variables
    # Assign variable names to the relevant variables from NetCDF file:
    ncTime = ncspec.variables['waveTime'][:]
    ncDate = epoch2datetime64(ncTime)
    Fq = ncspec.variables['waveFrequency'][:]
    Ed = ncspec.variables['waveEnergyDensity'][:]

    # Create dataframe
    d = {Fq.data[i]: Ed[:, i].data for i in range(len(Fq))}
    d['Date'] = ncDate
    CDIPspecdf = pd.DataFrame(d)
    # set time as index
    CDIPspecdf.set_index('Date', inplace=True)

    # Crop time window
    kk = np.logical_and(CDIPspecdf.index >= ta, CDIPspecdf.index <= tb)
    CDIPspecdf = CDIPspecdf.loc[kk]
    return CDIPspecdf

def get_CDIP_Dmean(stn, startdate, enddate=datetime.datetime.utcnow()):
    '''
    This function returns wave direction as a function of frequency using OPeNDAP service from CDIP THREDDS Server.
    references: http://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html
    :param stn (str): station id
    :param startdate (str,timestamp,datetime): start date
    :param enddate (str,timestamp,datetime): end date
    :return (dataframe): wave direction as a function of frequency
    '''
    # time window
    ta = pd.to_datetime(startdate)
    tb = pd.to_datetime(enddate)

    # CDIP Archived Dataset URL
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    ncspec = nc.Dataset(data_url)
    ckDate = epoch2datetime64(ncspec.variables['waveTime'][:])
    if np.sum(np.logical_and(ckDate >= ta, ckDate <= tb)) == 0:
        # CDIP Realtime Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
        ncspec = nc.Dataset(data_url)

    # Read Buoy Variables
    # Assign variable names to the relevant variables from NetCDF file:
    ncTime = ncspec.variables['waveTime'][:]
    ncDate = epoch2datetime64(ncTime)
    Dmean = ncspec.variables['waveMeanDirection'][:]
    Fq = ncspec.variables['waveFrequency'][:]

    # Create dataframe
    d = {Fq.data[i]: Dmean[:, i].data for i in range(len(Fq))}
    d['Date'] = ncDate
    CDIPspecdf = pd.DataFrame(d)
    # set time as index
    CDIPspecdf.set_index('Date', inplace=True)

    # Crop time window
    kk = np.logical_and(CDIPspecdf.index >= ta, CDIPspecdf.index <= tb)
    CDIPspecdf = CDIPspecdf.loc[kk]
    return CDIPspecdf

def get_CDIP_wavevar(stn, var, startdate, enddate=datetime.datetime.utcnow()):
    '''
    This function returns variable of interest using OPeNDAP service from CDIP THREDDS Server.
    references: http://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html
    :param stn (str): station id
    :param var (str): variable of interest (see supported examples below)
                        var = 'waveFrequencyBounds'
                        var = 'waveBandwidth'
                        var = 'waveEnergyDensity'
                        var = 'waveMeanDirection'
                        var = 'waveA1Value'
                        var = 'waveB1Value'
                        var = 'waveA2Value'
                        var = 'waveB2Value'
                        var = 'waveCheckFactor'
                        var = 'waveSpread'
                        var = 'waveM2Value'
                        var = 'waveN2Value'
                        var = 'sstSeaSurfaceTemperature'
    :param startdate (str,timestamp,datetime): start date
    :param enddate (str,timestamp,datetime): end date
    :return (dataframe): variable of interest as a function of frequency (and time).
    '''
    # time window
    ta = pd.to_datetime(startdate)
    tb = pd.to_datetime(enddate)

    # CDIP Archived Dataset URL
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    ncspec = nc.Dataset(data_url)
    ckDate = epoch2datetime64(ncspec.variables['waveTime'][:])
    if np.sum(np.logical_and(ckDate >= ta, ckDate <= tb)) == 0:
        # CDIP Realtime Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
        ncspec = nc.Dataset(data_url)

    # Read Buoy Variables
    # Assign variable names to the relevant variables from NetCDF file:
    ncTime = ncspec.variables['waveTime'][:]
    ncDate = epoch2datetime64(ncTime)

    # Crop time window
    kk = np.logical_and(ncDate >= ta, ncDate <= tb)

    Fq = ncspec.variables['waveFrequency'][:]
    if (var == 'waveFrequencyBounds'):
        vardata = ncspec.variables[var][:]
        d = {Fq.data[i]: vardata[i].data for i in range(len(Fq))}
        # Create dataframe
        CDIPvardf = pd.DataFrame(d)
    elif (var == 'waveBandwidth'):
        vardata = ncspec.variables[var][:]
        d = {Fq.data[i]: vardata.data[i] for i in range(len(Fq))}
        CDIPvardf = pd.Series(d)
    elif (var == 'sstSeaSurfaceTemperature'):
        vardata = ncspec.variables[var][kk]
        d = {"SST": vardata.data}
        d['Date'] = ncDate[kk]
        # Create dataframe
        CDIPvardf = pd.DataFrame(d)
        # set time as index
        CDIPvardf.set_index('Date', inplace=True)
    else:
        vardata = ncspec.variables[var][kk]
        d = {Fq.data[i]: vardata[:, i].data for i in range(len(Fq))}
        d['Date'] = ncDate[kk]
        # Create dataframe
        CDIPvardf = pd.DataFrame(d)
        # set time as index
        CDIPvardf.set_index('Date', inplace=True)
    return CDIPvardf

def get_CDIP_currents(stn, startdate, enddate=datetime.datetime.utcnow()):
    '''
    This function returns water current data at 0.75 m depth obtained from the CDIP THREDDS server.
    references: https://docs.google.com/document/d/1Uz_xIAVD2M6WeqQQ_x7ycoM3iKENO38S4Bmn6SasHtY/edit
    :param stn (str): station id
    :param startdate (str,timestamp,datetime): start date
    :param enddate (str,timestamp,datetime): end date
    :return (dataframe): water current data at 0.75 m depth
    '''
    # time window
    ta = pd.to_datetime(startdate)
    tb = pd.to_datetime(enddate)

    # CDIP Archived Dataset URL
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_historic.nc'
    ncspec = nc.Dataset(data_url)
    ckDate = epoch2datetime64(ncspec.variables['acmTime'][:])
    if np.sum(np.logical_and(ckDate >= ta, ckDate <= tb)) == 0:
        # CDIP Realtime Dataset URL
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'
        ncspec = nc.Dataset(data_url)

    # Read Buoy Variables
    # Assign variable names to the relevant variables from NetCDF file:
    try:
        ncTime = ncspec.variables['acmTime'][:]
        ncDate = epoch2datetime64(ncTime)

        Speed = ncspec.variables['acmSpeed'][:].data
        Direction = ncspec.variables['acmDirection'][:].data
        VerticalSpeed = ncspec.variables['acmVerticalSpeed'][:].data
        SpeedStdDev = ncspec.variables['acmSpeedStdDev'][:].data
        DirectionStdDev = ncspec.variables['acmDirectionStdDev'][:].data
        VerticalSpeedStdDev = ncspec.variables['acmVerticalSpeedStdDev'][:].data
        u = np.sin(Direction * np.pi / 180) * Speed
        v = np.cos(Direction * np.pi / 180) * Speed
        # TODO: u_StdDev, v_StdDev
        # Create dictionary
        d = {
            'Date': ncDate,
            'Speed (z=-0.75m)': Speed,
            'Direction': Direction,
            'VerticalSpeed': VerticalSpeed,
            'SpeedStdDev': SpeedStdDev,
            'DirectionStdDev': DirectionStdDev,
            'VerticalSpeedStdDev': VerticalSpeedStdDev,
            'u': u,
            'v': v
        }
        # Create dataframe
        CDIPcurrdf = pd.DataFrame(d)
        # set time as index
        CDIPcurrdf.set_index('Date', inplace=True)
        # Crop time window
        kk = np.logical_and(CDIPcurrdf.index >= ta, CDIPcurrdf.index <= tb)
        CDIPcurrdf = CDIPcurrdf.loc[kk]
        return CDIPcurrdf
    except:
        print('CDIP current data unavailable for this buoy')
        return None
