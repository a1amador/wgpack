# High frequency radar (HFR) module
import datetime
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
import glob
from .timeconv import timeIndexToDatetime

def get_HFR_currents(data_url, myLon, myLat, startdate, enddate=datetime.datetime.utcnow()):
    '''
    This function returns sea surface current real time vectors at a single point from HFRnet THREDDS server.
    References:
    https://github.com/rowg/HFRnet-Thredds-support/blob/master/PythonNotebooks/TimeseriesRTVfromSIO_TDS.ipynb
    https://hfrnet-tds.ucsd.edu/thredds/catalog.html
    http://cordc.ucsd.edu/projects/mapping/
    :param data_url (str): adjust your map parameters if you change regions, some examples shown below:
    USEGC region
    netcdf_data = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/6km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_6km_Resolution_Hourly_RTV_best.ncd'
    netcdf_data = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd'
    USWC region
    data_url = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/500m/hourly/RTV/HFRADAR_US_West_Coast_500m_Resolution_Hourly_RTV_best.ncd'
    data_url = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd'
    data_url = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd'
    data_url = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/6km/hourly/RTV/HFRADAR_US_West_Coast_6km_Resolution_Hourly_RTV_best.ncd'
    :param myLon (float,list or array): Longitude of interest
    :param myLat (float,list or array): Latitude of interest
    :param startdate (str,timestamp,datetime): start date
    :param enddate (str,timestamp,datetime): end date
    :return (dataframe): sea surface current real time vectors at a single point from HFRnet
    '''
    # Load RTV dataset through THREDDS
    netcdf_data = nc.Dataset(data_url)
    # Grab lat, lon and time variables from dataset
    lat = netcdf_data.variables['lat'][:]
    lon = netcdf_data.variables['lon'][:]
    time = netcdf_data.variables['time'][:]

    # set the time base
    baseTime = datetime.datetime.strptime(netcdf_data.variables['time'].units, "hours since %Y-%m-%d %H:%M:%S.%f %Z")
    # Turn time index into timestamps
    times = timeIndexToDatetime(baseTime, time)
    # convert to DatetimeIndex
    tt = pd.to_datetime(times)
    t_st = pd.to_datetime(startdate)
    t_en = pd.to_datetime(enddate)
    # find index of start and end time
    startIndex = tt.get_loc(t_st, method='nearest')
    endIndex = tt.get_loc(t_en, method='nearest')
    times = timeIndexToDatetime(baseTime, time[startIndex:endIndex + 1])

    # TODO interpolate myLon,myLat onto a common time base (use times) and then find corresponding coordinates
    # find index corresponding to closest coordinates
    if isinstance(myLon, float):
        lonIdx = (np.abs(lon - myLon)).argmin()
        latIdx = (np.abs(lat - myLat)).argmin()
    else:
        lonIdx, latIdx = [], []
        for ln, lt in zip(myLon, myLat):
            lonIdx.append((np.abs(lon - ln)).argmin())
            latIdx.append((np.abs(lat - lt)).argmin())

    # This loads u & v current component data
    # Note the indexing is [time, latitude, longitude]
    u = netcdf_data.variables['u'][startIndex:endIndex + 1, latIdx, lonIdx]
    v = netcdf_data.variables['v'][startIndex:endIndex + 1, latIdx, lonIdx]

    # Create dictionary
    d = {
        'Date': pd.to_datetime(times),
        'Lon': lon[lonIdx],
        'Lat': lat[latIdx],
        'u': u,
        'v': v
    }
    # Create dataframe
    HFRdf = pd.DataFrame(d)
    # set time as index
    HFRdf.set_index('Date', inplace=True)
    return HFRdf


def get_HFR_hourly_from_dir(hf_dir):
    '''
    This function extracts HFR data from a directory and merges into a dictionary
    :param hf_dir: full path to directory
                    (e.g., '/Volumes/cordc-data/PROJECTS/hfrnet-fs-data/cordc/hfrtv/PW/2022_12/NetCDF')
    :return: dictionary with HFR spatial and temporal data for that directory
    '''

    # get hourly files
    hf_filst = [f for f in glob.glob(os.path.join(hf_dir, '*uwls_SIO.nc')) if os.path.isfile(os.path.join(hf_dir, f))]

    times_lst, u_lst, v_lst = [], [], []
    for fi in np.sort(hf_filst):
        # get NetCDF data
        netcdf_data = nc.Dataset(fi)
        # Grab lat, lon and time variables from dataset
        lat = netcdf_data.variables['lat'][:]
        lon = netcdf_data.variables['lon'][:]
        time = netcdf_data.variables['time'][:]
        # set the time base
        baseTime = datetime.datetime.strptime(netcdf_data.variables['time'].units, "seconds since %Y-%m-%d")
        # Turn time index into timestamps
        times = timeIndexToDatetime(baseTime, time.data.astype(float),
                                    units=netcdf_data.variables['time'].units.split()[0])
        # convert to DatetimeIndex
        times_lst.append(times[0])
        # This loads u & v current component data
        # Note the indexing is [time, latitude, longitude]
        u_lst.append(netcdf_data.variables['u'][0, :, :].filled(fill_value=np.nan))
        v_lst.append(netcdf_data.variables['v'][0, :, :].filled(fill_value=np.nan))
    # Create dictionary
    hfr_d = {
        'time': pd.to_datetime(times_lst),
        'lon': lon.data,
        'lat': lat.data,
        'u': np.array(u_lst),
        'v': np.array(v_lst)
    }
    return hfr_d