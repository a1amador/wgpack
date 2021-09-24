# NOMADS module
# References:
# https://nomads.ncep.noaa.gov/

def getWW3_predictions(myLat,myLon):
    """
    This function retrieves Wave Watch 3 predictions from NOMADS (Ocean Models-->GFS Ensemble Wave).
    The operational ocean wave predictions of NOAA/NWS/NCEP use the wave model WAVEWATCH III using operational
    NCEP products as input.
    The model is run four times a day: 00Z, 06Z, 12Z, and 18Z. Each run starts with 9-, 6- and 3-hour hindcasts
    and produces forecasts of every 3 hours from the initial time out to 180 hours (84 hours for the Great Lakes).
    The wave model is run as a mosaic of eight grids (one global, four regional, and three costal) with two - way
    interaction between the grids.
    References:
    https://polar.ncep.noaa.gov/waves/examples/usingpython.shtml
    https://nomads.ncep.noaa.gov/dods/wave
    https://nomads.ncep.noaa.gov/dods/wave/gfswave
    https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php
    Details of the wave model can be found here
    http://polar.ncep.noaa.gov/waves/wavewatch/wavewatch.shtml
    :param myLat (float): Latitude
    :param myLon (float): Longitude
    :return: dataframe containing model output
    """
    import datetime
    import netCDF4 as nc
    import numpy as np
    import pandas as pd
    # Set up the URL to access the data server.
    # Construct time of most recent model run date
    now_utc = datetime.datetime.utcnow()
    # now_utc = 19
    if now_utc.hour>18:
        date_suffix = '_12z'
    elif now_utc.hour>12:
        date_suffix = '_06z'
    elif now_utc.hour>6:
        date_suffix = '_00z'
    else:
        now_utc = now_utc-datetime.timedelta(days=1)
        date_suffix = '_18z'
    mydate = now_utc.strftime("%Y%m%d")
    # See the NWW3 directory on NOMADS for the list of available model run dates
    # url='https://nomads.ncep.noaa.gov/dods/wave/nww3/nww3'+ \
    #       mydate+'/nww3'+mydate+date_suffix
    url = 'http://nomads.ncep.noaa.gov:80/dods/wave/gfswave/' \
        + mydate + '/gfswave.global.0p25' + date_suffix

    # Extract WW3 data
    ww3_ds = nc.Dataset(url)
    lat  = ww3_ds.variables['lat'][:]
    lon  = ww3_ds.variables['lon'][:]
    # modify longitude to wrap around 180
    lon0 = lon%180
    lon0[int(len(lon)/2+1):] = -lon0[int(len(lon)/2+1):][::-1]
    # indices associated with given coordinates
    latIdx = (np.abs(lat-myLat)).argmin()
    lonIdx = (np.abs(lon0-myLon)).argmin()

    # significant wave height [m]
    Hs_ww3 = ww3_ds.variables['htsgwsfc'][:,latIdx,lonIdx]
    # peak wave period [s]
    Tp_ww3 = ww3_ds.variables['perpwsfc'][:,latIdx,lonIdx]
    # primary wave direction [deg]
    Dp_ww3 = ww3_ds.variables['dirpwsfc'][:,latIdx,lonIdx]
    # surface u-component of wind [m/s]
    uw_ww3 = ww3_ds.variables['ugrdsfc'][:,latIdx,lonIdx]
    # surface v-component of wind [m/s]
    vw_ww3 = ww3_ds.variables['vgrdsfc'][:,latIdx,lonIdx]
    # surface wind direction (from which blowing) [deg]
    wdir_ww3 = ww3_ds.variables['wdirsfc'][:,latIdx,lonIdx]
    # surface wind speed [m/s]
    wspd_ww3 = ww3_ds.variables['windsfc'][:,latIdx,lonIdx]

    # Convert WW3 time into datetime64
    times = ww3_ds.variables['time'][:].data
    tt_ww3=[]
    for mt in times:
        tt_ww3.append(datetime.datetime.fromordinal(int(mt))
                      + datetime.timedelta(days=mt%1)
                      - datetime.timedelta(days=1))
    # convert to pandas datetime64
    tt_ww3 = pd.to_datetime(tt_ww3)

    # Create dictionary
    WW3dict = {
        'Date': tt_ww3,
        'Hs': Hs_ww3,
        'Tp': Tp_ww3,
        'Dp': Dp_ww3,
        'u_wind': uw_ww3,
        'v_wind': vw_ww3,
        'wdir': wdir_ww3,
        'wspd': wspd_ww3,
    }
    # Create dataframe
    WW3df = pd.DataFrame(WW3dict)
    # set time as index
    WW3df.set_index('Date', inplace=True)

    return WW3df