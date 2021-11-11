# LRI Data Portal module
import numpy as np
import pandas as pd
import datetime
import json
import requests
from requests.auth import HTTPBasicAuth
from wgpack.creds import DPun,DPpw

# create dictionary for vehicle id's (Data Portal)
veh_list = {
    "sv3-125" : '1219280337',
    "sv3-251" : '704144535',
    "sv3-253" : '131608817',
    "magnus"  : '502354689',
    "sv3-1087": '1981322853',
    "sv3-1101": '773827499'
        }

def readDP_waves(vid,start_date,end_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")):
    '''
    This function reads in bulk wave parameters from the Wave Glider GPSWaves sensor.
    references: Data Portal User Guide, Liquid Robotics Inc.
                https://requests.readthedocs.io/en/master/user/authentication/
                https://stackoverflow.com/questions/39780403/python3-read-json-file-from-url
                https://stackoverflow.com/questions/21104592/json-to-pandas-dataframe
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
                https://stackoverflow.com/questions/35314887/set-a-pandas-datetime64-pandas-dataframe-column-as-a-datetimeindex-without-the-t
    :param vid (str):           The vehicle identification number
    :param start_date (str):    start time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
    :param end_date (str):      end time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
                                if end date is not specified, then default to current time
    :return (pandas dataframe): dataframe of bulk wave data with the following columns:
                                index: The time at which this measurement was made (UTC, datetime64[ns])
                                accessionTime: The time at which this measurement was added to the Shoreside Data
                                               Depository (UTC, datetime64[ns]).
                                latitude (decimal degrees)
                                longitude (decimal degrees)
                                time: Unix time (seconds)
                                Fs: Sampling frequency (Hz)
                                Samples: Number of samples used
                                Dp: Dominant wave direction (degrees)
                                Hs: Significant wave height
                                Tp: Peak wave period
                                Tav: Average wave period
    '''
    # TODO: make sure that date is in the correct format
    # assert (start_date >= 0), "start date must be in ISO 8601 format"
    # assert (end_date >= 0), "end date must be in ISO 8601 format"

    # Create url
    data_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
               "&format=json&kinds=Waves&vids=" + vid

    # read in data
    response = requests.get(url=data_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    obj = json.loads(response)
    # Note the JSON file begins with a Wave Glider metadata summary, available in obj[0][0]
    metadata = pd.json_normalize(obj[0][0])

    # create dataframe
    df = pd.json_normalize(obj[0][1])
    for ii in list(range(1, len(obj))):
        df = df.append(pd.json_normalize(obj[ii][0]), ignore_index=True)

    # convert to float
    df['latitude']      = df['latitude'].astype(float)
    df['longitude']     = df['longitude'].astype(float)
    df['time']          = df['time'].astype(float)/1000
    df['accessionTime'] = df['accessionTime'].astype(float)/1000
    df['Samples']       = df['Samples'].astype(float)
    df['Fs']            = df['Fs'].astype(float)
    df['Hs']            = df['Hs'].astype(float)
    df['Tav']           = df['Tav'].astype(float)
    df['Tp']            = df['Tp'].astype(float)
    df['Dp']            = df['Dirp'].astype(float)
    # set unix timestamp column as datetimeindex
    df = df.set_index(pd.DatetimeIndex(df['time'].values.astype('datetime64[s]')))
    df['accessionTime'] = df['accessionTime'].astype('datetime64[s]')

    # drop unnecessary columns
    df.drop(columns=['kind','vid','AveragedSpectra','SampleGaps','sensorType',
                     'undefined','wasEncrypted','Dirp'],inplace=True)
    return df

def readDP_weather(vid,start_date,end_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")):
    '''
    This function reads in weather parameters from the Wave Glider Airmar sensor.
    references:                 Data Portal User Guide, Liquid Robotics Inc.
    :param vid (str):           The vehicle identification number
    :param start_date (str):    start time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
    :param end_date (str):      end time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
                                if end date is not specified, then default to current time
    :return (pandas dataframe): dataframe of weather data with the following columns:
                                index: The time at which this measurement was made (UTC, datetime64[ns])
                                accessionTime: The time at which this measurement was added to the Shoreside Data
                                               Depository (UTC, datetime64[ns]).
                                latitude (decimal degrees)
                                longitude (decimal degrees)
                                time: Unix time (seconds)
                                nWindSamples: number of samples
                                pressure: Air Pressure (millibars)
                                temperature: Air Temperature (degrees C)
                                WindSpeed: average wind speed (knots)
                                stdDevWindSpeed: standard deviation of wind speed (knots)
                                maxWindSpeed: max wind speed (knots)
                                WindDirection: average wind direction (degT)
                                stdDevWindDir: standard deviation of wind direction (deg)
    '''
    # TODO: make sure that date is in the correct format
    # assert (start_date >= 0), "start date must be in ISO 8601 format"
    # assert (end_date >= 0), "end date must be in ISO 8601 format"

    # Create url
    data_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
               "&format=json&kinds=Weather&vids=" + vid

    # read in data
    response = requests.get(url=data_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    obj = json.loads(response)
    # Note the JSON file begins with a Wave Glider metadata summary, available in obj[0][0]
    metadata = pd.json_normalize(obj[0][0])

    # create dataframe
    df = pd.json_normalize(obj[0][1])
    for ii in list(range(1, len(obj))):
        df = df.append(pd.json_normalize(obj[ii][0]), ignore_index=True)

    # convert to float
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    df['time'] = df['time'].astype(float) / 1000
    df['accessionTime'] = df['accessionTime'].astype(float) / 1000
    df['nWindSamples'] = df['nWindSamples'].astype(float)
    df['pressure'] = df['pressure'].astype(float)
    df['temperature'] = df['temperature'].astype(float)
    df['WindSpeed'] = df['avgWindSpeed'].astype(float) #* kt2mps
    df['maxWindSpeed'] = df['maxWindSpeed'].astype(float) #* kt2mps
    df['stdDevWindSpeed'] = df['stdDevWindSpeed'].astype(float) #* kt2mps
    df['WindDirection'] = df['avgWindDirection'].astype(float)
    df['stdDevWindDir'] = df['stdDevWindDir'].astype(float)

    # set unix timestamp column as datetimeindex
    df = df.set_index(pd.DatetimeIndex(df['time'].values.astype('datetime64[s]')))
    df['accessionTime'] = df['accessionTime'].astype('datetime64[s]')

    # drop unnecessary columns
    df.drop(columns=['kind', 'vid', 'undefined', 'wasEncrypted',
                     'avgWindDirection', 'avgWindSpeed'], inplace=True)
    return df

def readDP_wavesEnergy(vid,start_date,end_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")):
    '''
    This function reads in wave energy spectra from the Wave Glider GPSwaves sensor.
    references:                 Data Portal User Guide, Liquid Robotics Inc.
    :param vid (str):           The vehicle identification number
    :param start_date (str):    start time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
    :param end_date (str):      end time for data (dates are ISO 8601 e.g., '2020-05-15T20:00:00.000Z')
                                if end date is not specified, then default to current time
    :return (pandas dataframe): dataframe of weather data with the following columns:
                                index: The time at which this measurement was made (UTC, datetime64[ns])
                                accessionTime: The time at which this measurement was added to the Shoreside Data
                                               Depository (UTC, datetime64[ns]).
                                latitude (decimal degrees)
                                longitude (decimal degrees)
                                time: Unix time (seconds)
                                energy (list): wave energy spectra (m^2/Hz)
                                frequency (list): spectral frequency (Hz)
    '''

    Evar = 'Waves-Energy'
    Fvar = 'Waves-Frequency'
    # Create url
    Edata_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
                "&format=json&kinds=" + Evar + "&vids=" + vid
    Fdata_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
                "&format=json&kinds=" + Fvar + "&vids=" + vid
    # read in data
    Eresponse = requests.get(url=Edata_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    Fresponse = requests.get(url=Fdata_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    Eobj = json.loads(Eresponse)
    Fobj = json.loads(Fresponse)
    # create dataframe
    Edf = pd.json_normalize(Eobj[0][1])
    Fdf = pd.json_normalize(Fobj[0][1])
    assert (len(Edf) == len(Fdf))

    # convert to float values
    Elst = [float(i) for i in Edf['spectrumData'][0]]
    Flst = [float(i) for i in Fdf['spectrumData'][0]]
    Edf['spectrumData'][0] = Elst
    Fdf['spectrumData'][0] = Flst

    # append dataframe
    if len(Eobj)==len(Fobj):
        for ii in list(range(1, len(Eobj))):
            Edf = Edf.append(pd.json_normalize(Eobj[ii][0]), ignore_index=True)
            Fdf = Fdf.append(pd.json_normalize(Fobj[ii][0]), ignore_index=True)
            # convert spectra to float values
            Elst = [float(i) for i in Edf['spectrumData'][ii]]
            Flst = [float(i) for i in Fdf['spectrumData'][ii]]
            Edf['spectrumData'][ii] = Elst
            Fdf['spectrumData'][ii] = Flst
    else:
        for ii in list(range(1, len(Eobj))):
            Edf = Edf.append(pd.json_normalize(Eobj[ii][0]), ignore_index=True)
            # convert spectra to float values
            Elst = [float(i) for i in Edf['spectrumData'][ii]]
            Edf['spectrumData'][ii] = Elst
            Fdf['spectrumData'][ii] = Flst

    # store in Waves-Energy dataframe
    Edf['energy'] = Edf['spectrumData']
    Edf['frequency'] = Fdf['spectrumData']
    # TODO: make freq. be the columns on dataframe

    # convert to float
    Edf['latitude'] = Edf['latitude'].astype(float)
    Edf['longitude'] = Edf['longitude'].astype(float)
    Edf['time'] = Edf['time'].astype(float) / 1000
    Edf['accessionTime'] = Edf['accessionTime'].astype(float) / 1000

    # set unix timestamp column as datetimeindex
    Edf = Edf.set_index(pd.DatetimeIndex(Edf['time'].values.astype('datetime64[s]')))
    Edf['accessionTime'] = Edf['accessionTime'].astype('datetime64[s]')

    # drop unnecessary columns
    Edf.drop(columns=['kind', 'vid', 'undefined', 'wasEncrypted', 'sensorType','spectrumData'], inplace=True)
    return Edf

def readDP_WG(vid,start_date,end_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")):
    '''
    This function returns Wave Glider state parameters obtained from Liquid Robotics Data Portal.
    references: Data Portal User Guide, Liquid Robotics Inc.
                https://requests.readthedocs.io/en/master/user/authentication/
                https://stackoverflow.com/questions/39780403/python3-read-json-file-from-url
                https://stackoverflow.com/questions/21104592/json-to-pandas-dataframe
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
                https://stackoverflow.com/questions/35314887/set-a-pandas-datetime64-pandas-dataframe-column-as-a-datetimeindex-without-the-t
    :param vid (str):           The vehicle identification number
    :param start_date (str):    start time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
    :param end_date (str):      end time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
                                if end date is not specified, then default to current time
    :return (pandas dataframe): dataframe with the WG state parameters.
    '''

    # TODO: make sure that date is in the correct format
    # assert (start_date >= 0), "start date must be in ISO 8601 format"
    # assert (end_date >= 0), "end date must be in ISO 8601 format"

    # Create url
    data_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
               "&format=json&kinds=Waveglider&vids=" + vid

    # read in data
    response = requests.get(url=data_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    obj = json.loads(response)
    # Note the JSON file begins with a Wave Glider metadata summary, available in obj[0][0].
    metadata = pd.json_normalize(obj[0][0])
    # {'kind': 'wgmeta',
    #  'vid': '131608817',
    #  'vehicleName': 'SV3-253',
    #  'avatar': 'http://10.20.30.53/icon/SV3-253.png',
    #  'ip': '10.20.33.134',
    #  'model': 'SV3',
    #  'imei': '300434062573500',
    #  'RV': '1.1.3',
    #  'build': 'Regulus-2017-09-28T01:02:25Z',
    #  'buildnumber': '32',
    #  'systembuild': 'true'}

    # create dataframe
    df = pd.json_normalize(obj[0][1])
    for ii in list(range(1, len(obj))):
        df = df.append(pd.json_normalize(obj[ii][0]), ignore_index=True)

    # convert to float
    df['latitude']      = df['latitude'].astype(float)
    df['longitude']     = df['longitude'].astype(float)
    df['time']          = df['time'].astype(float)/1000
    df['waterSpeed']    = df['waterSpeed'].astype(float) # in kt
    df['headingDesired']= df['headingDesired'].astype(float)
    df['headingSub']    = df['headingSub'].astype(float)
    df['distanceOverGround'] = df['distanceOverGround'].astype(float)
    df['targetWaypoint']= df['targetWaypoint']
    df['totalPower']    = df['totalPower'].astype(float)
    df['tempSub']       = df['tempSub'].astype(float)
    df['pressureSensorSub'] = df['pressureSensorSub'].astype(float) # in kPa
    # set unix timestamp column as datetimeindex
    df = df.set_index(pd.DatetimeIndex(df['time'].values.astype('datetime64[s]')))

    # drop unnecessary columns
    df.drop(columns=['kind','vid','accessionTime','currentBearing','currentSpeed',
                     'floatBatteryLowAlarm', 'floatLeakAlarm',
                     'floatPressureThresholdExceededAlarm', 'floatRebootAlarm',
                     'floatTempThresholdExceededAlarm', 'floatToSubCommsAlarm',
                     'gpsNotFunctioningAlarm', 'headingSkew',
                     'internalVehicleCommAlarm', 'overCurrentAlarm',
                     'payloadErrorConditionAlarm', 'speedToGoal',
                     'subLeakAlarm', 'subPressureThresholdAlarm', 'subRebootAlarm',
                     'subTempThresholdExceededAlarm', 'subToFloatCommsAlarm',
                     'umbilicalFaultAlarm','undefined', 'wasEncrypted'],inplace=True)
    return df

def readDP_WGalarms(vid,start_date,end_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")):
    '''
    This function returns Wave Glider alarms obtained from Liquid Robotics Data Portal.
    references: Data Portal User Guide, Liquid Robotics Inc.
                https://requests.readthedocs.io/en/master/user/authentication/
                https://stackoverflow.com/questions/39780403/python3-read-json-file-from-url
                https://stackoverflow.com/questions/21104592/json-to-pandas-dataframe
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
                https://stackoverflow.com/questions/35314887/set-a-pandas-datetime64-pandas-dataframe-column-as-a-datetimeindex-without-the-t
    :param vid (str):           The vehicle identification number
    :param start_date (str):    start time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
    :param end_date (str):      end time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
                                if end date is not specified, then default to current time
    :return (pandas dataframe): dataframe with the WG state parameters.
    '''

    # Create url
    data_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
               "&format=json&kinds=Waveglider&vids=" + vid

    # read in data
    response = requests.get(url=data_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    obj = json.loads(response)
    # Note the JSON file begins with a Wave Glider metadata summary, available in obj[0][0].
    metadata = pd.json_normalize(obj[0][0])
    # create dataframe
    df = pd.json_normalize(obj[0][1])
    for ii in list(range(1, len(obj))):
        df = df.append(pd.json_normalize(obj[ii][0]), ignore_index=True)

    # # convert to float
    latitude  = df['latitude'].astype(float)
    longitude = df['longitude'].astype(float)
    time      = df['time'].astype(float)/1000

    # drop unnecessary columns
    df.drop(columns=['kind','vid','accessionTime',
                     'latitude','longitude','time',
                     'currentBearing','currentSpeed',
                     'waterSpeed', 'headingDesired','headingSub',
                     'distanceOverGround', 'targetWaypoint',
                     'totalPower', 'speedToGoal','tempSub','headingSkew',
                     'pressureSensorSub', 'undefined', 'wasEncrypted'],inplace=True)

    # Replace 'true' and 'false' with boolean
    booleanDictionary = {'true': True, 'false': False}
    for column in df:
        df[column] = df[column].map(booleanDictionary)

    # put back latitude, longitude, and time
    df.insert (0, "latitude", latitude)
    df.insert (1, "longitude", longitude)
    df.insert (2, "time", time)

    # set unix timestamp column as datetimeindex
    df = df.set_index(pd.DatetimeIndex(df['time'].values.astype('datetime64[s]')))
    return df

def readDP_AMPSPowerStatusSummary(vid,start_date,end_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")):
    '''
    This function returns Wave Glider power status from Liquid Robotics Data Portal.
    references: Data Portal User Guide, Liquid Robotics Inc.
                https://requests.readthedocs.io/en/master/user/authentication/
                https://stackoverflow.com/questions/39780403/python3-read-json-file-from-url
                https://stackoverflow.com/questions/21104592/json-to-pandas-dataframe
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
                https://stackoverflow.com/questions/35314887/set-a-pandas-datetime64-pandas-dataframe-column-as-a-datetimeindex-without-the-t
    :param vid (str):           The vehicle identification number
    :param start_date (str):    start time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
    :param end_date (str):      end time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
                                if end date is not specified, then default to current time
    :return (pandas dataframe): dataframe with the WG state parameters.
    '''

    # Create url
    data_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
               "&format=json&kinds=AMPSPowerStatusSummary&vids=" + vid
    # read in data
    response = requests.get(url=data_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    obj = json.loads(response)

    # create dataframe
    df = pd.json_normalize(obj[0][1])
    for ii in list(range(1, len(obj))):
        df = df.append(pd.json_normalize(obj[ii][0]), ignore_index=True)

    df['latitude']      = df['latitude'].astype(float)
    df['longitude']     = df['longitude'].astype(float)
    df['time']          = df['time'].astype(float)/1000
    df['batteryChargingCurrent'] = df['batteryChargingCurrent'].astype(float)
    df['batteryChargingPower']   = df['batteryChargingPower'].astype(float)/1000
    df['outputPowerGenerated']   = df['outputPowerGenerated'].astype(float)/1000
    df['solarPowerGenerated']    = df['solarPowerGenerated'].astype(float)/1000
    df['totalBatteryPower']      = df['totalBatteryPower'].astype(float)/1000

    # set unix timestamp column as datetimeindex
    df = df.set_index(pd.DatetimeIndex(df['time'].values.astype('datetime64[s]')))

    # drop unnecessary columns
    df.drop(columns=['kind','vid','accessionTime','ports',
                     'reportVersion','undefined', 'wasEncrypted'],inplace=True)
    return df

def readDP_CTD(vid,start_date,end_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")):
    '''
    This function reads-in and returns Wave Glider CTD data from Liquid Robotics Data Portal.
    :param vid (str):           The vehicle identification number
    :param start_date (str):    start time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
    :param end_date (str):      end time for data (dates are ISO 8601 [https:// en.wikipedia.org/wiki/ISO_8601])
                                if end date is not specified, then default to current time
    :return (pandas dataframe): dataframe with the WG state parameters.
    '''
    # Create url
    data_url = "https://dataportal.liquidr.net/firehose/?start=" + start_date + "&end=" + end_date + \
               "&format=json&kinds=CTD&vids=" + vid
    # read in data
    response = requests.get(url=data_url, auth=HTTPBasicAuth(DPun, DPpw)).text
    obj = json.loads(response)
    # create dataframe
    df = pd.json_normalize(obj[0][1])
    for ii in list(range(1, len(obj))):
        df = df.append(pd.json_normalize(obj[ii][0]), ignore_index=True)
    # convert to float
    df['latitude']         = df['latitude'].astype(float)
    df['longitude']        = df['longitude'].astype(float)
    df['time']             = df['time'].astype(float)/1000
    df['conductivity']     = df['conductivity'].astype(float)
    df['dissolvedOxygen']  = df['dissolvedOxygen'].astype(float)
    df['oxygenHz']         = df['oxygenHz'].astype(float)
    df['oxygenSolubility'] = df['oxygenSolubility'].astype(float)
    df['pressure']         = df['pressure'].astype(float)
    df['salinity']         = df['salinity'].astype(float)
    df['temperature']      = df['temperature'].astype(float)
    # set unix timestamp column as datetimeindex
    df = df.set_index(pd.DatetimeIndex(df['time'].values.astype('datetime64[s]')))
    # drop unnecessary columns
    df.drop(columns=['kind','vid','accessionTime','sensorIdentifier',
                     'undefined','wasEncrypted'],inplace=True)
    return df

def OxHz2DO(F, dFdt, T, P, S, OxSol, vnam):
    '''
    This function converts oxygenHz (measured) to dissolved oxygen (DO) in ml/L and muM/kg.
    :param F:       ctdDPdf['oxygenHz'].values
    :param dFdt:    np.gradient(F, ctdDPdf['time'].values)
                    Time derivative of SBE 43 output oxygen signal (volts/second)
    :param T:       ctdDPdf['temperature'].values
    :param P:       ctdDPdf['pressure'].values
    :param S:       ctdDPdf['salinity'].values
    :param OxSol:   ctdDPdf['oxygenSolubility'].values
                    Oxygen saturation value after Garcia and Gordon (1992)
    :param vnam:    vehicle name e.g., 'magnus', 'sv3-253'
    :return:        output dictionary containing dissolved oxygen (DO) in ml/L and muM/kg

    References:
    05574 - USER GUIDE, SEA-BIRD ELECTRONICS GLIDER PAYLOAD CTD.pdf
    https://pythonhosted.org/seawater/eos80.html
    SBGPCTD command to display calibration coefficients on SMC
    ServerExec SBGPCTD "ctd --expert dc"
    '''
    import seawater as sw
    from seawater.library import T90conv
    # Calibration Coefficients (currently stored on the vehicle)
    if vnam == 'magnus':
        FOFFSET = -8.295800e+02  # Voltage at zero oxygen signal
        SOC = 3.117100e-04  # Oxygen signal slope
        # A, B, C: Residual temperature correction factors
        A = -5.250800e-03
        B = 2.206300e-04
        C = -2.873600e-06
        E = 3.600000e-02  # Pressure correction factor
        TAU20 = 1.020000e+00  # Sensor time constant tau (T,P) at 20 degC, 1 atmosphere, 0 PSU; slope term in calculation of tau(T,P)
        # D1, D2: Temperature and pressure correction factors in calculation of tau(T,P)
        D1 = 1.926340e-04
        D2 = -4.648030e-02
        # H1, H2, H3: Hysteresis correction factors
        H1, H2, H3 = -3.300000e-02, 5.000000e+03, 1.450000e+03
    elif vnam == 'sv3-1101':
        # SBE 43 S/N 3490 10-Oct-20
        FOFFSET = -8.247100e+02  # Voltage at zero oxygen signal
        SOC = 2.289400e-04   # Oxygen signal slope
        # A, B, C: Residual temperature correction factors
        A = -4.360500e-03
        B = 2.280200e-04
        C = -3.572600e-06
        E = 3.600000e-02  # Pressure correction factor
        TAU20 = 8.399999e-01  # Sensor time constant tau (T,P) at 20 degC, 1 atmosphere, 0 PSU; slope term in calculation of tau(T,P)
        # D1, D2: Temperature and pressure correction factors in calculation of tau(T,P)
        D1 = 1.926340e-04
        D2 = -4.648030e-02
        # H1, H2, H3: Hysteresis correction factors
        H1, H2, H3 = -3.300000e-02, 5.000000e+03, 1.450000e+03
    else:
        FOFFSET = 0  # Voltage at zero oxygen signal
        SOC = 0  # Oxygen signal slope
        # A, B, C: Residual temperature correction factors
        A = 0
        B = 0
        C = 0
        E = 0  # Pressure correction factor
        TAU20 = 0  # Sensor time constant tau (T,P) at 20 degC, 1 atmosphere, 0 PSU; slope term in calculation of tau(T,P)
        # D1, D2: Temperature and pressure correction factors in calculation of tau(T,P)
        D1 = 0
        D2 = 0
        # H1, H2, H3: Hysteresis correction factors
        H1, H2, H3 = 0, 0, 0

    # Calculated Values
    TAU = TAU20 * np.exp(D1 * P + D2 * (T - 20))  # sensor time constant at temperature and pressure
    K = T + 273.15

    # Calculate dissolved oxygen (ml/L)
    DOmlL = (SOC * (F + FOFFSET + TAU * dFdt)) * OxSol * (1.0 + A * T + B * T ** 2 + C * T ** 3) * np.exp(E * P / K)
    # Calculate dissolved oxygen (micro Mol/Kg)
    # [μmole/Kg] = [ml/L] * 44660 / (sigma_theta(P=0,Theta,S) + 1000)
    # Sigma_theta (potential density) is the density a parcel of water would have if it were raised adiabatically
    # to the surface without change in salinity.
    T90 = T90conv(T)
    sigma_theta = sw.pden(S, T90, P, pr=0)
    DOmuMkg = DOmlL * 44660 / (sigma_theta + 1000)

    # create output dictionary
    DOdict = {
        'DOmlL': DOmlL,
        'DOmuMkg': DOmuMkg
    }
    return DOdict