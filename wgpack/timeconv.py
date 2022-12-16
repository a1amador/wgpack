# time conversion module
import datetime
import time
import numpy as np
import pandas as pd

def epoch2datetime64(tep):
    '''
    This function converts Unix time (also known as Epoch time, POSIX time, seconds since the Epoch, or UNIX Epoch time)
    into numpy datetime64.
    references: https://stackoverflow.com/questions/15053791/numpy-datetime64-from-unix-utc-seconds
    :param tep (numpy ndarray): Unix time (UTC)
    :return (numpy ndarray):    datetime64 format
    '''
    t64 = np.empty(len(tep), dtype='datetime64[s]')
    cc  = 0
    for tt in tep:
        t64[cc] = np.datetime64(datetime.datetime.utcfromtimestamp(tt))
        cc += 1
    return t64

def getUnixTimestamp(humanTime,dateFormat):
    """
    Convert from human-format to UNIX timestamp.
    :param humanTime:
    :param dateFormat:
    :return:
    """
    unixTimestamp = int(time.mktime(datetime.datetime.strptime(humanTime, dateFormat).timetuple()))
    return unixTimestamp

def timeIndexToDatetime(baseTime, times, units='hours'):
    '''
    Function to turn time index into timestamp
    :param baseTime:
    :param times:
    :param units:
    :return:
    '''
    newTimes = []
    if units=='hours':
        for ts in times:
            newTimes.append(baseTime + datetime.timedelta(hours=ts))
    elif units=='days':
        for ts in times:
            newTimes.append(baseTime + datetime.timedelta(days=ts))
    return newTimes

def mtime2datetime64(mtime):
    '''
    This function converts MATLAB datenum into datetime64
    :param mtime: list or array of MATLAB datenum values
    :return: DatetimeIndex array
    '''
    tt=[]
    for mt in mtime:
        tt.append(datetime.datetime.fromordinal(int(mt))
                       + datetime.timedelta(days=mt%1)
                       - datetime.timedelta(days = 366))
    # convert to pandas datetime64
    tt = pd.to_datetime(tt)
    return tt

def datetime2matlabdn(dt):
    # This function converts Python datetime to Matlab datenum
    # References: https://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum
    import datetime
    mdn = dt + datetime.timedelta(days = 366)
    frac_seconds = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds
