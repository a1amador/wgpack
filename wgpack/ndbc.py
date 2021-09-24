# NDBC module
import pandas as pd
from geopy.distance import distance

def get_ndbc_latest():
    """
    This function finds and returns the most recent observation (provided that the observation is less than two hours
    old) from all stations hosted on the NDBC web site. It also returns the position information (latitude and
    longitude) for each station.
    references: https://www.ndbc.noaa.gov/docs/ndbc_web_data_guide.pdf
    :return (dataframe): dataframe containing the most recent observations
    """

    # The latest observation file is available at:
    link = 'https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt'
    # Read-in data
    Ldf = pd.read_csv(link,delim_whitespace=True,skiprows=0,header='infer',
			parse_dates=[3,4,5,6,7], index_col=0)
    # drop the second row
    Ldf.drop(['#text'], inplace=True)
    return Ldf

def get_ndbc_nearest(coord,Ldf,N=1,type=None):
    """
    This function finds the nearest N number of NDBC station(s) for a given coordinate.
    references: https://www.ndbc.noaa.gov/docs/ndbc_web_data_guide.pdf
    :param coord (tup): (latitude,longitude)
    :param Ldf (dataframe): dataframe containing the most recent observations
    :param N (int): number of nearest stations to return
    :param type (str): 'waves' or 'mets'
    :return (tup): nearest NDBC station (station id, distance in meters)
    """
    dtup = []
    for bid, row in Ldf.iterrows():
        # calculate distance to each NDBC station
        lat, lon = float(row['LAT']), float(row['LON'])
        d = distance(coord, (lat, lon)).m
        # store if wave/mets data are available
        dtup.append((bid, d, row['WVHT']!='MM',row['WSPD'] != 'MM'))
    # sort according to distance / find nearest 100 NDBC stations
    sdtup = sorted(dtup, key=lambda l: l[1])[:100]
    if type=='waves':
        # output closest station containing wave data
        outtup = list(filter(lambda x: x[2]==True in x, sdtup))
        outtup = [x[:-2] for x in outtup[:N]]
    elif type=='mets':
        # output closest station containing mets data
        outtup = list(filter(lambda x: x[3]==True in x, sdtup))
        outtup = [x[:-2] for x in outtup[:N]]
    else:
        # output closest station containing wave data
        outtup = [x[:-2] for x in sdtup[:N]]

    return outtup

def find_nearest_ndbc(coord):
    """
    This function finds the nearest NDBC station for a given coordinate.
    references: https://www.ndbc.noaa.gov/docs/ndbc_web_data_guide.pdf
    :param coord (tup): (latitude,longitude)
    :return (tup): (station id, distance in meters)
    """
    # The latest observation file is available at:
    link = 'https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt'
    # Read-in data
    tdf = pd.read_csv(link,delim_whitespace=True,skiprows=0,header='infer',
			parse_dates=[3,4,5,6,7], index_col=0)
    # drop the second row
    tdf.drop(['#text'],inplace=True)
    # calculate distance to each NDBC station
    dtup = []
    for bid,lat,lon in zip(tdf.index.values,tdf['LAT'].values.astype(float),tdf['LON'].values.astype(float)):
        d = distance(coord, (lat, lon)).m
        dtup.append((bid, d))
    # find nearest NDBC station
    return sorted(dtup,key=lambda l:l[1])[0]
