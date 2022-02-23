import sys,os
import datetime
import wget
import numpy as np
import pandas as pd
import netCDF4 as netcdf

module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
datadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/forecasts')
# datadir  = '/data/seachest/cordc-data/PROJECTS/ARCTERX2022/forecasts'

ta = datetime.datetime.utcnow().strftime('%Y-%m-%dT')+'00%3A00%3A00Z&'
tb = (datetime.datetime.utcnow() + pd.Timedelta(days=5)).strftime('%Y-%m-%dT')+'00%3A00%3A00Z&'


# ----------------------------------------------------------------------------------------------------------------------
# LOAD ROMS data
# https://www.pacioos.hawaii.edu/currents/model-mariana/
# ----------------------------------------------------------------------------------------------------------------------

# url
data_url = 'https://pae-paha.pacioos.hawaii.edu/thredds/ncss/roms_marig/ROMS_Guam_Regional_Ocean_Model_best.ncd?'+\
           'var=zeta&var=salt&var=temp&var=u&var=v&'+\
           'north=15.9753'+'&'+\
           'west=142.9187'+'&'+\
           'east=146.9722'+'&'+\
           'south=11.9373'+'&'+\
           'disableLLSubset=on&disableProjSubset=on&horizStride=1&'+\
           'time_start='+ta+'&'+\
           'time_end='+tb+'&'+\
           'timeStride=1&vertCoord='

# filename
filename = 'ROMS_Guam_' + datetime.datetime.utcnow().strftime('%Y%m%d') + '.ncd'
dfnam    = os.path.join(datadir,filename)

# Download data?
dwld_flg=True
# if file exist, remove it directly
if dwld_flg:
    if os.path.exists(dfnam):
        os.remove(dfnam)
    print('Beginning file download with wget module...')
    wget.download(data_url, out=dfnam)
    print('Downloaded ' + filename)

# Read-in data
# ROMS_data = netcdf.Dataset(dfnam)


# ----------------------------------------------------------------------------------------------------------------------
# Load WRF data
# https://www.pacioos.hawaii.edu/weather/model-wind-mariana/
# ----------------------------------------------------------------------------------------------------------------------

# url
data_url = 'https://pae-paha.pacioos.hawaii.edu/thredds/ncss/wrf_guam/WRF_Guam_Regional_Atmospheric_Model_best.ncd?'+\
           'var=Pair&var=Qair&var=Tair&var=Uwind&var=Vwind&var=lwrad_down&var=rain&var=swrad&'+\
           'north=15.9785'+'&'+\
           'west=142.9017'+'&'+\
           'east=147.0724'+'&'+\
           'south=11.9318'+'&'+\
           'disableLLSubset=on&disableProjSubset=on&horizStride=1&'+\
           'time_start='+ta+'&'+\
           'time_end='+tb+'&'+\
           'timeStride=1'

# filename
filename = 'WRF_Guam_' + datetime.datetime.utcnow().strftime('%Y%m%d') + '.ncd'
dfnam    = os.path.join(datadir,filename)

# Download data?
dwld_flg=True
# if file exist, remove it directly
if dwld_flg:
    if os.path.exists(dfnam):
        os.remove(dfnam)
    print('Beginning file download with wget module...')
    wget.download(data_url, out=dfnam)
    print('Downloaded ' + filename)

# Read-in data
# WRF_data = netcdf.Dataset(dfnam)


# ----------------------------------------------------------------------------------------------------------------------
# Load WW3 data
# https://www.pacioos.hawaii.edu/waves/model-mariana/
# ----------------------------------------------------------------------------------------------------------------------

# url
data_url = 'https://pae-paha.pacioos.hawaii.edu/thredds/ncss/ww3_mariana/WaveWatch_III_Mariana_Regional_Wave_Model_best.ncd?'+\
           'var=Tdir&var=Thgt&var=Tper&var=sdir&var=shgt&var=sper&var=wdir&var=whgt&var=wper&'+\
           'disableLLSubset=on&disableProjSubset=on&horizStride=1&'+\
           'time_start='+ta+'&'+\
           'time_end='+tb+'&'+\
           'timeStride=1&vertCoord='

# filename
filename = 'WaveWatch_III_Mariana_' + datetime.datetime.utcnow().strftime('%Y%m%d') + '.ncd'
dfnam    = os.path.join(datadir,filename)

# Download data?
dwld_flg=True
# if file exist, remove it directly
if dwld_flg:
    if os.path.exists(dfnam):
        os.remove(dfnam)
    print('Beginning file download with wget module...')
    wget.download(data_url, out=dfnam)
    print('Downloaded ' + filename)

# Read-in data
# WW3_data = netcdf.Dataset(dfnam)

