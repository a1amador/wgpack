import sys,os
import datetime
import wget
import subprocess
import numpy as np
import pandas as pd
import netCDF4 as netcdf

module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
datadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/forecasts/PacIOOS')

ta = datetime.datetime.utcnow().strftime('%Y-%m-%dT')+'00%3A00%3A00Z&'
tb = (datetime.datetime.utcnow() + pd.Timedelta(days=5)).strftime('%Y-%m-%dT')+'00%3A00%3A00Z&'


# ----------------------------------------------------------------------------------------------------------------------
# LOAD ROMS data
# https://www.pacioos.hawaii.edu/currents/model-mariana/
# PacIOOS’ Regional Ocean Modeling System (ROMS) provides a 6-day, 3-hourly forecast for the region surrounding the
# Mariana Islands, including Guam and the main islands of the Commonwealth of the Northern Mariana Islands (CNMI),
# at approximately 2-km (1.2-mile) resolution. The forecast is run daily and gets updated on this website around
# 9:30 AM Chamorro Standard Time (UTC+10) every morning. ROMS (http://myroms.org) is an open source,
# community-supported model widely adopted by the scientific community for a diverse range of applications. Model runs
# are configured and produced for the Mariana region by Dr. Brian Powell and lab within the Department of Oceanography
# in the School of Ocean and Earth Science and Technology (SOEST) at the University of Hawaiʻi at Mānoa.
# ----------------------------------------------------------------------------------------------------------------------

# url
# data_url = 'https://pae-paha.pacioos.hawaii.edu/thredds/ncss/roms_marig/ROMS_Guam_Regional_Ocean_Model_best.ncd?'+\
#            'var=zeta&var=salt&var=temp&var=u&var=v&'+\
#            'north=15.9753'+'&'+\
#            'west=142.9187'+'&'+\
#            'east=146.9722'+'&'+\
#            'south=11.9373'+'&'+\
#            'disableLLSubset=on&disableProjSubset=on&horizStride=1&'+\
#            'time_start='+ta+'&'+\
#            'time_end='+tb+'&'+\
#            'timeStride=1&vertCoord='
data_url = 'https://pae-paha.pacioos.hawaii.edu/thredds/ncss/roms_marig/ROMS_Guam_Regional_Ocean_Model_best.ncd?'+\
           'var=zeta&var=salt&var=temp&var=u&var=v&'+\
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
# PacIOOS’ Weather Research and Forecasting (WRF) model provides a 6-day, hourly forecast for the region surrounding
# the Mariana Islands, including Guam and the main islands of the Commonwealth of the Northern Mariana Islands (CNMI),
# at approximately 3-km (1.9-mile) resolution. The forecast is run daily and gets updated on this website between
# 9:30 AM and 10:30 AM Chamorro Standard Time (UTC+10) every morning. WRF is an open source numerical weather
# prediction system widely adopted by the scientific community for atmospheric research and operational forecasting.
# Shown here is the Advanced Research WRF (ARW) dynamical solver developed and maintained by the Mesoscale and
# Microscale Meteorology (MMM) Laboratory of the National Center for Atmospheric Research (NCAR). Model runs are
# configured and produced for the Mariana region by Dr. Yi-Leng Chen and lab within the Department of Atmospheric
# Sciences in the School of Ocean and Earth Science and Technology (SOEST) at the University of Hawaiʻi at Mānoa.
# ----------------------------------------------------------------------------------------------------------------------

# url
# data_url = 'https://pae-paha.pacioos.hawaii.edu/thredds/ncss/wrf_guam/WRF_Guam_Regional_Atmospheric_Model_best.ncd?'+\
#            'var=Pair&var=Qair&var=Tair&var=Uwind&var=Vwind&var=lwrad_down&var=rain&var=swrad&'+\
#            'north=15.9785'+'&'+\
#            'west=142.9017'+'&'+\
#            'east=147.0724'+'&'+\
#            'south=11.9318'+'&'+\
#            'disableLLSubset=on&disableProjSubset=on&horizStride=1&'+\
#            'time_start='+ta+'&'+\
#            'time_end='+tb+'&'+\
#            'timeStride=1'
data_url = 'https://pae-paha.pacioos.hawaii.edu/thredds/ncss/wrf_guam/WRF_Guam_Regional_Atmospheric_Model_best.ncd?'+\
           'var=Pair&var=Qair&var=Tair&var=Uwind&var=Vwind&var=lwrad_down&var=rain&var=swrad&'+\
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
# PacIOOS’ WaveWatch III (WW3) model is a 5-day, hourly forecast for the region surrounding the Mariana Islands,
# including Guam and the main islands of the Commonwealth of the Northern Mariana Islands (CNMI), at approximately
# 0.05-degree (~5-km) resolution. The forecast is run daily and gets updated on this website around 10:00 PM Chamorro
# Standard Time (UTC+10) each day. This regional wave model helps capture island effects such as island shadowing,
# refraction, and accurate modeling of local wind waves. WW3 (https://polar.ncep.noaa.gov/waves/wavewatch/) is an open source wave model developed by the National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Prediction (NCEP). Model runs are configured and produced for the Mariana region by Dr. Kwok Fai Cheung and lab within the Department of Ocean and Resources Engineering (ORE) in the School of Ocean and Earth Science and Technology (SOEST) at the University of Hawaiʻi at Mānoa. Boundary conditions are provided by the global WaveWatch III model (ww3_global) at approximately 50-km resolution.
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

# ----------------------------------------------------------------------------------------------------------------------
# LOAD high-res ROMS (Harper's Model)
# ----------------------------------------------------------------------------------------------------------------------
# data url and output filename
yearday_str = datetime.datetime.utcnow().strftime('%j')
filename = 'guam_his_00' + yearday_str + '.nc'
data_url = os.path.join('https://vertmix.alaska.edu/ARCTERX/SA_2022/ROMS/GUAMDinner_1km_2022_03_21_UH_UH',filename)
datadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/forecasts/Harper/')
dfnam    = os.path.join(datadir,filename)

# Download data? - TODO: need to fix this!
dwld_flg=False
# if file exist, remove it directly
if dwld_flg:
    if os.path.exists(dfnam):
        os.remove(dfnam)
    print('Beginning file download with wget subprocess...')
    ping = 'wget --directory-prefix=' + datadir + ' --no-check-cn ertificate https://vertmix.alaska.edu/ARCTERX/SA_2022/ROMS/GUAMDinner_1km_2022_03_21_UH_UH/guam_his_00083.nc'
    subprocess.check_output(['bash', '-c', ping])
    print('Downloaded ' + filename)

