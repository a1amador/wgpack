import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# --------------------------------------------------------
# module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
from wgpack.dportal import veh_list, readDP_CTD, OxHz2DO

# Data service path
ds_folder = os.path.join(str(Path.home()),'src/lri-wgms-data-service')
if ds_folder not in sys.path:
    sys.path.insert(0, ds_folder)
from DataService import DataService


# --------------------------------------------------------
args = sys.argv
# Select vehicle
vnam    = args[1]           # vehicle name
vid     = veh_list[vnam]    # vehicle id (Data Portal)
tw      = args[2]           # time window to display specified in days (ends at present time)
tw = None if tw == 'None' else tw
winch_flg = float(args[3])  # CTD winch? (e.g., '0', '1')
prj     = args[4]           # project folder name (e.g., 'calcofi', 'tfo', etc.)
# vnam,channels,tw,prj =  'sv3-251','C0158P2JJTT','None','westpac'
# tw = None if tw == 'None' else tw
# vid = veh_list[vnam]

print('vehicle:',vnam)
print('project:',prj)

# --------------------------------------------------------
# Set start date according to the time window to be considered
now = datetime.utcnow()
if tw is not None:
    # use specified time window
    tst = now - pd.Timedelta(days=float(tw))
else:
    # use last 7 days
    # tst = now - pd.Timedelta(days=7)
    # use prescribed splash date
    tst = datetime(2022, 3, 9, 8, 0, 0, 0)

# Read in CTD output from Data Portal
try:
    ctdDPdf = readDP_CTD(vid=vid, start_date=tst.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    flg_plt = True
    # Convert oxygenHz to dissolved oxygen
    # Measured values
    F = ctdDPdf['oxygenHz'].values
    T = ctdDPdf['temperature'].values
    P = ctdDPdf['pressure'].values
    S = ctdDPdf['salinity'].values
    dFdt = np.gradient(F, ctdDPdf['time'].values)  # Time derivative of SBE 43 output oxygen signal (volts/second)
    OxSol = ctdDPdf['oxygenSolubility'].values  # Oxygen saturation value after Garcia and Gordon (1992)
    DOdict = OxHz2DO(F, dFdt, T, P, S, OxSol, vnam)
    # Store in dataframe
    ctdDPdf['DO (ml/L)'] = DOdict['DOmlL']
    ctdDPdf['DO (muM/kg)'] = DOdict['DOmuMkg']
except RuntimeError as err:
    print(err)
    print("Something went wrong when retrieving CTD data from LRI's Data Portal")
    flg_plt = False

