'''
This script downloads drifter data to Seachest. Drifter data are obtained from
https://glidervm3.ceoas.oregonstate.edu/ARCTERX/Sync/Shore/Drifter/
'''

import sys,os
import pandas as pd

module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir
datadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022/drifters_leg1')

# ----------------------------------------------------------------------------------------------------------------------
# Read in leg 1 drifter data
# ----------------------------------------------------------------------------------------------------------------------
# Create an URL object
url = 'https://glidervm3.ceoas.oregonstate.edu/ARCTERX/Sync/Shore/Drifter/drifter.csv'
driftdf = pd.read_csv(url)
# set time stamp column as datetimeindex
Slocdf = driftdf.set_index(pd.DatetimeIndex(driftdf[' Timestamp(UTC)'].values))
Slocdf.sort_index(inplace=True)
# drop unnecessary columns
Slocdf.drop(columns=[' </br>'],inplace=True)

# ----------------------------------------------------------------------------------------------------------------------
# Save as csv file
# ----------------------------------------------------------------------------------------------------------------------
fnamout = os.path.join(datadir,url.split('/')[-1])
Slocdf.to_csv(fnamout)