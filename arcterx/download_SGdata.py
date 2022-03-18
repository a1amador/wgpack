import sys,os
import requests
from urllib.parse import urlparse

module_path = os.path.join(os.path.abspath(os.path.join('..')),'wgpack')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from wgpack.config import seachest_data_dir

args = sys.argv
vnam = args[1]  # vehicle name (e.g., sg526)


# ----------------------------------------------------------------------------------------------------------------------
# Download Sea Glider data
# ----------------------------------------------------------------------------------------------------------------------
# Sea Glider data urls
SGurl_lst = ['https://iop.apl.washington.edu/seaglider/sg526/current/data/sg526_ARCTERX_1.0m_up_and_down_profile.nc',
             'https://iop.apl.washington.edu/seaglider/sg526/current/data/sg526_ARCTERX_timeseries.nc',
             'https://iop.apl.washington.edu/seaglider/sg526/current/data/sg526.kmz']
user = 'guest'
password = 'IOPdata!'
# download directory (Seachest)
SGdatadir  = os.path.join(os.path.dirname(seachest_data_dir),'ARCTERX2022',vnam)

for SGurl in SGurl_lst:
    # output directory and filename
    filename = os.path.basename(urlparse(SGurl).path)
    SGfnam    = os.path.join(SGdatadir,filename)
    # get data
    resp = requests.get(SGurl, auth=(user, password))
    # save data
    if resp.status_code == 200:
        with open(SGfnam, 'wb') as out:
            for bits in resp.iter_content():
                out.write(bits)
        print('Downloaded' + filename)