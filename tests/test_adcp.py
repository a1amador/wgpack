import sys,os
import h5py
from wgpack.config import DATAPATH,RDRPATH
from wgpack.adcp import readADCP_raw,motion_correct_ADCP_gps_h5py

# file path and name for input .PD0 file
fnam = "20210515_020002UTC_continuous_20210516.PD0"
adcp_filepath_in = os.path.join(DATAPATH,fnam)
# file path and name for output .mat file
fnam_out, file_extension = os.path.splitext(adcp_filepath_in)

# read-in ADCP binaries
adcpr = readADCP_raw(adcp_filepath_in, RDRPATH, adcp_filepath_out=fnam_out, eng_exit=True)

# read-in processed adcp data
hf = h5py.File(fnam_out+'.mat', 'r')
adcpr = hf['adcpr']

# apply motion correction
dt_gps = 120   # Time-averaging interval for GPS-derived velocities (s)
dt_avg = 120*3 # Time-averaging interval for motion-corrected ADCP velocities (s)
adcpm = motion_correct_ADCP_gps_h5py(adcpr, dt_gps, dt_avg)
# adcpm.keys()

# ----------------------------------------------------------------------------------------------------------------------
# Plot results
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
import numpy as np

# create masked colormap
cm_msk = plt.get_cmap('seismic').copy()
cm_msk.set_bad(color='lightgray')

tt_WG = adcpm['time']
ranges_WG = adcpm['ranges']
# motion-corrected velocities
U_WG = adcpm['Evel']
V_WG = adcpm['Nvel']
# motion-corrected and filtered velocities
Uf_WG = adcpm['Evelf']
Vf_WG = adcpm['Nvelf']

# Plot WG water velocities
yL = [4,80]
uL = 0.5
vL = 0.5
fntsz = 16
labsz = 14
ylab_str = 'depth [m]'

# Do not show vehicle heading
heights = [1,1]
nrows = 2
ncols = 2
gs_kw = dict(height_ratios=heights)
fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                            sharex=True,
                            sharey=False,
                            gridspec_kw=gs_kw,
                            figsize=(24, 11))


# WG: East Vel
x,y = np.meshgrid(tt_WG,ranges_WG)
c = ax[0,0].pcolormesh(x,y,U_WG,vmin=-uL, vmax=uL,cmap = cm_msk)
ax[0,0].set_ylim(yL)
ax[0,0].set_ylabel(ylab_str,fontsize=fntsz)
ax[0,0].set_title('WG: East Velocities',fontsize=fntsz)
ax[0,0].invert_yaxis()

# WG: North Vel
x,y = np.meshgrid(tt_WG,ranges_WG)
ax[0,1].pcolormesh(x,y,V_WG,vmin=-vL, vmax=vL, cmap = cm_msk)
ax[0,1].set_ylim(yL)
ax[0,1].set_title('WG: North Velocities',fontsize=fntsz)
ax[0,1].invert_yaxis()

# WG: East Vel (filtered)
x,y = np.meshgrid(tt_WG,ranges_WG)
c = ax[1,0].pcolormesh(x,y,Uf_WG,vmin=-uL, vmax=uL,cmap = cm_msk)
ax[1,0].set_ylim(yL)
ax[1,0].set_ylabel(ylab_str,fontsize=fntsz)
ax[1,0].set_title('WG: East Velocities (filtered)',fontsize=fntsz)
ax[1,0].invert_yaxis()

# WG: North Vel (filtered)
x,y = np.meshgrid(tt_WG,ranges_WG)
ax[1,1].pcolormesh(x,y,Vf_WG,vmin=-vL, vmax=vL, cmap = cm_msk)
ax[1,1].set_ylim(yL)
ax[1,1].set_title('WG: North Velocities (filtered)',fontsize=fntsz)
ax[1,1].invert_yaxis()

# rotate and align the tick labels so they look better
ax[0,0].tick_params(labelsize=labsz)
ax[1,0].tick_params(labelsize=labsz)
ax[0,1].tick_params(labelsize=labsz)
ax[1,1].tick_params(labelsize=labsz)

# Define the date format
date_form = DateFormatter("%m-%d-%y %H")
ax[1,1].xaxis.set_major_formatter(date_form)
fig.autofmt_xdate()

# add a colorobar
gs = gridspec.GridSpec(ncols=3, nrows=nrows, height_ratios=gs_kw['height_ratios'], right=0.95,figure=fig)
axc = fig.add_subplot(gs[0,-1])
axc.set_visible(False)
cbar = fig.colorbar(c, ax=axc,orientation='vertical')
cax = cbar.ax
# Add label on top of colorbar.
cbar.ax.set_xlabel("[m/s]\n",fontsize=labsz)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=labsz)

fig.show()
