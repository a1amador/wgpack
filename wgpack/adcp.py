# Acoustic Doppler Current Profiler (ADCP) module
import os
from .helperfun import movingaverage,nan_interpolate

def readADCP_raw(adcp_filepath_in, rdrpath, adcp_filepath_out=False, eng_exit=True):
    # To install matlab use the following commands:
    # 1) cd /Applications/MATLAB_R2020b.app/extern/engines/python
    # 2) python setup.py install --prefix "/Users/a1amador/opt/miniconda3/envs/wgpack"
    # More generically:
    # 1) cd <matlabroot>/extern/engines/python
    # 2) python setup.py install --prefix "/your_path_to_anaconda3/envs/your_env"
    # References:
    # https://www.mathworks.com/matlabcentral/answers/346068-how-do-i-properly-install-matlab-engine-using-the-anaconda-package-manager-for-python
    # https://stackoverflow.com/questions/44823720/starting-matlab-engine-in-anaconda-virtual-environment-returns-segmentation-fau
    # ----------------------------------------------------------------------------
    # begin function
    import matlab.engine
    # sart-up Matlab
    eng = matlab.engine.start_matlab()
    # add path to rdradcp.m
    eng.addpath(rdrpath,nargout=0)
    # Read (raw binary) RDI ADCP files
    print('processing '+ os.path.basename(adcp_filepath_in))
    # adcpr = eng.rdradcp(adcp_filepath_in)
    adcpr = eng.wrap_rdradcp_v0(adcp_filepath_in,adcp_filepath_out)
    print('Pre-processing complete')
    if adcp_filepath_out:
        print('saved '+ os.path.basename(adcp_filepath_out))
    # exit matlab engine
    if eng_exit:
        eng.exit()
    return adcpr


def motion_correct_ADCP_gps_h5py(adcpr, dt_gps, dt_avg, mag_dec=None, qc_flg=False,dtc=None):
    '''
    This function corrects ADCP velocities for Wave Glider motion using GPS-derived velocities.
    Reads-in output from rdrdadcp.m (Matlab).
    :param adcpr: output file from rdradcp.m (.mat file read using h5py)
    :param dt_gps: Time-averaging interval for GPS-derived velocities (s)
    :param dt_avg: Time-averaging interval for motion-corrected ADCP velocities (s)
    :param dtc: time offset correction as numpy.timedelta64
    :return: dictionary containing motion-corrected ADCP velocities and auxiliary variables
    '''
    import datetime
    import pyproj
    import numpy as np
    import pandas as pd
    # Collect variables:
    # time
    mtime = np.array(adcpr['nav_mtime'][:-1]).flatten()  #
    # convert matlab datenum to datetime
    nav_time = []
    for mt in mtime:
        nav_time.append(datetime.datetime.fromordinal(int(mt))
                        + datetime.timedelta(days=mt % 1)
                        - datetime.timedelta(days=366))
    # convert to pandas datetime
    nav_time = pd.to_datetime(nav_time)
    # correct time offset
    if dtc is None:
        pass
    else:
        # correct time offset
        nav_time = nav_time+dtc

    kk = len(nav_time)
    # nav variables
    pitch = np.array(adcpr['pitch'][:kk]).flatten()  #
    roll = np.array(adcpr['roll'][:kk]).flatten()  #
    heading = np.array(adcpr['heading'][:kk]).flatten()  #
    nav_elongitude = np.array(adcpr['nav_elongitude'][:kk]).flatten()  #
    nav_elatitude = np.array(adcpr['nav_elatitude'][:kk]).flatten()  #
    # Bottom tracking
    bt_range = np.array(adcpr['bt_range'])[:kk, :].T  #
    bt_range_mean = np.mean(bt_range, axis=0)
    # Doppler velocities (beam)
    b1_vel = np.array(adcpr['east_vel'])[:kk, :].T  #
    b2_vel = np.array(adcpr['north_vel'])[:kk, :].T  #
    b3_vel = np.array(adcpr['vert_vel'])[:kk, :].T  #
    b4_vel = np.array(adcpr['error_vel'])[:kk, :].T  #
    # QC variables
    perc_good = np.array(adcpr['perc_good'])[:kk, :, :].T  #
    corr = np.array(adcpr['corr'])[:kk, :, :].T  #
    intens = np.array(adcpr['intens'])[:kk, :, :].T  #
    # raw echo intensities
    b1_intens = intens[:, 0, :]
    b2_intens = intens[:, 1, :]
    b3_intens = intens[:, 2, :]
    b4_intens = intens[:, 3, :]
    # config variables
    ranges = np.array(adcpr['config']['ranges']).flatten()
    beam_angle = float(adcpr['config']['beam_angle'][:])  #
    EA = float(adcpr['config']['xducer_misalign'][:])  #
    EB = float(adcpr['config']['magnetic_var'][:])  #
    xmit_pulse = float(adcpr['config']['xmit_pulse'][:])  #
    xmit_lag = float(adcpr['config']['xmit_lag'][:])  #
    # Magnetic declination correction (necessary when EB is configured incorrectly)
    if mag_dec is None:
        pass
    else:
        heading = (heading - EB + mag_dec) % 360

    # ------------------------------------------------------------
    # Process GPS data
    # ------------------------------------------------------------
    geodesic = pyproj.Geod(ellps='WGS84')
    # ping to ping dt
    dt_p2p = np.array(np.median(np.diff(nav_time)), dtype='timedelta64[s]').item().total_seconds()
    # number of points (full-step)
    wz_gps = int(dt_gps / dt_p2p)
    # number of points (half-step)
    nn = int(wz_gps / 2)
    # calculate GPS based velocities
    sog_gps, sog_gpse, sog_gpsn, cog_gps, pitch_mean, roll_mean = [], [], [], [], [], []
    for i, t in enumerate(nav_time[nn:-nn]):
        # apply central differencing scheme
        ii = i + nn
        # dt in seconds
        # dt = np.array(nav_time[ii+nn]-nav_time[ii-nn], dtype='timedelta64[s]').item().total_seconds()
        # this method is probably faster
        dt = (nav_time[ii + nn] - nav_time[ii - nn]).value / 1E9
        azfwd, azback, dx = geodesic.inv(nav_elongitude[ii - nn], nav_elatitude[ii - nn],
                                         nav_elongitude[ii + nn], nav_elatitude[ii + nn])
        sog = dx / dt
        # store values
        cog_gps.append((azfwd + 360) % 360)
        sog_gps.append(sog)
        sog_gpse.append(sog * np.sin(np.deg2rad(azfwd)))
        sog_gpsn.append(sog * np.cos(np.deg2rad(azfwd)))
        pitch_mean.append(np.mean(pitch[ii - nn:ii + nn]))
        roll_mean.append(np.mean(roll[ii - nn:ii + nn]))
    # concatenate nans
    app_nan = np.zeros(nn) + np.nan
    cog_gps = np.concatenate((app_nan, np.array(cog_gps), app_nan))
    sog_gps = np.concatenate((app_nan, np.array(sog_gps), app_nan))
    sog_gpse = np.concatenate((app_nan, np.array(sog_gpse), app_nan))
    sog_gpsn = np.concatenate((app_nan, np.array(sog_gpsn), app_nan))
    # ------------------------------------------------------------
    # low-pass filter raw ADCP heading
    huf = movingaverage(np.sin(np.deg2rad(heading)), window_size=wz_gps)
    hvf = movingaverage(np.cos(np.deg2rad(heading)), window_size=wz_gps)
    headingf = np.rad2deg(np.arctan2(huf, hvf))
    # compute float heading (corrected for heading offset (EA)
    headingf_float = (headingf - EA) % 360
    heading_float = (heading - EA) % 360

    # ------------------------------------------------------------
    # Q/C ADCP velocities
    # ------------------------------------------------------------
    # Note that percent-good (perc_good < 100) already masks velocity data with nans
    if qc_flg:
        pass

    # ------------------------------------------------------------
    # Process ADCP velocities
    # ------------------------------------------------------------
    # Beam to Instrument
    # Constants
    c = 1
    theta = np.deg2rad(beam_angle)  # beam angle
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    d = a / np.sqrt(2)
    # instrument velocities
    x_vel = c * a * (b1_vel - b2_vel)
    y_vel = c * a * (b4_vel - b3_vel)
    z_vel = b * (b1_vel + b2_vel + b3_vel + b4_vel)
    err_vel = d * (b1_vel + b2_vel - b3_vel - b4_vel)
    # ------------------------------------------------------------
    # Instrument to Ship
    h = -EA
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    w = (-cp * sr) * x_vel + (sp) * y_vel + (cp * cr) * z_vel
    # TODO: calculate and output velocities in vehicle reference frame
    # ------------------------------------------------------------
    # Instrument to Earth
    h = heading
    Tilt1 = np.deg2rad(pitch)
    Tilt2 = np.deg2rad(roll)
    P = np.arctan(np.tan(Tilt1) * np.cos(Tilt2))
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    # cp = np.cos(P)
    # sp = np.sin(P)
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    # Correct ADCP velocities (gps-derived velocities)
    Evel = u + sog_gpse
    Nvel = v + sog_gpsn
    # ------------------------------------------------------------
    # Low-pass filter ADCP velocities
    wz_avg = int(dt_avg / dt_p2p)  # averaging window for ADCP velocities
    nanlim = 5  # maximum NaN gap over which interpolation is permitted
    Nvelf, Evelf = [], []
    for nv, ev in zip(Nvel, Evel):
        # interpolate over nans
        nv = nan_interpolate(nv, nanlim)
        ev = nan_interpolate(ev, nanlim)
        # moving average
        Nvelf.append(movingaverage(nv, window_size=wz_avg))
        Evelf.append(movingaverage(ev, window_size=wz_avg))
    Nvelf = np.array(Nvelf)
    Evelf = np.array(Evelf)

    # Create output dictionary for motion-corrected ADCP data
    adcpmdict = {
        'time': nav_time,
        'longitude': nav_elongitude,
        'latitude': nav_elatitude,
        'ranges': ranges,
        'Evel': Evel,
        'Nvel': Nvel,
        'err_vel': err_vel,
        'Evelf': Evelf,
        'Nvelf': Nvelf,
        'cog_gps': cog_gps,
        'sog_gps': sog_gps,
        'sog_gpse': sog_gpse,
        'sog_gpsn': sog_gpsn,
        'heading_float': heading_float,
        'headingf_float': headingf_float
    }
    return adcpmdict


def Doppler_vel_ADCP_h5py(adcpr, mag_dec=None, qc_flg=False):
    '''
    This function transforms Wave Glider ADCP beam velocities into an Earth reaference frame.
    :param adcpr: output file from rdradcp.m (.mat file read using h5py)
    :param mag_dec:
    :param qc_flg:
    :return: dictionary containing ADCP Doppler velocities and auxiliary variables
    '''
    import datetime
    import pyproj
    import numpy as np
    import pandas as pd
    # Collect variables:
    # time
    mtime = np.array(adcpr['nav_mtime'][:-1]).flatten()  #
    # convert matlab datenum to datetime
    nav_time = []
    for mt in mtime:
        nav_time.append(datetime.datetime.fromordinal(int(mt))
                        + datetime.timedelta(days=mt % 1)
                        - datetime.timedelta(days=366))
    # convert to pandas datetime
    nav_time = pd.to_datetime(nav_time)
    kk = len(nav_time)
    # nav variables
    pitch = np.array(adcpr['pitch'][:kk]).flatten()  #
    roll = np.array(adcpr['roll'][:kk]).flatten()  #
    heading = np.array(adcpr['heading'][:kk]).flatten()  #
    nav_elongitude = np.array(adcpr['nav_elongitude'][:kk]).flatten()  #
    nav_elatitude = np.array(adcpr['nav_elatitude'][:kk]).flatten()  #
    # Bottom tracking
    bt_range = np.array(adcpr['bt_range'])[:kk, :].T  #
    bt_range_mean = np.mean(bt_range, axis=0)
    # Doppler velocities (beam)
    b1_vel = np.array(adcpr['east_vel'])[:kk, :].T  #
    b2_vel = np.array(adcpr['north_vel'])[:kk, :].T  #
    b3_vel = np.array(adcpr['vert_vel'])[:kk, :].T  #
    b4_vel = np.array(adcpr['error_vel'])[:kk, :].T  #
    # QC variables
    perc_good = np.array(adcpr['perc_good'])[:kk, :, :].T  #
    corr = np.array(adcpr['corr'])[:kk, :, :].T  #
    intens = np.array(adcpr['intens'])[:kk, :, :].T  #
    # config variables
    ranges = np.array(adcpr['config']['ranges']).flatten()
    beam_angle = float(adcpr['config']['beam_angle'][:])  #
    EA = float(adcpr['config']['xducer_misalign'][:])  #
    EB = float(adcpr['config']['magnetic_var'][:])  #
    xmit_pulse = float(adcpr['config']['xmit_pulse'][:])  #
    xmit_lag = float(adcpr['config']['xmit_lag'][:])  #
    # Magnetic declination correction (necessary when EB is configured incorrectly)
    if mag_dec is None:
        pass
    else:
        heading = (heading - EB + mag_dec) % 360

    # ------------------------------------------------------------
    # Q/C ADCP velocities
    # ------------------------------------------------------------
    # Note that percent-good (perc_good < 100) already masks velocity data with nans
    if qc_flg:
        # Along-beam velocities are rejected for instrument tilts greater than 20 deg
        tilt_THRESH = 20
        # pitch
        b1_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        b2_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        b3_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        b4_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        # roll
        b1_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        b2_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        b3_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        b4_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        # Along-beam velocities are rejected for along beam correlations below 90 counts
        corr_THRESH = 80
        b1_vel[corr[:, 0, :] < corr_THRESH] = np.nan
        b2_vel[corr[:, 1, :] < corr_THRESH] = np.nan
        b3_vel[corr[:, 2, :] < corr_THRESH] = np.nan
        b4_vel[corr[:, 3, :] < corr_THRESH] = np.nan
        # Along-beam velocities are rejected for along beam correlations below 90 counts
        intens_THRESH = 50
        b1_vel[intens[:, 0, :] < intens_THRESH] = np.nan
        b2_vel[intens[:, 1, :] < intens_THRESH] = np.nan
        b3_vel[intens[:, 2, :] < intens_THRESH] = np.nan
        b4_vel[intens[:, 3, :] < intens_THRESH] = np.nan

    # ------------------------------------------------------------
    # Process ADCP velocities
    # ------------------------------------------------------------
    # Beam to Instrument
    # Constants
    c = 1
    theta = np.deg2rad(beam_angle)  # beam angle
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    d = a / np.sqrt(2)
    # instrument velocities
    x_vel = c * a * (b1_vel - b2_vel)
    y_vel = c * a * (b4_vel - b3_vel)
    z_vel = b * (b1_vel + b2_vel + b3_vel + b4_vel)
    err_vel = d * (b1_vel + b2_vel - b3_vel - b4_vel)
    # ------------------------------------------------------------
    # Instrument to Ship
    h = -EA
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    w = (-cp * sr) * x_vel + (sp) * y_vel + (cp * cr) * z_vel
    # TODO: calculate and output velocities in vehicle reference frame
    # ------------------------------------------------------------
    # Instrument to Earth
    h = heading
    Tilt1 = np.deg2rad(pitch)
    Tilt2 = np.deg2rad(roll)
    P = np.arctan(np.tan(Tilt1) * np.cos(Tilt2))
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    # cp = np.cos(P)
    # sp = np.sin(P)
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel

    # Create output dictionary for motion-corrected ADCP data
    adcpmdict = {
        'time': nav_time,
        'longitude': nav_elongitude,
        'latitude': nav_elatitude,
        'ranges': ranges,
        'Evel': u,
        'Nvel': v,
        'vert_vel': w,
        'err_vel': err_vel,
    }
    return adcpmdict


def motion_correct_ADCP_gps(adcpr, dt_gps, mag_dec=None, qc_flg=False,dtc=None):
    '''
    This function corrects ADCP velocities for Wave Glider motion using GPS-derived velocities.
    Reads-in output from rdrdadcp.py (python)
    :param adcpr: output structure from rdradcp.py
    :param dt_gps: Time-averaging interval for GPS-derived velocities (s)
    :param dt_avg: Time-averaging interval for motion-corrected ADCP velocities (s)
    :param dtc: time offset correction as numpy.timedelta64
    :return: dictionary containing motion-corrected ADCP velocities and auxiliary variables
    References:
    https://github.com/rustychris/stompy/blob/master/stompy/io/rdradcp.py
    Other resources:
    https://seachest.ucsd.edu/cordc/analysis/rdradcpy
    https://pypi.org/project/ADCPy/
    '''
    import datetime
    import numpy as np
    import pandas as pd
    from geopy.distance import distance
    # local imports
    from .nav import get_bearing
    # Collect variables:
    # time
    mtime = adcpr.nav_mtime
    # convert matlab datenum to datetime
    nav_time = []
    for mt in mtime:
        nav_time.append(datetime.datetime.fromordinal(int(mt))
                        + datetime.timedelta(days=mt % 1)
                        - datetime.timedelta(days=366))

    # convert to pandas datetime
    nav_time = pd.to_datetime(nav_time)
    # correct time offset
    if dtc is None:
        pass
    else:
        # correct time offset
        nav_time = nav_time+dtc

    # nav variables
    pitch = adcpr.pitch
    roll = adcpr.roll
    heading = adcpr.heading
    nav_elongitude = adcpr.nav_elongitude
    nav_elatitude = adcpr.nav_elatitude
    # Bottom tracking
    bt_range = adcpr.bt_range
    bt_range_mean = np.mean(bt_range, axis=1)
    # Doppler velocities (beam)
    b1_vel = adcpr.east_vel
    b2_vel = adcpr.north_vel
    b3_vel = adcpr.vert_vel
    b4_vel = adcpr.error_vel
    # QC variables
    perc_good = adcpr.perc_good
    corr = adcpr.corr
    intens = adcpr.intens
    # raw echo intensities
    b1_intens = intens[:, :, 0]
    b2_intens = intens[:, :, 1]
    b3_intens = intens[:, :, 2]
    b4_intens = intens[:, :, 3]
    # Temperature
    temperature = adcpr.temperature
    # config variables
    ranges = adcpr.config.ranges
    beam_angle = adcpr.config.beam_angle
    EA = adcpr.config.xducer_misalign
    EB = adcpr.config.magnetic_var
    xmit_pulse = adcpr.config.xmit_pulse
    xmit_lag = adcpr.config.xmit_lag
    # Magnetic declination correction (necessary when EB is configured incorrectly)
    if mag_dec is None:
        pass
    else:
        heading = (heading - EB + mag_dec) % 360

    # ------------------------------------------------------------
    # Process GPS data
    # ------------------------------------------------------------
    # ping to ping dt
    dt_p2p = np.array(np.median(np.diff(nav_time)), dtype='timedelta64[s]').item().total_seconds()
    # number of points (full-step)
    wz_gps = int(dt_gps / dt_p2p)
    # number of points (half-step)
    nn = int(wz_gps / 2)
    # calculate GPS based velocities
    sog_gps, sog_gpse, sog_gpsn, cog_gps, pitch_mean, roll_mean = [], [], [], [], [], []
    for i, t in enumerate(nav_time[nn:-nn]):
        # apply central differencing scheme
        ii = i + nn
        # dt in seconds
        # dt = np.array(nav_time[ii+nn]-nav_time[ii-nn], dtype='timedelta64[s]').item().total_seconds()
        # this method is probably faster
        dt = (nav_time[ii + nn] - nav_time[ii - nn]).value / 1E9
        # Calculate cog and sog from WG mwb coordinates
        p1 = (nav_elatitude[ii - nn], nav_elongitude[ii - nn])
        p2 = (nav_elatitude[ii + nn], nav_elongitude[ii + nn])
        sog = distance(p1,p2).m/dt
        cog = get_bearing(p1,p2)
        # store values
        cog_gps.append(cog)
        sog_gps.append(sog)
        sog_gpse.append(sog * np.sin(np.deg2rad(cog)))
        sog_gpsn.append(sog * np.cos(np.deg2rad(cog)))
        pitch_mean.append(np.mean(pitch[ii - nn:ii + nn]))
        roll_mean.append(np.mean(roll[ii - nn:ii + nn]))

    # concatenate nans
    app_nan = np.zeros(nn) + np.nan
    cog_gps = np.concatenate((app_nan, np.array(cog_gps), app_nan))
    sog_gps = np.concatenate((app_nan, np.array(sog_gps), app_nan))
    sog_gpse = np.concatenate((app_nan, np.array(sog_gpse), app_nan))
    sog_gpsn = np.concatenate((app_nan, np.array(sog_gpsn), app_nan))
    # # ------------------------------------------------------------
    # # low-pass filter raw ADCP heading
    # huf = movingaverage(np.sin(np.deg2rad(heading)), window_size=wz_gps)
    # hvf = movingaverage(np.cos(np.deg2rad(heading)), window_size=wz_gps)
    # headingf = np.rad2deg(np.arctan2(huf, hvf))
    # compute float heading (corrected for heading offset (EA)
    heading_float = (heading - EA) % 360

    # ------------------------------------------------------------
    # Q/C ADCP velocities
    # ------------------------------------------------------------
    # Note that percent-good (perc_good < 100) already masks velocity data with nans
    if qc_flg:
        pass

    # ------------------------------------------------------------
    # Process ADCP velocities
    # ------------------------------------------------------------
    # Beam to Instrument
    # Constants
    c = 1
    theta = np.deg2rad(beam_angle)  # beam angle
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    d = a / np.sqrt(2)
    # instrument velocities
    x_vel = (c * a * (b1_vel - b2_vel)).T
    y_vel = (c * a * (b4_vel - b3_vel)).T
    z_vel = (b * (b1_vel + b2_vel + b3_vel + b4_vel)).T
    err_vel = (d * (b1_vel + b2_vel - b3_vel - b4_vel)).T
    # ------------------------------------------------------------
    # Instrument to Ship
    h = -EA
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    w = (-cp * sr) * x_vel + (sp) * y_vel + (cp * cr) * z_vel
    # TODO: calculate and output velocities in vehicle reference frame
    # ------------------------------------------------------------
    # Instrument to Earth
    h = heading
    Tilt1 = np.deg2rad(pitch)
    Tilt2 = np.deg2rad(roll)
    P = np.arctan(np.tan(Tilt1) * np.cos(Tilt2))
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    # cp = np.cos(P)
    # sp = np.sin(P)
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    # Correct ADCP velocities (gps-derived velocities)
    Evel = u + sog_gpse
    Nvel = v + sog_gpsn
    # ------------------------------------------------------------
    # Create output dictionary for motion-corrected ADCP data
    adcpmdict = {
        'time': nav_time,
        'longitude': nav_elongitude,
        'latitude': nav_elatitude,
        'ranges': ranges,
        'Evel': Evel,
        'Nvel': Nvel,
        'err_vel': err_vel,
        'cog_gps': cog_gps,
        'sog_gps': sog_gps,
        'sog_gpse': sog_gpse,
        'sog_gpsn': sog_gpsn,
        'heading_float': heading_float,
        'temperature': temperature,
        }
    return adcpmdict


def Doppler_vel_ADCP(adcpr, mag_dec=None, qc_flg=False, dtc=None):
    '''
    This function transforms Wave Glider ADCP beam velocities into an Earth reaference frame (no motion compensation).
    :param adcpr: output file from rdradcp.m (.mat file read using h5py)
    :param mag_dec:
    :param qc_flg:
    :return: dictionary containing ADCP Doppler velocities and auxiliary variables
    '''
    import datetime
    import numpy as np
    import pandas as pd
    from geopy.distance import distance
    # local imports
    # from .nav import get_bearing
    # Collect variables:
    # time
    mtime = adcpr.nav_mtime
    # convert matlab datenum to datetime
    nav_time = []
    for mt in mtime:
        nav_time.append(datetime.datetime.fromordinal(int(mt))
                        + datetime.timedelta(days=mt % 1)
                        - datetime.timedelta(days=366))

    # convert to pandas datetime
    nav_time = pd.to_datetime(nav_time)
    # correct time offset
    if dtc is None:
        pass
    else:
        # correct time offset
        nav_time = nav_time+dtc

    # nav variables
    pitch = adcpr.pitch
    roll = adcpr.roll
    heading = adcpr.heading
    nav_elongitude = adcpr.nav_elongitude
    nav_elatitude = adcpr.nav_elatitude
    # Bottom tracking
    bt_range = adcpr.bt_range
    bt_range_mean = np.mean(bt_range, axis=1)
    # Doppler velocities (beam)
    b1_vel = adcpr.east_vel
    b2_vel = adcpr.north_vel
    b3_vel = adcpr.vert_vel
    b4_vel = adcpr.error_vel
    # QC variables
    perc_good = adcpr.perc_good
    corr = adcpr.corr
    intens = adcpr.intens
    # raw echo intensities
    b1_intens = intens[:, :, 0]
    b2_intens = intens[:, :, 1]
    b3_intens = intens[:, :, 2]
    b4_intens = intens[:, :, 3]
    # Temperature
    temperature = adcpr.temperature
    # config variables
    ranges = adcpr.config.ranges
    beam_angle = adcpr.config.beam_angle
    EA = adcpr.config.xducer_misalign
    EB = adcpr.config.magnetic_var
    xmit_pulse = adcpr.config.xmit_pulse
    xmit_lag = adcpr.config.xmit_lag
    # Magnetic declination correction (necessary when EB is configured incorrectly)
    if mag_dec is None:
        pass
    else:
        heading = (heading - EB + mag_dec) % 360


    # ------------------------------------------------------------
    # Q/C ADCP velocities
    # ------------------------------------------------------------
    # Note that percent-good (perc_good < 100) already masks velocity data with nans
    if qc_flg:
        # Along-beam velocities are rejected for instrument tilts greater than 20 deg
        tilt_THRESH = 20
        # pitch
        b1_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        b2_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        b3_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        b4_vel[:, np.abs(pitch > tilt_THRESH)] = np.nan
        # roll
        b1_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        b2_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        b3_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        b4_vel[:, np.abs(roll > tilt_THRESH)] = np.nan
        # Along-beam velocities are rejected for along beam correlations below 90 counts
        corr_THRESH = 80
        b1_vel[corr[:, 0, :] < corr_THRESH] = np.nan
        b2_vel[corr[:, 1, :] < corr_THRESH] = np.nan
        b3_vel[corr[:, 2, :] < corr_THRESH] = np.nan
        b4_vel[corr[:, 3, :] < corr_THRESH] = np.nan
        # Along-beam velocities are rejected for along beam correlations below 90 counts
        intens_THRESH = 50
        b1_vel[intens[:, 0, :] < intens_THRESH] = np.nan
        b2_vel[intens[:, 1, :] < intens_THRESH] = np.nan
        b3_vel[intens[:, 2, :] < intens_THRESH] = np.nan
        b4_vel[intens[:, 3, :] < intens_THRESH] = np.nan

    # ------------------------------------------------------------
    # Process ADCP velocities
    # ------------------------------------------------------------
    # Beam to Instrument
    # Constants
    c = 1
    theta = np.deg2rad(beam_angle)  # beam angle
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    d = a / np.sqrt(2)
    # instrument velocities
    x_vel = (c * a * (b1_vel - b2_vel)).T
    y_vel = (c * a * (b4_vel - b3_vel)).T
    z_vel = (b * (b1_vel + b2_vel + b3_vel + b4_vel)).T
    err_vel = (d * (b1_vel + b2_vel - b3_vel - b4_vel)).T
    # ------------------------------------------------------------
    # Instrument to Ship
    h = -EA
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel
    w = (-cp * sr) * x_vel + (sp) * y_vel + (cp * cr) * z_vel
    # TODO: calculate and output velocities in vehicle reference frame
    # ------------------------------------------------------------
    # Instrument to Earth
    h = heading
    Tilt1 = np.deg2rad(pitch)
    Tilt2 = np.deg2rad(roll)
    P = np.arctan(np.tan(Tilt1) * np.cos(Tilt2))
    ch = np.cos(np.deg2rad(h))
    sh = np.sin(np.deg2rad(h))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    # cp = np.cos(P)
    # sp = np.sin(P)
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    # From Teledyne ADCP Coordinate Transformation, Formulas and Calculations
    u = (ch * cr + sh * sp * sr) * x_vel + (sh * cp) * y_vel + (ch * sr - sh * sp * cr) * z_vel
    v = (-sh * cr + ch * sp * sr) * x_vel + (ch * cp) * y_vel + (-sh * sr - ch * sp * cr) * z_vel

    # Create output dictionary for motion-corrected ADCP data
    adcpmdict = {
        'time': nav_time,
        'longitude': nav_elongitude,
        'latitude': nav_elatitude,
        'ranges': ranges,
        'Evel': u,
        'Nvel': v,
        'vert_vel': w,
        'err_vel': err_vel,
    }
    return adcpmdict