# Physical Oceanography module
# 3rd party imports
import numpy as np
from scipy.interpolate import interp1d
# local imports
from .nav import get_WEAthet
from .helperfun import nan_interpolate


def compute_Dmbar(f, Dm, E):
    '''
    Given the frequency dependent wave direction, this function computes the mean (energy weighted) wave direction.
    :param f: frequency array
    :param Dm: frequency dependent wave direction array
    :param E: wave energy spectra
    :return: mean (energy weighted) wave direction
    '''
    if len(np.shape(Dm)) == 2:
        # use this for time-series
        Dmbar = []
        for d, e in zip(Dm, E):
            # ignore nans
            ii = ~np.isnan(d) * ~np.isnan(e)
            dx = d[ii]
            ex = e[ii]
            fx = f[ii]
            # compute energy-weighted mean direction
            sin_bar = np.trapz(ex * np.sin(np.deg2rad(dx)), x=fx) / np.trapz(ex, x=fx)
            cos_bar = np.trapz(ex * np.cos(np.deg2rad(dx)), x=fx) / np.trapz(ex, x=fx)
            thet_bar = (np.rad2deg(np.arctan2(sin_bar, cos_bar)) + 360) % 360
            Dmbar.append(thet_bar)
    else:
        # use this for single instance
        # ignore nans
        ii = ~np.isnan(Dm) * ~np.isnan(E)
        dx = Dm[ii]
        ex = E[ii]
        fx = f[ii]
        # compute energy-weighted mean direction
        sin_bar = np.trapz(ex * np.sin(np.deg2rad(dx)), x=fx) / np.trapz(ex, x=fx)
        cos_bar = np.trapz(ex * np.cos(np.deg2rad(dx)), x=fx) / np.trapz(ex, x=fx)
        Dmbar = (np.rad2deg(np.arctan2(sin_bar, cos_bar)) + 360) % 360

    return np.array(Dmbar)

def compute_Dp(f,Dm,E):
    '''
    Given the frequency dependent wave direction (Dm), this function computes the peak wave direction (Dp).
    :param f: frequency array
    :param Dm: frequency dependent wave direction array
    :param E: wave energy spectra
    :return: mean (energy weighted) wave direction
    '''
    Dp=[]
    for d,e in zip(Dm,E):
        # find peak energy frequency band from infinity norm of normalized spectra
        e_norm = e/np.trapz(e,f)
        idx = (np.abs(e-np.linalg.norm(e_norm, ord=np.inf))).argmin()
        Dp.append(d[idx])
    return np.array(Dp)

def compute_Hs(f,E,fc=0.04):
    '''
    Given the wave energy spectra, this function computes the significant wave height.
    :param f: frequency array
    :param E: wave energy spectra
    :param fc: low-frequency cutoff
    :return: significant wave height (Hs)
    '''
    Hs=[]
    for e in E:
        # apply low frequency cutoff
        e[f < fc]=0
        # compute significant wave height
        Hs.append(4*np.sqrt(np.trapz(e,f)))
    Hs = np.array(Hs)
    return Hs

def compute_Tp(f,E):
    '''
    Given the wave energy spectra, this function computes the peak period.
    :param f: frequency array
    :param E: wave energy spectra
    :return: peak period (Tp)
    '''
    if len(E.shape) == 1:
        e = E
        # find peak period from infinity norm of normalized spectra
        e_norm = e / np.trapz(e, f)
        idx = (np.abs(e - np.linalg.norm(e_norm, ord=np.inf))).argmin()
        Tp = 1 / f[idx]
    elif len(E.shape)==2:
        Tp=[]
        for e in E:
            # find peak period from infinity norm of normalized spectra
            e_norm = e/np.trapz(e,f)
            idx = (np.abs(e-np.linalg.norm(e_norm, ord=np.inf))).argmin()
            Tp.append(1/f[idx])
        Tp = np.array(Tp)
    else:
        print("error")
    return Tp

def compute_Ta(f,E):
    '''
    Given the wave energy spectra, this function computes the average (energy weighted) period.
    :param f: frequency array
    :param E: wave energy spectra
    :return: average period (Ta)
    '''
    if len(E.shape)==1:
        e=E
        # not-nan boolean
        inot_nan = np.logical_and(~np.isnan(f),~np.isnan(e))
        # calculate average period
        Ta = np.trapz(e[inot_nan],x=f[inot_nan])/np.trapz(f[inot_nan]*e[inot_nan],x=f[inot_nan])
    elif len(E.shape)==2:
        Ta=[]
        for e in E:
            # not-nan boolean
            inot_nan = np.logical_and(~np.isnan(f),~np.isnan(e))
            # calculate average period
            Ta_tmp = np.trapz(e[inot_nan],x=f[inot_nan])/np.trapz(f[inot_nan]*e[inot_nan],x=f[inot_nan])
            Ta.append(Ta_tmp)
        Ta = np.array(Ta)
    else:
        print("error")
    return Ta

def cdnlp(sp, z):
    '''
    This function computes neutral drag coefficient and wind speed at 10m given
    the wind speed at height z following Large and Pond (1981), J. Phys. Oceanog., 11, 324-336.
    Converted from matlab to python (https://github.com/sea-mat/air-sea/blob/master/cdnlp.m)
    :param sp (array): wind speed  [m/s]
    :param z (float): measurement height [m]
    :return: cd - neutral drag coefficient at 10m
             u10 - wind speed at 10m  [m/s]
    '''
    # define physical constants
    kappa = 0.4  # von Karman's constant
    a = np.log(z / 10) / kappa  # log-layer correction factor
    tol = .001  # tolerance for iteration [m/s]

    u10o = np.zeros_like(sp)
    cd = 1.15e-3 * np.ones_like(sp)
    u10 = sp / (1 + a * np.sqrt(cd))

    ii = abs(u10 - u10o) > tol
    while any(ii):
        u10o = u10
        cd = 4.9e-4 + 6.5e-5 * u10o  # compute cd(u10)
        cd[u10o < 10.15385] = 1.15e-3
        u10 = sp / (1 + a * np.sqrt(cd))  # next iteration
        ii = abs(u10 - u10o) > tol  # keep going until iteration converges
    return cd, u10


def Doppler_Ecorrect(fin, fout, E, Dm, cog, sog):
    '''
    % This function applies Doppler corrections tp wave frequency spectra
% following the work of Collins III et al. (2017), and Amador et al. (2022)
% Inputs:
%       fin:    input vector containing the frequency bins [Hz]
%       fout:   output frequencies (energy values will be interpolated
%               onto this vector) [Hz]
%       E:      omni-directional wave energy spectrum [m^2/Hz]
%       Dm:     mean wave directions as a function of frequency
%       cog:    vehicle course over ground [deg relative to true north]
%       sog:    vehicle speed over ground [m/s]
% Outputs:
%       E0i: 	Doppler-corrected omni-directional wave spectrum
% created by: Andre Amador
% date: 10/12/2022
    :param fin:
    :param fout:
    :param E:
    :param Dm:
    :param cog:
    :param sog:
    :return:
    '''
    import numpy as np
    # gravity constant
    g = 9.8  # m/s^2
    # calculate wave encounter angle based on mwb mean direction and convert to radians
    thetr = np.deg2rad(get_WEAthet(cog, Dm))
    # Calculate Doppler-corrected frequency
    f0_tmp = (g - np.sqrt(g ** 2 - 4 * (2 * np.pi * sog * np.cos(thetr)) * (g * fin))) / (
                2 * (2 * np.pi * sog * np.cos(thetr)))
    # Doppler correction changes the frequency resolution, so a Jacobian is required to conserve spectral energy
    J = np.gradient(fin) / np.gradient(f0_tmp)
    # Remove outliers (TODO: there's probably a better way to deal with this issue)
    J[J > 5] = np.nan
    J[J < 0] = np.nan
    E0_tmp = E * J
    # sort frequency array
    ii_sort = np.argsort(f0_tmp)
    f0_tmp = f0_tmp[ii_sort]
    E0_tmp = E0_tmp[ii_sort]
    # interpolate over nans
    n = 5
    E0_tmp = nan_interpolate(E0_tmp, n)
    # Interpolation function
    fi_E = interp1d(f0_tmp, E0_tmp, fill_value="extrapolate")
    # interpolate onto fout frequency array
    E0i = fi_E(fout)
    # extrapolated values are padded with zeros (or nans)
    E0i[fout < fin[0]] = 0
    E0i[fout > fin[-1]] = 0
    return E0i
