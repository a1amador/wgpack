# helper functions module
import numpy as np

# Find nearest value in numpy array
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def BinDat2mod(x, y, Xctr, Xwid, nth=1, med=0, ciflg=None):
    '''
    Returns averages from binned data y as f(x) with bin
    centers at Xctr and bin widths at Xwid
    nth: min # of points for a valid return
    yb: bin-averaged value
    sx: standard dev
    ny: # of points in average
    ci: hi/lo boundaries (95% CI)
        no confidence intervals needed (speeds up code)
    med: if 1 use median otherwise use mean
    GP 2010, modified by AA 2019
    Adapted to python by AA 2020
    '''
    assert (np.logical_or(np.shape(y)[0] == len(x), np.shape(y)[1] == len(x)))

    # number of bootstrap samples
    nboot = 1000
    maxlen = nboot * 2
    # find indices for all data that fits within bin
    yb, sx, ny, ci = [], [], [], []
    for X, Xw in zip(Xctr, Xwid):
        # ii = np.logical_and(x<X+Xw/2, x>X-Xw/2, not np.isnan(y).all())
        ii = np.logical_and(x < X + Xw / 2, x > X - Xw / 2)
        if sum(ii) >= nth:
            if med:
                yb.append(np.nanmedian(y[:, ii]))
            else:
                yb.append(np.nanmean(y[:, ii]))
        else:
            yb.append(np.nan)
            sx.append(np.nan)
            ci.append(np.nan)
        ny.append(sum(ii))
        sx.append(np.nanstd(y[:, ii]))
    # TODO: CI
    return yb, sx, ny


def moving_average_3pt(x, y):
    '''
    This function computes a 3pt weighted smoother [.25 .5 .25] centered about x.
    :param x: independent variable
    :param y: dependent variable
    :return: 3pt moving average of y
    '''
    bin_centers = x
    bin_avg = np.zeros(len(bin_centers))
    for index in range(1, len(bin_centers) - 1):
        # We're going to weight with a 3pt weighted smoother [.25 .5 .25]
        w = np.zeros(len(bin_centers))
        w[index] = 0.5
        w[index - 1] = .25
        w[index + 1] = .25
        bin_avg[index] = np.average(y, weights=w)
    return bin_avg


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def nan_interpolate(x, n):
    '''
    This function linearly interpolates over NaNs, if the number of contiguous NaNs is less than n.
    :param x: data array
    :param n: maximum NaN gap over which interpolation is permitted.
    :return: de-NaN'ed array
    '''
    import pandas as pd
    return pd.Series(x).interpolate(method='linear', limit=n).values


