# Autoplot module
import datetime
import numpy as np
import pandas as pd
from geopy import distance
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

# knots to m/s
kt2mps = 0.514444

def waves_bulk_autoplot(mwbdf, gpsdf, Bdf_lst, bidstr_lst, figshow=False):
    '''
    This function creates timeseries and scatter plots of bulk waves parameters (Hs,Tp,Dp) for `plotWGwaves_vx.py`.
    :param mwbdf: dataframe with bulk waves parameters from CORDC Mini Wave Buoy sensor package
    :param gpsdf: dataframe with bulk waves parameters from LRI GPSwaves sensor package
    :param Bdf_lst: list of dataframes with bulk waves parameters from NDBC/CDIP
    :param bidstr_lst: list of strings with buoy id's of len(Bdf_lst)
    :param figshow: if True plt.show() is executed
    :return: figure object
    '''

    # configure plot parameters
    labsz = 8
    fntsz = 10
    now_utc = datetime.datetime.utcnow()
    if mwbdf.empty:
        xL = (gpsdf.index[0], now_utc)
    else:
        xL = (mwbdf.index[0], now_utc)
    # widths = [3, 1]
    widths = [4, 1]
    heights = [1, 1, 1]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, fig_axes = plt.subplots(3, 2, sharey='row', gridspec_kw=gs_kw,
                                 figsize=(15, 9))

    # Time-series
    ax1 = fig_axes[0][0]
    ax2 = fig_axes[1][0]
    ax3 = fig_axes[2][0]

    # Significant wave height (Hs)
    l1, = ax1.plot(mwbdf.index, mwbdf.Hs, '.b')
    l2, = ax1.plot(gpsdf.index, gpsdf.Hs, '.r')
    ax1.set_ylabel('Hs [m]', fontsize=fntsz)
    ax1.tick_params(labelsize=labsz)
    ax1.set_xlim(xL)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Peak wave period (Tp)
    ax2.plot(mwbdf.index, mwbdf.Tp, '.b')
    ax2.plot(gpsdf.index, gpsdf.Tp, '.r')
    ax2.set_ylabel('Tp [s]', fontsize=fntsz)
    ax2.tick_params(labelsize=labsz)
    ax2.set_xlim(xL)
    ax2.set_yticks(np.arange(4, 22, step=2))
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Peak wave direction (Dp)
    ax3.plot(mwbdf.index, mwbdf.Dp, '.b')
    ax3.plot(gpsdf.index, gpsdf.Dp, '.r')
    ax3.set_ylabel('Dp [deg]', fontsize=fntsz)
    ax3.set_ylim(0, 360)
    ax3.set_yticks(np.arange(0, 361, step=60))
    ax3.tick_params(labelsize=fntsz)
    ax3.set_xlabel('Time [UTC]', fontsize=fntsz)
    ax3.set_xlim(xL)

    leg_hlst = [l1, l2]
    leg_strlst = ['MWB', 'GPSWaves']
    if Bdf_lst:
        # Dynamically assign symbols/markers for each NDBC station
        sym_lst = ['.', '*', 'x', '^', 'v', '<', '>', '1', '2', '3', '4']
        bidstr_unique = list(set(bidstr_lst))
        bstr_chk = None
        cc = 0
        for Bdf, bstr in zip(Bdf_lst, bidstr_lst):
            # find a symbol/marker
            try:
                sym = sym_lst[bidstr_unique.index(bstr)]
            except:
                sym = '.'
            l3, = ax1.plot(Bdf.index, Bdf.WVHT, sym + 'k')
            ax2.plot(Bdf.index, Bdf.DPD, sym + 'k')
            ax3.plot(Bdf.index, Bdf.MWD, sym + 'k')
            # Update legend handle and string
            if bstr_chk != bstr:
                leg_hlst.append(l3)
                leg_strlst.append('NDBC ' + bstr)
                bstr_chk = bstr
                cc += 1
    # Legend
    ax1.legend(tuple(leg_hlst), tuple(leg_strlst), fontsize=fntsz, loc='best')

    # --------------------------------------------------------
    # Scatter plots
    ax4 = fig_axes[0][1]
    ax5 = fig_axes[1][1]
    ax6 = fig_axes[2][1]
    # --------------------------------------------------------
    # bin average MWB and GPSWaves data using CDIP time base
    # --------------------------------------------------------
    Nb_mwb = []
    Hsb_mwb = []
    Tpb_mwb = []
    Dpb_mwb = []
    cogb_mwb = []
    lonb_mwb = []
    latb_mwb = []
    d_mwb = []
    Nb_gps = []
    Hsb_gps = []
    Tpb_gps = []
    Dpb_gps = []
    if Bdf_lst:
        # Bdf30 = pd.DataFrame(columns=Bdf_lst[0].keys())
        Bdf30 = pd.DataFrame(columns=['WVHT', 'DPD', 'APD', 'MWD', 'LON', 'LAT'])
        for Bdf in Bdf_lst:
            # TODO: remove nans in wave height
            Bdf = Bdf[['WVHT', 'DPD', 'APD', 'MWD', 'LON', 'LAT']].dropna(thresh=4)
            # Resample NDBC/CDIP data into 30 min interval bins
            # Bdftpm = Bdf.resample('1H').mean()
            Bdftpm = Bdf.resample('30T').mean().dropna(how='all')
            dt = np.abs(np.median(np.diff(Bdftpm.index.values)))  # .astype('timedelta64[m]')
            for tt in Bdftpm.index.values:
                iimwb = np.where(np.logical_and(mwbdf.index.values >= tt - dt / 2,
                                                mwbdf.index.values < tt + dt / 2))
                iigps = np.where(np.logical_and(gpsdf.index.values >= tt - dt / 2,
                                                gpsdf.index.values < tt + dt / 2))
                if iimwb[0].size>0 or iigps[0].size>0:
                    # Store mwb values
                    Nb_mwb.append(len(iimwb[0]))
                    Hsb_mwb.append(np.nanmean(mwbdf.Hs.values[iimwb]))
                    Tpb_mwb.append(np.nanmedian(mwbdf.Tp.values[iimwb]))
                    cogb_mwb.append(np.nanmedian(mwbdf.cog.values[iimwb]))
                    # vector average peak direction Dp
                    u = np.mean(np.cos(mwbdf.Dp.values[iimwb] * np.pi / 180))
                    v = np.mean(np.sin(mwbdf.Dp.values[iimwb] * np.pi / 180))
                    if np.arctan2(v, u) * 180 / np.pi < 0:
                        Dpb_mwb.append(np.arctan2(v, u) * 180 / np.pi + 360)
                    else:
                        Dpb_mwb.append(np.arctan2(v, u) * 180 / np.pi)
                    # calculate WG distance to NDBC buoy
                    # lat = np.nanmedian(mwbdf.lat.values[iimwb])
                    # lon = np.nanmedian(mwbdf.lon.values[iimwb])
                    lat = np.nanmedian(gpsdf.latitude.values[iigps]) if \
                        np.isnan(np.nanmedian(mwbdf.lat.values[iimwb])) else \
                        np.nanmedian(mwbdf.lat.values[iimwb])
                    lon = np.nanmedian(gpsdf.longitude.values[iigps]) if \
                        np.isnan(np.nanmedian(mwbdf.lon.values[iimwb])) else \
                        np.nanmedian(mwbdf.lon.values[iimwb])
                    try:
                        d_mwb.append(distance((Bdftpm.LAT.values[0], Bdftpm.LON.values[0]), (lat, lon)).m)
                    except:
                        d_mwb.append(np.nan)
                    latb_mwb.append(lat)
                    lonb_mwb.append(lon)
                    # Store gpswaves values
                    Nb_gps.append(len(iigps[0]))
                    if len(iigps[0])==0:
                        Hsb_gps.append(np.nan)
                        Tpb_gps.append(np.nan)
                        Dpb_gps.append(np.nan)
                    else:
                        Hsb_gps.append(np.nanmean(gpsdf['Hs'].values[iigps]))
                        Tpb_gps.append(np.nanmedian(gpsdf['Tp'].values[iigps]))
                        # vector average peak direction Dp
                        u = np.mean(np.cos(gpsdf['Dp'].values[iigps] * np.pi / 180))
                        v = np.mean(np.sin(gpsdf['Dp'].values[iigps] * np.pi / 180))
                        if np.arctan2(v, u) * 180 / np.pi < 0:
                            Dpb_gps.append(np.arctan2(v, u) * 180 / np.pi + 360)
                        else:
                            Dpb_gps.append(np.arctan2(v, u) * 180 / np.pi)
                else:
                    # Drop value if not in GPSwaves and MWB
                    Bdftpm.drop([tt],inplace=True)
            # Append NDBC/CDIP dataframe
            Bdf30 = Bdf30.append(Bdftpm)

        # Significant wave height (Hs)
        ax4.plot(Bdf30.WVHT.values, Hsb_gps, '.r')
        ax4.plot(Bdf30.WVHT.values, Hsb_mwb, '.b')
        ax4.set_xlabel('NDBC Hs [m]', fontsize=fntsz)
        ax4.set_ylabel('WG Hs [m]', fontsize=fntsz)
        ax4.tick_params(labelsize=labsz)
        ax4.set_xlim(ax1.get_ylim())
        ax4.grid()

        # Peak wave period (Tp)
        ax5.plot(Bdf30.DPD.values, Tpb_gps, '.r')
        ax5.plot(Bdf30.DPD.values, Tpb_mwb, '.b')
        ax5.set_xlabel('NDBC Tp [s]', fontsize=fntsz)
        ax5.set_ylabel('WG Tp [s]', fontsize=fntsz)
        ax5.tick_params(labelsize=labsz)
        ax5.set_xlim(ax2.get_ylim())
        ax5.grid()

        # Peak wave direction (Dp)
        ax6.plot(Bdf30.MWD.values, Dpb_gps, '.r')
        ax6.plot(Bdf30.MWD.values, Dpb_mwb, '.b')
        ax6.set_xlabel('NDBC Dp [deg]', fontsize=fntsz)
        ax6.set_ylabel('WG Dp [deg]', fontsize=fntsz)
        ax6.tick_params(labelsize=labsz)
        ax6.set_xlim(ax6.get_ylim())
        ax6.grid()

    # Set ticks
    # ax4.set_xticks(ax1.get_yticks())
    # ax5.set_xticks(ax2.get_yticks())
    # ax6.set_xticks(ax3.get_yticks())

    # One to one lines
    ax4.plot([ax4.get_yticks()[0], ax4.get_yticks()[-1]],
             [ax4.get_yticks()[0], ax4.get_yticks()[-1]], '-k')
    ax5.plot([ax5.get_yticks()[0], ax5.get_yticks()[-1]],
             [ax5.get_yticks()[0], ax5.get_yticks()[-1]], '-k')
    ax6.plot([ax6.get_yticks()[0], ax6.get_yticks()[-1]],
             [ax6.get_yticks()[0], ax6.get_yticks()[-1]], '-k')

    # set x axis ticks
    ax4.axis('equal')
    ax5.axis('equal')
    ax6.axis('equal')
    if np.median(np.diff(ax1.get_yticks())) < 0.5:
        stp = 0.5
        ax1.set_yticks(np.arange(ax1.get_yticks()[1], ax1.get_yticks()[-1], step=stp))
    ax4.set_xticks(ax1.get_yticks())
    ax5.set_xticks(ax2.get_yticks())
    ax6.set_xticks(ax3.get_yticks())

    if figshow:
        fig.show()
    return fig

def waves_bulk_autoplot_timeseries(mwbdf, mwbspec, gpsdf, figshow=False):
    '''
    This function creates timeseries of bulk waves parameters (Hs,Tp,Dp) for `plotWGwaves_vx.py`.
    :param mwbdf: dataframe with bulk waves parameters from CORDC Mini Wave Buoy sensor package
    :param gpsdf: dataframe with bulk waves parameters from LRI GPSwaves sensor package
    :param Bdf_lst: list of dataframes with bulk waves parameters from NDBC/CDIP
    :param bidstr_lst: list of strings with buoy id's of len(Bdf_lst)
    :param figshow: if True plt.show() is executed
    :return: figure object
    '''
    # configure plot parameters
    labsz = 8
    fntsz = 10
    now_utc = datetime.datetime.utcnow()
    if mwbdf.empty:
        xL = (gpsdf.index[0], now_utc)
    else:
        xL = (mwbdf.index[0], now_utc)
    heights = [1, 1, 1, 1]
    gs_kw = dict(height_ratios=heights)
    fig, fig_axes = plt.subplots(4, 1, gridspec_kw=gs_kw, figsize=(12, 8), sharex=True)
    # Time-series
    ax1 = fig_axes[0]
    ax2 = fig_axes[1]
    ax3 = fig_axes[2]
    ax4 = fig_axes[3]

    # Significant wave height (Hs)
    l1, = ax1.plot(mwbdf.index, mwbdf.Hs, '.b')
    l2, = ax1.plot(gpsdf.index, gpsdf.Hs, '.r')
    ax1.set_ylabel('Hs [m]', fontsize=fntsz)
    ax1.tick_params(labelsize=labsz)
    ax1.set_xlim(xL)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Peak wave period (Tp)
    ax2.plot(mwbdf.index, mwbdf.Tp, '.b')
    ax2.plot(gpsdf.index, gpsdf.Tp, '.r')
    ax2.set_ylabel('Tp [s]', fontsize=fntsz)
    ax2.tick_params(labelsize=labsz)
    ax2.set_yticks(np.arange(4, 22, step=2))
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Peak wave direction (Dp)
    ax3.plot(mwbdf.index, mwbdf.Dp, '.b')
    ax3.plot(gpsdf.index, gpsdf.Dp, '.r')
    ax3.set_ylabel('Dp [deg]', fontsize=fntsz)
    ax3.set_ylim(0, 360)
    ax3.set_yticks(np.arange(0, 361, step=60))
    ax3.tick_params(labelsize=fntsz)

    # MWB spectrogram
    x, y = np.meshgrid(mwbspec.index, mwbspec.columns.values)
    Emin, Emax = 5E-2, 10
    fmin, fmax = 1 / 22, 1 / 2.5
    c = ax4.pcolormesh(x, y, mwbspec.values.transpose(), cmap='jet', norm=LogNorm(vmin=Emin, vmax=Emax))
    ax4.set_ylim(fmin, fmax)
    ax4.set_ylabel('Frequency [Hz]', fontsize=fntsz)
    ax4.set_xlabel('Time [UTC]', fontsize=fntsz)

    # add a colorobar
    gs = gridspec.GridSpec(ncols=3, nrows=4, right=0.95, figure=fig)
    axc = fig.add_subplot(gs[-1, -1])
    axc.set_visible(False)
    cbar = fig.colorbar(c, ax=axc, orientation='vertical')
    cax = cbar.ax
    # Add label on top of colorbar.
    # cbar.ax.set_xlabel("energy density \n[m$^2$/Hz]")
    cbar.ax.set_xlabel("$S_{\eta\eta}(f)$ [m$^2$/Hz]")
    cbar.ax.xaxis.set_label_position('top')

    # Legend
    leg_hlst = [l1, l2]
    leg_strlst = ['MWB', 'GPSWaves']
    ax1.legend(tuple(leg_hlst), tuple(leg_strlst), fontsize=fntsz, loc='best')

    if figshow:
        fig.show()
    return fig

def waves_bulk_autoplot_ww3(mwbdf, gpsdf, airdf, ww3df, figshow=False):
    '''
    This function creates timeseries plots with WW3 predictions of bulk waves parameters (Hs,Tp,Dp)
    for `plotWGwaves_vx.py`.
    :param mwbdf: dataframe with bulk waves parameters from CORDC Mini Wave Buoy sensor package
    :param gpsdf: dataframe with bulk waves parameters from LRI GPSwaves sensor package
    :param Bdf_lst: list of dataframes with bulk waves parameters from NDBC/CDIP
    :param bidstr_lst: list of strings with buoy id's of len(Bdf_lst)
    :param figshow: if True plt.show() is executed
    :return: figure object
    '''
    # configure plot parameters
    labsz = 8
    fntsz = 10
    if mwbdf.empty:
        xL = (gpsdf.index[0], ww3df.index[-1])
    else:
        xL = (mwbdf.index[0], ww3df.index[-1])
    heights = [1, 1, 1, 1]
    gs_kw = dict(height_ratios=heights)
    fig, fig_axes = plt.subplots(4, 1, gridspec_kw=gs_kw, figsize=(12, 8), sharex=True)
    # Time-series
    ax1 = fig_axes[0]
    ax2 = fig_axes[1]
    ax3 = fig_axes[2]
    ax4 = fig_axes[3]

    # Significant wave height (Hs)
    l1, = ax1.plot(mwbdf.index, mwbdf.Hs, '.b')
    l2, = ax1.plot(gpsdf.index, gpsdf.Hs, '.r')
    l3, = ax1.plot(ww3df.index, ww3df.Hs, '-m')
    ax1.set_ylabel('Hs [m]', fontsize=fntsz)
    ax1.tick_params(labelsize=labsz)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Peak wave period (Tp)
    ax2.plot(mwbdf.index, mwbdf.Tp, '.b')
    ax2.plot(gpsdf.index, gpsdf.Tp, '.r')
    ax2.plot(ww3df.index, ww3df.Tp, '-m')
    ax2.set_ylabel('Tp [s]', fontsize=fntsz)
    ax2.tick_params(labelsize=labsz)
    ax2.set_yticks(np.arange(4, 22, step=2))
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Peak wave direction (Dp)
    ax3.plot(mwbdf.index, mwbdf.Dp, '.b')
    ax3.plot(gpsdf.index, gpsdf.Dp, '.r')
    ax3.plot(ww3df.index, ww3df.Dp, '-m')
    ax3.set_ylabel('Dp [deg]', fontsize=fntsz)
    ax3.set_ylim(0, 360)
    ax3.set_yticks(np.arange(0, 361, step=60))
    ax3.tick_params(labelsize=fntsz)

    # wind observations
    iidec = 6
    uw_obs = -airdf['WindSpeed'].values * kt2mps * np.sin(np.deg2rad(airdf['WindDirection'].values))
    vw_obs = -airdf['WindSpeed'].values * kt2mps * np.cos(np.deg2rad(airdf['WindDirection'].values))
    wspd_obs = airdf['WindSpeed'].values * kt2mps
    yL4 = np.nanmax(np.append(np.abs(vw_obs), np.abs(ww3df['v_wind'].values)))
    cL4 = np.nanmax(np.append(wspd_obs, ww3df['wspd'].values))
    # qobs = ax4.quiver(airdf.index[::iidec], 0,
    #                   uw_obs[::iidec], vw_obs[::iidec],
    #                   wspd_obs[::iidec],
    #                   cmap=plt.cm.OrRd,
    #                   clim=[0, cL4],
    #                   edgecolor='k',
    #                   headwidth=2, headlength=3, headaxislength=2.75)
    #
    # qww3 = ax4.quiver(ww3df.index, 0,
    #                  ww3df['u_wind'].values, ww3df['v_wind'].values,
    #                  ww3df['wspd'].values,
    #                  cmap=plt.cm.OrRd,
    #                  clim=[0, cL4],
    #                  edgecolor='m',
    #                  hatch='/',
    #                  headwidth=2, headlength=3, headaxislength=2.75)
    try:
        qww3 = ax4.quiver(np.append(airdf.index[::iidec],ww3df.index), 0,
                          np.append(uw_obs[::iidec],ww3df['u_wind'].values),
                          np.append(vw_obs[::iidec],ww3df['v_wind'].values),
                          np.append(wspd_obs[::iidec],ww3df['wspd'].values),
                          cmap=plt.cm.OrRd,
                          clim=[0, cL4],
                          edgecolor='k',
                          headwidth=2, headlength=3, headaxislength=2.75)
        ax4.plot([airdf.index[-1], airdf.index[-1]], [-yL4, yL4], '--m')
    except:
        qww3 = ax4.quiver(ww3df.index, 0,
                          ww3df['u_wind'].values,
                          ww3df['v_wind'].values,
                          ww3df['wspd'].values,
                          cmap=plt.cm.OrRd,
                          clim=[0, cL4],
                          edgecolor='k',
                          headwidth=2, headlength=3, headaxislength=2.75)
    # units = 'xy',
    # hatch = '/',

    ax4.set_ylim(-yL4, yL4)
    ax4.set_ylabel('Wspd [m/s]', fontsize=fntsz)
    ax4.set_xlabel('Time [UTC]', fontsize=fntsz)
    ax4.set_xlim(xL)

    # Legend
    leg_hlst = [l1, l2, l3]
    leg_strlst = ['MWB', 'GPSWaves', 'WW3']
    ax1.legend(tuple(leg_hlst), tuple(leg_strlst), fontsize=fntsz, loc='best')

    if figshow:
        fig.show()
    return fig


def mets_bulk_autoplot(WXTdf,airdf,Bdf_lst,bidstr_lst,figshow=False):
    '''
    This function creates timeseries and scatter plots of bulk meteorological parameters (WindSpeed, WindDirection,
    air temperature) for `plotWGmets_vx.py`.
    :param WXTdf: dataframe with CORDC Mets sensor package data
    :param gpsdf: dataframe with LRI Airmar sensor package data
    :param Bdf_lst: list of dataframes with NDBC/CDIP data
    :param bidstr_lst: list of strings with buoy id's of len(Bdf_lst)
    :param figshow: if True, plt.show() is executed
    :return: figure object
    '''
    import datetime
    import matplotlib.pyplot as plt
    from geopy.distance import distance
    # configure plot parameters
    labsz = 8
    fntsz = 10
    now_utc = datetime.datetime.utcnow()
    if WXTdf.empty:
        tL = (airdf.index[0], now_utc)
    else:
        tL = (WXTdf.index[0], now_utc)
    widths = [4, 1]
    heights = [1, 1, 1, 1]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, fig_axes = plt.subplots(4, 2, sharey='row', gridspec_kw=gs_kw,
                                 figsize=(15, 12))

    # Time-series
    ax1 = fig_axes[0][0]
    ax2 = fig_axes[1][0]
    ax3 = fig_axes[2][0]
    ax4 = fig_axes[3][0]

    # Wind speed
    l2, = ax1.plot(airdf.index, airdf.WindSpeed, '.r')
    l1, = ax1.plot(WXTdf.index, WXTdf.WindSpeed, '.b')
    ax1.set_ylabel('wind speed [m/s]', fontsize=fntsz)
    ax1.tick_params(labelsize=labsz)
    ax1.set_xlim(tL)
    # ax1.legend((l1, l2), ('WXT', 'Airmar'), fontsize=12, loc='best')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Wind direction
    ax2.plot(airdf.index, airdf.WindDirection, '.r')
    ax2.plot(WXTdf.index, WXTdf.WindDirection, '.b')
    ax2.set_ylabel('wind direction [deg]', fontsize=fntsz)
    ax2.set_ylim(0, 360)
    ax2.set_yticks(np.arange(0, 361, step=60))
    ax2.tick_params(labelsize=labsz)
    ax2.set_xlim(tL)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Air pressure
    ax3.plot(airdf.index, airdf.pressure, '.r')
    ax3.plot(WXTdf.index, WXTdf['pressure_baro'], '.b')
    ax3.set_ylabel('pressure [mbar]', fontsize=fntsz)
    ax3.tick_params(labelsize=labsz)
    ax3.set_xlim(tL)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # Air temperature
    ax4.plot(airdf.index, airdf.temperature, '.r')
    ax4.plot(WXTdf.index, WXTdf.temperature, '.b')
    ax4.set_ylabel('air temperature [C]', fontsize=fntsz)
    ax4.tick_params(labelsize=labsz)
    ax4.set_xlim(tL)
    ax4.set_xlabel('Time [UTC]', fontsize=fntsz)

    leg_hlst = [l1, l2]
    leg_strlst = ['CORDC', 'Airmar']
    if Bdf_lst:
        # Dynamically assign symbols/markers for each NDBC station
        sym_lst = ['.', '*', 'x', '^', 'v', '<', '>', '1', '2', '3', '4']
        bidstr_unique = list(set(bidstr_lst))
        bstr_chk = None
        cc = 0
        for Bdf, bstr in zip(Bdf_lst, bidstr_lst):
            # find a symbol/marker
            try:
                sym = sym_lst[bidstr_unique.index(bstr)]
            except:
                sym = '.'

            l3, = ax1.plot(Bdf.index, Bdf.WSPD, sym + 'k')
            ax2.plot(Bdf.index, Bdf.WDIR, sym + 'k')
            ax3.plot(Bdf.index, Bdf.PRES, sym + 'k')
            ax4.plot(Bdf.index, Bdf.ATMP, sym + 'k')
            # Update legend handle and string
            if bstr_chk != bstr:
                leg_hlst.append(l3)
                leg_strlst.append('NDBC ' + bstr)
                bstr_chk = bstr
                cc += 1
    # Legend
    ax1.legend(tuple(leg_hlst), tuple(leg_strlst), fontsize=fntsz, loc='best')

    # --------------------------------------------------------
    # Scatter plots
    ax5 = fig_axes[0][1]
    ax6 = fig_axes[1][1]
    ax7 = fig_axes[2][1]
    ax8 = fig_axes[3][1]
    # --------------------------------------------------------
    # bin average WXT and Airmar data using NDBC time base
    # --------------------------------------------------------

    Nb_wxt, wspd_wxt, wdir_wxt, pres_wxt, atmp_wxt, d_wxt, latb_wxt, lonb_wxt = [], [], [], [], [], [], [], []
    Nb_air, wspd_air, wdir_air, pres_air, atmp_air = [], [], [], [], []
    if Bdf_lst:
        # Bdf30 = pd.DataFrame(columns=Bdf_lst[0].keys())
        Bdf30 = pd.DataFrame(columns=['WDIR', 'WSPD', 'PRES', 'ATMP','LON', 'LAT'])
        for Bdf in Bdf_lst:
            Bdf = Bdf[['WDIR', 'WSPD', 'PRES', 'ATMP','LON', 'LAT']].dropna(thresh=4)
            # Resample NDBC/CDIP data into 30 min interval bins
            Bdftpm = Bdf.resample('30T').mean().dropna(how='all')
            dt = np.abs(np.median(np.diff(Bdftpm.index.values)))
            for tt in Bdftpm.index.values:
                iiwxt = np.where(np.logical_and(WXTdf.index.values >= tt - dt / 2,
                                                WXTdf.index.values < tt + dt / 2))
                iiair = np.where(np.logical_and(airdf.index.values >= tt - dt / 2,
                                                airdf.index.values < tt + dt / 2))
                if iiwxt[0].size>0 or iiair[0].size>0:
                    # Store WXT values
                    Nb_wxt.append(len(iiwxt[0]))
                    wspd_wxt.append(np.nanmean(WXTdf.WindSpeed.values[iiwxt]))
                    pres_wxt.append(np.nanmedian(WXTdf['pressure_baro'].values[iiwxt]))
                    atmp_wxt.append(np.nanmedian(WXTdf.temperature.values[iiwxt]))
                    wdir_wxt.append(np.nanmedian(WXTdf.WindDirection.values[iiwxt]))
                    # # vector average wind direction
                    # wdir = WXTdf.WindDirection.values[iiwxt]
                    # u = np.nanmean(np.cos(wdir * np.pi / 180)*WXTdf.WindSpeed.values[iiwxt])
                    # v = np.nanmean(np.sin(wdir * np.pi / 180)*WXTdf.WindSpeed.values[iiwxt])
                    # if np.arctan2(v, u) * 180 / np.pi < 0:
                    #     wdir_wxt.append(np.arctan2(v, u) * 180 / np.pi + 360)
                    # else:
                    #     wdir_wxt.append(np.arctan2(v, u) * 180 / np.pi)
                    # calculate WG distance to NDBC buoy
                    lat = np.nanmedian(airdf.latitude.values[iiair]) if \
                        np.isnan(np.nanmedian(WXTdf.latitude.values[iiwxt])) else \
                        np.nanmedian(WXTdf.latitude.values[iiwxt])
                    lon = np.nanmedian(airdf.longitude.values[iiair]) if \
                        np.isnan(np.nanmedian(WXTdf.longitude.values[iiwxt])) else \
                        np.nanmedian(WXTdf.longitude.values[iiwxt])
                    d_wxt.append(distance((Bdftpm.LAT.values[0], Bdftpm.LON.values[0]), (lat, lon)).m)
                    latb_wxt.append(lat)
                    lonb_wxt.append(lon)
                    # Store Airmar values
                    Nb_air.append(len(iiair[0]))
                    if len(iiair[0])==0:
                        wspd_air.append(np.nan)
                        pres_air.append(np.nan)
                        atmp_air.append(np.nan)
                        wdir_air.append(np.nan)
                    else:
                        wspd_air.append(np.nanmean(airdf.WindSpeed.values[iiair]))
                        pres_air.append(np.nanmedian(airdf.pressure.values[iiair]))
                        atmp_air.append(np.nanmedian(airdf.temperature.values[iiair]))
                        wdir_air.append(np.nanmedian(airdf.WindDirection.values[iiair]))
                else:
                    # Drop value if not in Airmar and WXT
                    Bdftpm.drop([tt],inplace=True)
            # Append NDBC/CDIP dataframe
            Bdf30 = Bdf30.append(Bdftpm)

        # Wind Speed (m/s)
        ax5.plot(Bdf30.WSPD, wspd_air, '.r')
        ax5.plot(Bdf30.WSPD, wspd_wxt, '.b')
        ax5.set_xlabel('NDBC wind speed [m/s]', fontsize=fntsz)
        ax5.set_ylabel('WG wind speed [m/s]', fontsize=fntsz)
        ax5.tick_params(labelsize=labsz)
        ax5.set_xlim(ax1.get_ylim())
        ax5.grid()

        # Wind direction [deg]
        ax6.plot(Bdf30.WDIR, wdir_air, '.r')
        ax6.plot(Bdf30.WDIR, wdir_wxt, '.b')
        ax6.set_xlabel('NDBC wind dir. [deg]', fontsize=fntsz)
        ax6.set_ylabel('WG wind dir. [deg]', fontsize=fntsz)
        ax6.tick_params(labelsize=labsz)
        ax6.set_xlim(ax2.get_ylim())
        ax6.grid()
        # normalized difference (ad-hoc)
        # circ_diff = (wdir_wxt - Bdf30.WDIR) % 360
        # # absolute difference
        # circ_diff_abs = np.nanmin([360 - circ_diff, circ_diff], axis=0)
        # print(np.nanmean(circ_diff_abs))

        # Air pressure [hPa or mbar]
        ax7.plot(Bdf30.PRES, pres_air, '.r')
        ax7.plot(Bdf30.PRES, pres_wxt, '.b')
        ax7.set_xlabel('NDBC pres. [mbar]', fontsize=fntsz)
        ax7.set_ylabel('WG pres. [mbar]', fontsize=fntsz)
        ax7.tick_params(labelsize=labsz)
        ax7.set_xlim(ax3.get_ylim())
        ax7.grid()

        # Air temperature [deg C]
        ax8.plot(Bdf30.ATMP, atmp_air, '.r')
        ax8.plot(Bdf30.ATMP, atmp_wxt, '.b')
        ax8.set_xlabel('NDBC temp. [C]', fontsize=fntsz)
        ax8.set_ylabel('WG temp. [C]', fontsize=fntsz)
        ax8.tick_params(labelsize=labsz)
        ax8.set_xlim(ax4.get_ylim())
        ax8.grid()

        # One to one lines
        ax5.plot([ax5.get_yticks()[0], ax5.get_yticks()[-1]],
                 [ax5.get_yticks()[0], ax5.get_yticks()[-1]], '-k')
        ax6.plot([ax6.get_yticks()[0], ax6.get_yticks()[-1]],
                 [ax6.get_yticks()[0], ax6.get_yticks()[-1]], '-k')
        ax7.plot([ax7.get_yticks()[0], ax7.get_yticks()[-1]],
                 [ax7.get_yticks()[0], ax7.get_yticks()[-1]], '-k')
        ax8.plot([ax8.get_yticks()[0], ax8.get_yticks()[-1]],
                 [ax8.get_yticks()[0], ax8.get_yticks()[-1]], '-k')

        # set x axis ticks
        ax5.axis('equal')
        ax6.axis('equal')
        ax7.axis('equal')
        ax8.axis('equal')
        if np.median(np.diff(ax3.get_yticks())) < 2:
            # for readability
            stp = 2
            ax3.set_yticks(np.arange(ax3.get_yticks()[0], ax3.get_yticks()[-1], step=stp))
        ax5.set_xticks(ax1.get_yticks())
        ax6.set_xticks(ax2.get_yticks())
        ax7.set_xticks(ax3.get_yticks())
        ax8.set_xticks(ax4.get_yticks())

    if figshow:
        fig.show()
    return fig

def mets_bulk_autoplot_timeseries(WXTdf,airdf,figshow=False):
    '''
        This function creates timeseries plots of bulk meteorological parameters (WindSpeed, WindDirection,
        air temperature) for `plotWGmets_vx.py`.
        :param WXTdf: dataframe with CORDC Mets sensor package data
        :param gpsdf: dataframe with LRI Airmar sensor package data
        :param figshow: if True, plt.show() is executed
        :return: figure object
        '''

    # configure plot parameters
    labsz = 8
    fntsz = 10
    now_utc = datetime.datetime.utcnow()
    if WXTdf.empty:
        tL = (airdf.index[0], now_utc)
    else:
        tL = (WXTdf.index[0], now_utc)
    # widths = [3, 1]
    heights = [1, 1, 1, 1, 1]
    gs_kw = dict(height_ratios=heights)
    fig, fig_axes = plt.subplots(5, 1, gridspec_kw=gs_kw,figsize=(12, 10), sharex=True)
    # Time-series
    ax1 = fig_axes[0]
    ax2 = fig_axes[1]
    ax3 = fig_axes[2]
    ax4 = fig_axes[3]
    ax5 = fig_axes[4]

    # Wind speed
    l2, = ax1.plot(airdf.index, airdf.WindSpeed, '.r')
    l1, = ax1.plot(WXTdf.index, WXTdf.WindSpeed, '.b')
    ax1.set_ylabel('wind speed [m/s]', fontsize=fntsz)
    ax1.tick_params(labelsize=labsz)
    ax1.set_xlim(tL)

    # Wind direction
    ax2.plot(airdf.index, airdf.WindDirection, '.r')
    ax2.plot(WXTdf.index, WXTdf.WindDirection, '.b')
    ax2.set_ylabel('wind direction [deg]', fontsize=fntsz)
    ax2.set_ylim(0, 360)
    ax2.set_yticks(np.arange(0, 361, step=60))
    ax2.tick_params(labelsize=labsz)
    # plt.setp(ax2.get_xticklabels(), visible=False)

    # Air pressure
    ax3.plot(airdf.index, airdf.pressure, '.r')
    ax3.plot(WXTdf.index, WXTdf['pressure_baro'], '.b')
    ax3.set_ylabel('pressure [mbar]', fontsize=fntsz)
    ax3.tick_params(labelsize=labsz)
    # ax3.set_xlim(airdf.index[0], airdf.index[-1])

    # Air temperature
    ax4.plot(airdf.index, airdf.temperature, '.r')
    ax4.plot(WXTdf.index, WXTdf.temperature, '.b')
    ax4.set_ylabel('air temperature [C]', fontsize=fntsz)
    ax4.tick_params(labelsize=labsz)
    # ax4.set_xlabel('Time [UTC]', fontsize=fntsz)

    # Air temperature
    ax5.plot(WXTdf.index, WXTdf['RelativeHumidity'], '.b')
    ax5.set_ylabel('relative hum. [%]', fontsize=fntsz)
    ax5.tick_params(labelsize=labsz)
    ax5.set_xlabel('Time [UTC]', fontsize=fntsz)

    # Legend
    leg_hlst = [l1, l2]
    leg_strlst = ['CORDC', 'Airmar']
    ax1.legend(tuple(leg_hlst), tuple(leg_strlst), fontsize=fntsz, loc='best')

    # --------------------------------------------------------
    if figshow:
        fig.show()
    return fig

