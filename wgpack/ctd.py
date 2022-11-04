# Conductivity, Temperature and Depth (CTD) sensor module
import numpy as np


def read_in_SBGPCTD(CTD_filepath):
    import gsw
    import pandas as pd
    from scipy import stats

    # read in SBGPCTD file
    ctddf = pd.read_csv(CTD_filepath)

    # remove to all rows not containing data
    ctddf = ctddf[ctddf['CTDO'] == 'CTDO']
    # rename columns
    dftmp = pd.read_csv(CTD_filepath, header=0)
    og_columns = dftmp.columns
    zip_iterator = zip(ctddf.columns, og_columns)
    columns_mapper = dict(zip_iterator)
    ctddf.rename(columns=columns_mapper, inplace=True)
    # set date as index
    ctddf.set_index(pd.to_datetime(ctddf['Date'] + ' ' + ctddf['Time']), inplace=True)
    ctddf.drop(columns=['CTDO', 'Date', 'Time', 'Millisec'], inplace=True)
    #     ctddf.drop(columns=['CTDO','Date','Millisec'],inplace=True)
    #     ctddf.set_index('Time',inplace=True)
    # rename columns again to match Data Portal output
    col_old = ['Latitude', 'Longitude', 'Conductivity(S/m)', 'Temperature(C)', 'Pressure(decibars)', 'Oxygen(Hz)']
    col_new = ['latitude', 'longitude', 'conductivity', 'temperature', 'pressure', 'oxygenHz']
    zip_iterator = zip(col_old, col_new)
    columns_mapper = dict(zip_iterator)
    ctdDPdf = ctddf.rename(columns=columns_mapper)
    ctdDPdf['salinity'] = np.nan
    ctdDPdf['dissolvedOxygen'] = np.nan
    ctdDPdf['oxygenSolubility'] = np.nan
    ctdDPdf['DO (ml/L)'] = np.nan
    ctdDPdf['DO (muM/kg)'] = np.nan
    # convert to float values
    ctdDPdf['latitude'] = ctdDPdf['latitude'].astype(float)
    ctdDPdf['longitude'] = ctdDPdf['longitude'].astype(float)
    ctdDPdf['conductivity'] = ctdDPdf['conductivity'].astype(float)
    ctdDPdf['temperature'] = ctdDPdf['temperature'].astype(float)
    ctdDPdf['pressure'] = ctdDPdf['pressure'].astype(float)
    # Remove outliers
    ctdDPdf['conductivity'][np.abs(stats.zscore(ctdDPdf['conductivity']) > 4)] = np.nan
    ctdDPdf['temperature'][np.abs(stats.zscore(ctdDPdf['temperature']) > 4)] = np.nan
    # Compute salinity
    # Reference: https://teos-10.github.io/GSW-Python/conversions.html
    C = ctdDPdf['conductivity'].values * 10  # Conductivity, mS/cm
    t = ctdDPdf['temperature'].values  # In-situ temperature (ITS-90), degrees C
    p = ctdDPdf['pressure'].values  # Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    ctdDPdf['salinity'] = gsw.conversions.SP_from_C(C, t, p)

    return ctdDPdf

def compute_OxSol(T, S):
    '''
    This function computes oxygen saturation values from temperature and salinity, following Garcia and Gordon (1992).
    Oxygen saturation is the volume of oxygen gas at standard temperature and pressure  conditions (STP) absorbed from
    humidity-saturated air at a total pressure of one atmosphere, per unit volume of the liquid at the temperature
    of measurement. Units are in ml/l. To compute mg/l, multiply the values in the table by 1.42903.
    References:
    See Appendix A in Seabird Application note 64
    https://www.seabird.com/asset-get.download.jsa?code=251036
    Garcia and Gordon (1992) "Oxygen solubility in seawater: Better fitting equations", Limnology & Oceanography,
    vol 37(6), p1307-1312.
    :param T: water temperature [deg C]
    :param S: salinity [psu]
    :return: Oxygen saturation values [ml/l]
    '''
    from seawater.library import T90conv
    A0 = 2.00907; A1 = 3.22014; A2 = 4.0501; A3 = 4.94457; A4 = -0.256847; A5 = 3.88767
    B0 = -0.00624523; B1 = -0.00737614; B2 = -0.010341; B3 = -0.00817083
    C0 = -0.000000488682
    # convert to ITS-90 deg C
    T90 = T90conv(T)
    Ts = np.log((298.15 - T90) / (273.15 + T90))

    return np.exp(A0 + A1 * (Ts) + A2 * (Ts) ** 2 + A3 * (Ts) ** 3 + A4 * (Ts) ** 4 + A5 * (Ts) ** 5 +
                  S * (B0 + B1 * (Ts) + B2 * (Ts) ** 2 + B3 * (Ts) ** 3) + C0 * S ** 2)


def OxHz2DO(F, dFdt, T, P, S, OxSol, vnam):
    '''
    This function converts oxygenHz (measured) to dissolved oxygen (DO) in ml/L and muM/kg.
    :param F:       ctdDPdf['oxygenHz'].values
    :param dFdt:    np.gradient(F, ctdDPdf['time'].values)
                    Time derivative of SBE 43 output oxygen signal (volts/second)
    :param T:       ctdDPdf['temperature'].values
    :param P:       ctdDPdf['pressure'].values
    :param S:       ctdDPdf['salinity'].values
    :param OxSol:   ctdDPdf['oxygenSolubility'].values
                    Oxygen saturation value after Garcia and Gordon (1992)
    :param vnam:    vehicle name e.g., 'magnus', 'sv3-253'
    :return:        output dictionary containing dissolved oxygen (DO) in ml/L and muM/kg

    References:
    05574 - USER GUIDE, SEA-BIRD ELECTRONICS GLIDER PAYLOAD CTD.pdf
    https://pythonhosted.org/seawater/eos80.html
    https://www.seabird.com/asset-get.download.jsa?code=251036
    SBGPCTD command to display calibration coefficients on SMC
    ServerExec SBGPCTD "ctd --expert dc"
    '''
    import seawater as sw
    from seawater.library import T90conv
    # Calibration Coefficients (currently stored on the vehicle)
    if vnam == 'magnus':
        FOFFSET = -8.295800e+02  # Voltage at zero oxygen signal
        SOC = 3.117100e-04  # Oxygen signal slope
        # A, B, C: Residual temperature correction factors
        A = -5.250800e-03
        B = 2.206300e-04
        C = -2.873600e-06
        E = 3.600000e-02  # Pressure correction factor
        TAU20 = 1.020000e+00  # Sensor time constant tau (T,P) at 20 degC, 1 atmosphere, 0 PSU; slope term in calculation of tau(T,P)
        # D1, D2: Temperature and pressure correction factors in calculation of tau(T,P)
        D1 = 1.926340e-04
        D2 = -4.648030e-02
        # H1, H2, H3: Hysteresis correction factors
        H1, H2, H3 = -3.300000e-02, 5.000000e+03, 1.450000e+03
    elif vnam == 'sv3-1101':
        # SBE 43 S/N 3490 10-Oct-20
        FOFFSET = -8.247100e+02  # Voltage at zero oxygen signal
        SOC = 2.289400e-04   # Oxygen signal slope
        # A, B, C: Residual temperature correction factors
        A = -4.360500e-03
        B = 2.280200e-04
        C = -3.572600e-06
        E = 3.600000e-02  # Pressure correction factor
        TAU20 = 8.399999e-01  # Sensor time constant tau (T,P) at 20 degC, 1 atmosphere, 0 PSU; slope term in calculation of tau(T,P)
        # D1, D2: Temperature and pressure correction factors in calculation of tau(T,P)
        D1 = 1.926340e-04
        D2 = -4.648030e-02
        # H1, H2, H3: Hysteresis correction factors
        H1, H2, H3 = -3.300000e-02, 5.000000e+03, 1.450000e+03
    elif vnam == 'sv3-1103':
        # SBE 43 S/N 3492 13-Jun-18
        FOFFSET = -8.616800e+02  # Voltage at zero oxygen signal
        SOC = 2.819200e-04   # Oxygen signal slope
        # A, B, C: Residual temperature correction factors
        A = -3.947500e-03
        B = 1.961200e-04
        C = -3.255900e-06
        E = 3.600000e-02  # Pressure correction factor
        TAU20 = 8.700000e-01  # Sensor time constant tau (T,P) at 20 degC, 1 atmosphere, 0 PSU; slope term in calculation of tau(T,P)
        # D1, D2: Temperature and pressure correction factors in calculation of tau(T,P)
        D1 = 1.926340e-04
        D2 = -4.648030e-02
        # H1, H2, H3: Hysteresis correction factors
        H1, H2, H3 = -3.300000e-02, 5.000000e+03, 1.450000e+03
    else:
        FOFFSET = 0  # Voltage at zero oxygen signal
        SOC = 0  # Oxygen signal slope
        # A, B, C: Residual temperature correction factors
        A = 0
        B = 0
        C = 0
        E = 0  # Pressure correction factor
        TAU20 = 0  # Sensor time constant tau (T,P) at 20 degC, 1 atmosphere, 0 PSU; slope term in calculation of tau(T,P)
        # D1, D2: Temperature and pressure correction factors in calculation of tau(T,P)
        D1 = 0
        D2 = 0
        # H1, H2, H3: Hysteresis correction factors
        H1, H2, H3 = 0, 0, 0

    # Calculated Values
    TAU = TAU20 * np.exp(D1 * P + D2 * (T - 20))  # sensor time constant at temperature and pressure
    K = T + 273.15

    # Calculate dissolved oxygen (ml/L)
    DOmlL = (SOC * (F + FOFFSET + TAU * dFdt)) * OxSol * (1.0 + A * T + B * T ** 2 + C * T ** 3) * np.exp(E * P / K)
    # Calculate dissolved oxygen (micro Mol/Kg)
    # [μmole/Kg] = [ml/L] * 44660 / (sigma_theta(P=0,Theta,S) + 1000)
    # Sigma_theta (potential density) is the density a parcel of water would have if it were raised adiabatically
    # to the surface without change in salinity.
    T90 = T90conv(T)
    sigma_theta = sw.pden(S, T90, P, pr=0)
    DOmuMkg = DOmlL * 44660 / (sigma_theta + 1000)

    # create output dictionary
    DOdict = {
        'DOmlL': DOmlL,
        'DOmuMkg': DOmuMkg
    }
    return DOdict