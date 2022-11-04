# navigation module
import numpy as np

def get_bearing(p1,p2):
    '''
    This function returns the bearing from initial point to destination point.
    References:
    http://www.movable-type.co.uk/scripts/latlong.html
    :param p1: (lat1,lon1) point - Latitude/longitude of initial point
    :param p2: (lat2,lon2) point - Latitude/longitude of destination point
    :return: Bearing in degrees from north (0°..360°)
    '''
    # extract coordinates
    lat1,lon1 = p1[0],p1[1]
    lat2,lon2 = p2[0],p2[1]
    # convert to radians
    lat1,lat2 = lat1*np.pi/180, lat2*np.pi/180
    dlon = (lon2-lon1)*np.pi/180
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    y = np.sin(dlon) * np.cos(lat2)
    return (np.arctan2(y, x)*180/np.pi)%360

def wrapTo180(thet):
    '''
    Wrap angle in degrees to [-180 180].
    This function wraps angles in thet, in degrees, to the interval [–180, 180]
    such that 180 maps to 180 and -180 maps to -180. In general, odd, positive
    multiples of 180 map to 180 and odd, negative multiples of 180 map to –180.

    Call signatures::

     thet180 = thetagrids(thet)

    Parameters
    ----------
    thet : list, array or floats specified in degrees

    Returns
    -------
    Wrapped angles, specified as list, array or floats with values in the range [–180, 180].
    '''
    return ((thet-180)%360)-180

def get_WEAthet(COG,Dw):
    # Calculates and returns wave encounter angle  defined as...
    # References:
    # Estimations of on-site directional wave spectra from measured ship responses
    # https://www.sciencedirect.com/science/article/pii/S0951833906000529
    # compute difference
    dth = COG-(Dw+180)%360
    # convert from [0,360] to [-180,180]
    return wrapTo180(dth)

def get_WEA(COG,Dw):
    # Calculates and returns wave encounter angle  defined as...
    # References:
    # Estimations of on-site directional wave spectra from measured ship responses
    # compute difference
    dth = COG-Dw
    # convert from [0,360] to [-180,180]
    return wrapTo180(dth)

def rotate_via_numpy(x,y,radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    import numpy as np
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return float(m.T[0]), float(m.T[1])