import sys
import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as optimize


def cross(y1, y2, x):
    # linear interpolators
    opts = {'fill_value': 'extrapolate'}
    f1 = interpolate.interp1d(x, y1, **opts)
    f2 = interpolate.interp1d(x, y2, **opts)

    # possible range for an intersection
    xmin = np.min((x, x))
    xmax = np.max((x, x))

    # number of intersections
    xuniq = np.unique((x, x))
    xvals = xuniq[(xmin <= xuniq) & (xuniq <= xmax)]
    # note that it's bad practice to compare floats exactly
    # but worst case here is a bit of redundance, no harm

    # for each combined interval there can be at most 1 intersection,
    # so looping over xvals should hopefully be enough
    # one can always err on the safe side and loop over a `np.linspace`

    intersects = []
    for xval in xvals:
        x0, = optimize.fsolve(lambda x: f1(x)-f2(x), xval)
        if (xmin <= x0 <= xmax
            and np.isclose(f1(x0), f2(x0))
                and not any(np.isclose(x0, intersects))):
            intersects.append(x0)

    return intersects[0]
    # print(f1(intersects))
    # print(f2(intersects))


sys.modules[__name__] = cross
