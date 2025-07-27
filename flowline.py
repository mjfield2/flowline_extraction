import numpy as np
from scipy.interpolate import LinearNDInterpolator
from numbers import Number

def linear_interp(xx, yy, vx, vy, x0, y0, maxdist):
    """
    2D linear interpolation of velocity field

    Args:
        xx (numpy.ndarray): 2D x-coordinates, result of np.meshgrid
        yy (numpy.ndarray): 2D y-coordinates, result of np.meshgrid
        vx (numpy.ndarray): 2D x-component of velocity field
        vy (numpy.ndarray): 2D y-component of velocity field
        x0 (float): x-coordinate to interpolate
        y0 (float): y-coordinate to interpolate
        maxdist : maximum distance to look for interpolating
    Outputs:
        Interpolated x velocity and y velocity
    """
    dist = np.sqrt((xx-x0)**2+(yy-y0)**2)
    dist_mask = dist < maxdist
    points = np.array([xx[dist_mask], yy[dist_mask]]).T
    vx_trim = vx[dist_mask]
    vy_trim = vy[dist_mask]
    vx_interp = LinearNDInterpolator(points, vx_trim)
    vx_pred = vx_interp((x0, y0))
    vy_interp = LinearNDInterpolator(points, vy_trim)
    vy_pred = vy_interp((x0, y0))
    return vx_pred, vy_pred

def flowline(xx, yy, vx, vy, x0, y0, stride, total_dist, maxdist=5e3, direction='forward', mode='distance', max_iter=1000, extend=0):
    """
    Extract profile points along flowline.

    Args:
        xx (numpy.ndarray): 2D x-coordinates, result of np.meshgrid
        yy (numpy.ndarray): 2D y-coordinates, result of np.meshgrid
        vx (numpy.ndarray): 2D x-component of velocity field
        vy (numpy.ndarray): 2D y-component of velocity field
        x0 (numpy.ndarray): starting x-coordinate
        y0 (numpy.ndarray): starting y-coordinate
        stride : how far to move in time or distance, corresponding to mode
        total_dist : total distance to move
        maxdist : maximum distance to look for interpolating velocity
        direction : move forward or backward from starting point
        mode (str): distance or time for movement. Time results in
            points spaced scaled by velocity and distance results in
            points spaced equally in space
        max_iter (int): maximum number of points. This can be helpful when
            using time since a while loop is used
        extend (int): If the flowline hits NaN in the velocity field, extent
            by this many strides based on the past velocity vector
    Outputs:
        Array of points with x-coordinate in first column and y-coordinate in second
        and cumulative distances along profile
    """

    _check_params(xx, yy, vx, vy, x0, y0, stride, total_dist, maxdist, direction, mode, max_iter, extend)
    
    vx0, vy0 = linear_interp(xx, yy, vx, vy, x0, y0, maxdist)

    cum_dist = 0
    n_iter = 0
    x_current = x0
    y_current = y0
    points = [[x0, y0]]
    cum_dist_coll = [0]
    while (cum_dist < total_dist) & (n_iter < max_iter):
        vx1, vy1 = linear_interp(xx, yy, vx, vy, x_current, y_current, maxdist)

        # velocity is NaN, extend if needed otherwise break
        if np.any(np.isnan([vx1, vy1]))==True:
            x_current = points[-1][0]
            y_current = points[-1][1]
            vx_old, vy_old = linear_interp(xx, yy, vx, vy, x_current, y_current, maxdist)
            for i in range(extend):
                x_next, y_next = stride_move(x_current, y_current, vx_old, vy_old, stride, direction, mode)
                x_current = x_next
                y_current = y_next

                dist = np.sqrt((x_next-x_current)**2+(y_next-y_current)**2)
                points.append([x_current, y_current])
                
                x_current = x_next
                y_current = y_next
                cum_dist += dist
                cum_dist_coll.append(cum_dist)
            break
        else:
            x_next, y_next = stride_move(x_current, y_current, vx1, vy1, stride, direction, mode)
            
            dist = np.sqrt((x_next-x_current)**2+(y_next-y_current)**2)
            points.append([x_current, y_current])
            
            x_current = x_next
            y_current = y_next
            cum_dist += dist
            cum_dist_coll.append(cum_dist)
            n_iter += 1

    return np.array(points), np.array(cum_dist_coll)

def stride_move(x0, y0, vx0, vy0, stride, direction, mode):
    """
    Move to next point according to stride, direction, and mode.
    """
    if mode=='distance':
        comp_stride = np.sqrt(stride**2/(vx0**2+vy0**2))
        x_move = vx0*comp_stride
        y_move = vy0*comp_stride
    elif mode=='time':
        x_move = vx0*stride
        y_move = vy0*stride
    else:
        raise ValueError('mode must be distance or time')
        
    if direction=='forward':
        x_next = x0 + x_move
        y_next = y0 + y_move
    elif direction=='backward':
        x_next = x0 - x_move
        y_next = y0 - y_move

    return x_next, y_next

def _check_params(xx, yy, vx, vy, x0, y0, stride, total_dist, maxdist, direction, mode, max_iter, extend):
    """
    Check arguments and raise errors.
    """

    # check arrays
    for arr in [xx, yy, vx, vy]:
        if isinstance(arr, np.ndarray) == False:
            raise ValueError('xx, yy, vx, and vy must all be 2D numpy arrays')
    for p0 in [x0, y0, stride, total_dist, maxdist, extend]:
        if isinstance(p0, Number) == False:
            print(f'{p0} not a number')
            raise ValueError('x0, y0, stride, total_dist, maxdist, and extend must be numbers')

    for p in [stride, total_dist, maxdist, extend]:
        if p < 0:
            raise ValueError('stride, total_dist, maxdist, max_iter, and extend must be positive')

    if direction not in ['forward', 'backward']:
        raise ValueError("direction must be 'forward' or 'backward'")

    if mode not in ['distance', 'time']:
        raise ValueError("mode must be 'distance' or 'time'")
    
    