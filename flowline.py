import numpy as np
from scipy.interpolate import LinearNDInterpolator

def linear_interp(xx, yy, vx, vy, x0, y0, maxdist):
    """
    2D linear interpolation of velocity field

    Args:
        xx : 2D x-coordinates, result of np.meshgrid
        yy : 2D y-coordinates, result of np.meshgrid
        vx : 2D x-component of velocity field
        vy : 2D y-component of velocity field
        x0 : x-coordinate to interpolate
        y0 : y-coordinate to interpolate
        maxdist : maximum distance to look for interpolating
    Outputs:
        interpolated x velocity and y velocity
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

def flowline(xx, yy, vx, vy, x0, y0, stride, total_dist, maxdist=5e3, direction='forward', mode='distance', max_iter=1000):
    """
    Extract profile points along flowline.

    Args:
        xx : 2D x-coordinates, result of np.meshgrid
        yy : 2D y-coordinates, result of np.meshgrid
        vx : 2D x-component of velocity field
        vy : 2D y-component of velocity field
        x0 : starting x-coordinate
        y0 : starting y-coordinate
        stride : how far to move in time or distance, corresponding to mode
        total_dist : total distance to move
        maxdist : maximum distance to look for interpolating velocity
        direction : move forward or backward from starting point
        mode : distance or time for movement. Time results in
            points spaced scaled by velocity and distance results in
            points spaced equally in space
        max_iter : maximum number of points. This can be helpful when
            using time since a while loop is used
    Outputs:
        Array of points with x-coordinate in first column and y-coordinate in second
        and cumulative distances along profile
    """
    
    vx0, vy0 = linear_interp(xx, yy, vx, vy, x0, y0, maxdist)

    cum_dist = 0
    n_iter = 0
    x_current = x0
    y_current = y0
    points = [[x0, y0]]
    cum_dist_coll = [0]
    while (cum_dist < total_dist) & (n_iter < max_iter):
        vx1, vy1 = linear_interp(xx, yy, vx, vy, x_current, y_current, maxdist)
        if np.any(np.isnan([vx1, vy1]))==True:
            break
        if mode=='distance':
            comp_stride = np.sqrt(stride**2/(vx1**2+vy1**2))
            x_move = vx1*comp_stride
            y_move = vy1*comp_stride
        elif mode=='time':
            x_move = vx1*stride
            y_move = vy1*stride
        else:
            raise ValueError('mode must be distance or time')
            
        if direction=='forward':
            x_next = x_current + x_move
            y_next = y_current + y_move
        elif direction=='backward':
            x_next = x_current - x_move
            y_next = y_current - y_move
        
        dist = np.sqrt((x_next-x_current)**2+(y_next-y_current)**2)
        points.append([x_current, y_current])
        
        x_current = x_next
        y_current = y_next
        cum_dist += dist
        cum_dist_coll.append(cum_dist)
        n_iter += 1

    return np.array(points), np.array(cum_dist_coll)