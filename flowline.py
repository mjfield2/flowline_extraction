import numpy as np
from scipy.interpolate import LinearNDInterpolator

def linear_interp(xx, yy, vx, vy, x0, y0, maxdist):
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