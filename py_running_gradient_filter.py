#cython: boundscheck=False
#cython: wraparound=False

#############################################################################
#
# Program : running_gradient_filter.pyx
# Author  : Neil Massey
# Purpose : Calculate a running mean (moving average) on a 3D field of data
#           taking into account missing value.
#           The running mean is calculated by calculating a and b in the line
#           equation y = a + bx using least squares linear regression over a
#           window of values and then reconstructing the values using the line
#           equation
# Date    : 09/02/15
#
#############################################################################

import numpy

#############################################################################

def calc_a_b(sum_x, sum_y, sum_xy, sum_xx, N):
    mean_x = 0.0
    mean_xx = 0.0
    mean_y = 0.0
    D = 0.0
    
    mean_x = sum_x / N
    mean_xx = mean_x*mean_x
    mean_y = sum_y / N
    D = (sum_xx - N*mean_xx)

    a = (mean_y*sum_xx - mean_x*sum_xy) / D
    b = (sum_xy - N*mean_x*mean_y) / D
    return a,b

#############################################################################

def running_gradient_3D(data, period, mv=2e20):
    W = period/2           # window width
    N = W                  # number of samples (current)
    nt = data.shape[0]         # number of t points
    ny = data.shape[1]         # number of y points
    nx = data.shape[2]         # number of x points
    # output array
    out_data = numpy.zeros([nt, ny, nx], 'f')

    # variables for the summing of x,y, xy, x^2 and y^2
    sum_x  = 0.0
    sum_y  = 0.0
    sum_xy = 0.0
    sum_xx = 0.0
    
    # loop over the data array
    for y in range(0, ny):
        for x in range(0, nx):
            # skip if this box contains missing values
            if data[0,y,x] == mv:
                out_data[:,y,x] = mv
                continue
            # reset sums for this grid box
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_xx = 0.0
            N = W
            # create the first value by calculating the sums up to W
            for t in range(0, W):
                sum_x  += t
                sum_y  += data[t,y,x]
                sum_xy += t*data[t,y,x]
                sum_xx += t*t

            # calculate a and b
            a, b = calc_a_b(sum_x, sum_y, sum_xy, sum_xx, N)
            # do the first projection
            out_data[0,y,x] = a + b
            
            # calculate the values between 1 and W - just add more data
            for t in range(0, W):
                N += 1
                T1 = t+W
                sum_x  += T1
                sum_y  += data[T1,y,x]
                sum_xy += T1*data[T1,y,x]
                sum_xx += T1*T1
                
                # calculate a and b and reconstruct the value
                a, b = calc_a_b(sum_x, sum_y, sum_xy, sum_xx, N)
                out_data[t,y,x] = a + t*b
                
            # calculate the values between W and nt-W - subtract the last value and add then next
            for t in range(W, nt-W):
                T0 = t-W
                T1 = t+W
                sum_x  = sum_x - T0 + T1
                sum_y  = sum_y - data[T0,y,x] + data[T1,y,x]
                sum_xy = sum_xy - T0*data[T0,y,x] + T1*data[T1,y,x]
                sum_xx = sum_xx - T0*T0 + T1*T1
                
                # calc a and b and reconstruct
                a, b = calc_a_b(sum_x, sum_y, sum_xy, sum_xx, N)
                out_data[t,y,x] = a + t*b
                
            # finally calculate the values between nt-W and nt - just subtract the last values
            # and decrement N
            for t in range(nt-W, nt):
                N -= 1
                T0 = t-W
                sum_x  = sum_x - T0
                sum_y  = sum_y - data[T0,y,x]
                sum_xy = sum_xy - T0*data[T0,y,x]
                sum_xx = sum_xx - T0*T0
                
                # calc a and b and reconstruct
                a, b = calc_a_b(sum_x, sum_y, sum_xy, sum_xx, N)
                out_data[t,y,x] = a + t*b
    return out_data
