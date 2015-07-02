#cython: boundscheck=False
#cython: wraparound=True

#############################################################################
#
# Program : zonal_smoother.pyx
# Author  : Neil Massey
# Purpose : Smooth in the zonal distance (along longitudes) a 3D field
#           The width of the smoothing window is a function of the latitude
#           Input parameters are:
#               data  - field to be smoothed (numpy array)
#               lats  - vector of latitudes
#               power - power to raise cos(theta) by when calculating the
#                       number of grid boxes to smooth over
#               ww    - max width of smoothing window
#               mv    - missing data value (smoothed data does not take into
#                        account missing data)
# Date    : 25/06/15
#
#############################################################################

cimport numpy
import numpy

cpdef numpy.ndarray[float, ndim=3] zonal_smoother(numpy.ndarray[float, ndim=3] data, 
                                                  numpy.ndarray[float, ndim=1] lats,
                                                  int power, int ww, float mv=2e20):
    # create the storage for the smoothed output
    cdef nt = data.shape[0]         # number of t points
    cdef ny = data.shape[1]         # number of y points
    cdef nx = data.shape[2]         # number of x points
    
    cdef numpy.ndarray[float, ndim=3] out_data = numpy.zeros([nt, ny, nx], 'f')
    
    cdef int t, x, y, i             # indices into the array
    cdef int i0,i1                  # left / right indices
    cdef float sum, sum_w           # sum / sum of weights
    
    # calculate the bin widths
    cdef numpy.ndarray[int, ndim=1] bin_w = numpy.zeros([ny], 'i')
    bin_w = (0.5*(numpy.cos(numpy.radians(lats[0]-lats))**power)*ww).astype('i')
    
    # loop over the data array
    for t in range(0, nt):
        for y in range(0, ny):
            tww = bin_w[y]
            for x in range(0, nx):
                # skip if this box contains missing values
                if data[t,y,x] == mv:
                    out_data[t,y,x] = mv
                    continue
                if tww == 0:
                    out_data[t,y,x] = data[t,y,x]
                else:
                    i0 = x-tww
                    i1 = x+tww
                    sum = 0.0
                    sum_w = 0.0
                    for i in range(i0, i1+1):
                        if data[t,y,i] != mv:
                            sum += data[t,y,i]
                            sum_w += 1
                    if sum_w != 0.0:
                        out_data[t,y,x] = sum / sum_w
                    else:
                        out_data[t,y,x] = mv
            
    return out_data 