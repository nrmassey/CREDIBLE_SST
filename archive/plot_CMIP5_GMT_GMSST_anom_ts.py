#! /usr/bin/env python  
#############################################################################
#
# Program : plot_CMIP5_GMSST_anom_ts.py
# Author  : Neil Massey
# Purpose : Plot a timeseries of the GMSST anomalies of the CMIP5 ensemble 
# Inputs  : run_type  : rcp4.5 | rc8.5 | histo
#           ref_start : year to start reference period, 1850->2005
#           ref_end   : year to end reference period, 1850->2005
#           year      : year to take warming at as an alternative to warm
# Notes   : all reference values are calculated from the historical run_type
#           CMIP5 ensemble members are only included if their historical run 
#           includes the reference period
# Output  : 
# Date    : 17/04/15
#
#############################################################################

import os, sys, getopt
import numpy
import matplotlib.pyplot as plt

from create_CMIP5_GMT_GMSST_anom_ts import get_gmt_gmsst_anom_ts_fname
from create_CMIP5_sst_anoms import get_start_end_periods
from netcdf_file import *

import pyximport
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
from running_gradient_filter import *

#############################################################################

def plot_CMIP5_GMT_anom_ts(run_type, ref_start, ref_end, monthly):
    # load the data
    fname = get_gmt_gmsst_anom_ts_fname(run_type, ref_start, ref_end, monthly)
    fh = netcdf_file(fname, 'r')
    var = fh.variables["tas"]
    fh.close()

#############################################################################

def plot_CMIP5_GMSST_anom_ts(run_type, ref_start, ref_end, monthly,lat,lon):
    # load the data
    fname = get_gmt_gmsst_anom_ts_fname(run_type, ref_start, ref_end, monthly,lat,lon)
    print fname
    fh = netcdf_file(fname, 'r')
    var = fh.variables["tos"]
    sst = var[:]
    m_sst = numpy.ma.masked_greater(sst, 1000)
    ens_mean_sst = numpy.mean(m_sst, axis=0)
    # create the time dimension
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    if monthly:
        t = numpy.arange(histo_sy, rcp_ey+1)
    else:
        t = numpy.arange(histo_sy, rcp_ey+1)
    sp = plt.subplot(111)
    # calculate the yearly mean
    yr_mean = numpy.zeros([ens_mean_sst.shape[0]/12],'f')
    for y in range(0,yr_mean.shape[0]):
        m=y*12
        yr_mean[y] = numpy.mean(ens_mean_sst[m:m+12])
    # plot each month individually
    l = []
    COLS = ["#FF0000", "#444444", "#AAAA00", "#0000FF",
            "#660000", "#000000", "#666600", "#000066",
            "#00FFFF", "#440044", "#FF00FF", "#008800"]
    legends = ["jan", "feb", "mar", "apr", "may", "jun", 
               "jul", "aug", "sep", "oct", "nov", "dec"]
    for m in range(0, 12):
        # get the month data
        D = ens_mean_sst[m::12].byteswap().newbyteorder().astype('float32') - yr_mean
        # apply the smoother
        D = numpy.reshape(D, [D.shape[0],1,1])
        S = running_gradient_3D(D, 40).squeeze()
        l0 = sp.plot(t, S, COLS[m], lw=1.5)
        l.append(l0[0])
        sp.text(t[-1]+1.0, S[-1], legends[m], fontsize=8, color=COLS[m])
    sp.legend(l, legends, loc=6)
    out_name = fname[:-3] + ".png"
    print out_name
    plt.savefig(out_name)
    fh.close()

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    monthly = False
    lat = -1.0
    lon = -1.0
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:a:l:m',
                               ['run_type=', 'ref_start=', 'ref_end=', 
                                'latitude=', 'longitude=', 'monthly'])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True
        if opt in ['--latitude', '-a']:
            lat = float(val)
        if opt in ['--longitude', '-l']:
            lon = float(val)

    plot_CMIP5_GMSST_anom_ts(run_type, ref_start, ref_end, monthly, lat, lon)
#    plot_CMIP5_GMT_anom_ts(run_type, ref_start, ref_end, monthly, lat, lon)