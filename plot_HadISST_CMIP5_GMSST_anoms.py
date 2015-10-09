#! /usr/bin/env python  
#############################################################################
#
# Program : plot_HadISST_CMIP5_GMSST_anoms.py
# Author  : Neil Massey
# Purpose : Plot a timeseries of the GMSST of the CMIP5 ensemble plus the HadISST data
# Inputs  : run_type  : rcp4.5 | rc8.5 | histo
#           ref_start : year to start reference period, 1850->2005
#           ref_end   : year to end reference period, 1850->2005
#           year      : year to take warming at as an alternative to warm
# Notes   : all reference values are calculated from the historical run_type
#           CMIP5 ensemble members are only included if their historical run 
#           includes the reference period
#           requires Andrew Dawsons eofs python libraries:
#            http://ajdawson.github.io/eofs/
# Output  : 
# Date    : 12/02/15
#
#############################################################################

import os, sys, getopt
from create_CMIP5_GMT_GMSST_anom_ts import get_gmt_gmsst_anom_ts_fname
from netcdf_file import *
import matplotlib.pyplot as plt
import pyximport, numpy
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
sys.path.append("/Users/Neil/python_lib")
from running_mean import *

#############################################################################

def plot_CMIP5_timeseries(sp0, tos_data, time_data):
    # plot all of the CMIP5 timeseries data
    for i in range(0, tos_data.shape[0]):
        if (tos_data[i,0] < 1e10):
            ldata = tos_data[i]
            ldata = ldata.reshape([ldata.shape[0],1,1])
            sm_ldata = running_mean_3D(ldata,10)
            sm_ldata = sm_ldata.squeeze()
            sp0.plot(time_data, sm_ldata, 'k-', alpha=0.2, lw=1)

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    eof_year = -1
    monthly = False
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:f:m',
                               ['run_type=', 'ref_start=', 'ref_end=',
                                'eof_year=', 'monthly'])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True

    # read the (already computed) yearly mean cmip5 anomalies for this RCP scenario
    cmip5_tos_tas_ts_fname = get_gmt_gmsst_anom_ts_fname(run_type, ref_start, ref_end, monthly=monthly)
    fh = netcdf_file(cmip5_tos_tas_ts_fname)
    tos_data = fh.variables["tos"][:]
    tos_data = tos_data.byteswap().newbyteorder()
    fh.close()
    
    # load HadISST data
    hadisst_fname = "/Users/Neil/Coding/CREDIBLE_output/output/HadISST_1899_2010_400/hadisst_hist_1899_2010_1986_2005_400_anoms_gmsst_decmn.nc"
    fh = netcdf_file(hadisst_fname)
    hadisst_data = fh.variables["sst"][:].byteswap().newbyteorder()
    fh.close()
    
    # plot the CMIP5 timeseries data
    sp0 = plt.subplot(111)
    time_data = numpy.arange(1899,2101)
    plot_CMIP5_timeseries(sp0, tos_data, time_data)
    # plot the HadISST data
    hadisst_x = numpy.arange(1855,2007)
    sm_hadisst_data = running_mean_3D(hadisst_data,10)
    sp0.plot(hadisst_x, hadisst_data.squeeze(), 'r-', alpha=1.0, lw=2)
    sp0.set_xlim([1899,2090])
    sp0.set_xlabel("Year")
    sp0.set_ylabel("Anomaly from 1986 to 2005 mean, $^\circ$C")
    sp0.set_title("Anomalies in GMSST for CMIP5 ensemble, " + run_type.upper())
    plt.savefig("cmip5_gmsst_ts_"+run_type+".png")