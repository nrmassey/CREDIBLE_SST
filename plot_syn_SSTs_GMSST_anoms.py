#! /usr/bin/env python  
#############################################################################
#
# Program : plot_syn_SSTs_GMSST_anoms.py
# Author  : Neil Massey
# Purpose : Plot a timeseries of the GMSST of the synthetic SSTs
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
from create_HadISST_CMIP5_syn_SSTs import get_syn_sst_filename
from netcdf_file import *
import matplotlib.pyplot as plt
import pyximport, numpy
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
sys.path.append("/Users/Neil/python_lib")
from running_mean import *

#############################################################################

def plot_CMIP5_timeseries(sp0, tos_data, idx, time_data, name, c):
    # plot all of the CMIP5 timeseries data
    if (tos_data[idx,0] < 1e10):
        ldata = tos_data[idx]
        ldata = ldata.reshape([ldata.shape[0],1,1])
#        sm_ldata = running_mean_3D(ldata,10)
#        sm_ldata = sm_ldata.squeeze()
        sm_ldata = ldata.squeeze()
        sp0.plot(time_data, sm_ldata, c+'-', lw=2, zorder=1)
        sp0.text(2101, ldata[-1], name, color=c)

#############################################################################

def plot_synthetic_ssts(sp0, tos_data, time_data):
    # plot the envelope
    sp0.fill_between(time_data, tos_data[0], tos_data[-1], facecolor='r', alpha=0.2, zorder=0)
    for a in range(0, tos_data.shape[0]):
        ldata = tos_data[a]
        sp0.plot(time_data, ldata, 'r-', lw=1, alpha=0.5, zorder=0)

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    eof_year = -1
    monthly = False
    hemi=2
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:f:h:m',
                               ['run_type=', 'ref_start=', 'ref_end=',
                                'eof_year=', 'monthly', 'hemi='])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True
        if opt in ['--hemi', '-h']:
            hemi = int(val)

    # read the (already computed) yearly mean cmip5 anomalies for this RCP scenario
    cmip5_tos_tas_ts_fname = get_gmt_gmsst_anom_ts_fname(run_type, ref_start, ref_end, monthly=monthly)
    fh = netcdf_file(cmip5_tos_tas_ts_fname)
    tos_data = fh.variables["tos"][:]
    tos_data = tos_data.byteswap().newbyteorder()
    fh.close()
    
    # load HadISST data
    hadisst_fname = "/Users/Neil/Coding/CREDIBLE_output/output/HadISST_1899_2010_400/HadISST.2.1.0.0_realisation_dec2010_400_yrmn_anoms.nc"
    fh = netcdf_file(hadisst_fname)
    hadisst_data = fh.variables["sst"][:].byteswap().newbyteorder().squeeze()
    fh.close()
    
    # load synthetic data
    syn_fname = get_syn_sst_filename(run_type, ref_start, ref_end, 6, 2050, 0, 2, monthly)
    syn_fname = syn_fname.replace("a0", "ens")
    syn_fname = syn_fname.replace(".nc", "_yrmn_ts.nc")
    syn_fname = syn_fname.replace("/varmon/", "/yrmns/sst/")
    fh = netcdf_file(syn_fname)
    syn_sst_data = fh.variables["sst"][:].squeeze()
    fh.close()

    # global mean of reference value
    ref_val = 291.3006
    # calculate anomalies
    syn_sst_data = syn_sst_data - ref_val

    # set plot    
    sp0 = plt.subplot(111)
    time_data = numpy.arange(1899,2101)
    # plot the selected CMIP5 timeseries data
    if run_type == "rcp85":
        plot_CMIP5_timeseries(sp0, tos_data, 3, time_data, "MIROC5", 'k')
        plot_CMIP5_timeseries(sp0, tos_data, 75, time_data, "IPSL-CM5-LR", 'g')
        plot_CMIP5_timeseries(sp0, tos_data, 106, time_data, "HadGEM2-ES", 'c')
    elif run_type == "rcp45":
        plot_CMIP5_timeseries(sp0, tos_data, 4, time_data, "MIROC5", 'k')
        plot_CMIP5_timeseries(sp0, tos_data, 54, time_data, "IPSL-CM5-MR", 'g')
        plot_CMIP5_timeseries(sp0, tos_data, 77, time_data, "HadGEM2-ES", 'c')
    
    # plot the HadISST data
    hadisst_x = numpy.arange(1850,2011)
    sp0.plot(hadisst_x, hadisst_data, 'b-', alpha=1.0, lw=2, zorder=2)
    sp0.text(hadisst_x[-1]+2, hadisst_data[-1], "HadISST2", color='b')

    # plot the synthetic variables
    time_data = numpy.arange(1899, 2101)
    plot_synthetic_ssts(sp0, syn_sst_data, time_data)
    
    # set plot variables
    sp0.set_xlim([1900,2099])
    sp0.set_xlabel("Year")
    sp0.set_ylabel("Anomaly from 1986 to 2005 mean, $^\circ$C")
    sp0.set_title("Anomalies in GMSST for synthetic ensemble, " + run_type.upper())
    outname = "synth_gmsst_ts_"+run_type+".pdf"

    plt.savefig(outname)
    print outname
