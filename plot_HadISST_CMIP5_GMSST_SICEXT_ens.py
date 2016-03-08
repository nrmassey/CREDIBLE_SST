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
# Notes   : 
# Output  : 
# Date    : 27/11/15
#
#############################################################################

import sys,os,getopt
sys.path.append("/Users/Neil/Coding/python_lib")
sys.path.append("../CREDIBLE_SIC/")
from create_HadISST_CMIP5_syn_SSTs import get_syn_sst_filename
from create_HadISST_CMIP5_syn_SIC import get_syn_sic_filename
from cmip5_functions import load_data, load_sst_data, reconstruct_field, calc_GMSST
from calc_sea_ice_extent import *
import numpy
import matplotlib.pyplot as plt
from netcdf_file import *
from py_running_gradient_filter import *
from create_HadISST_sst_anoms import get_HadISST_input_filename
import matplotlib.gridspec as gridspec

#############################################################################

neofs=6
eof_year=2050
ivm=2
samples = numpy.arange(10,100,10, 'f')

#############################################################################

def save_ens_ts_file(out_fname, out_data, samples, dates, varname):
    # open the file
    out_fh = netcdf_file(out_fname, "w")
    # create dimensions
    hemi_out_dim = out_fh.createDimension("hemisphere", out_data.shape[0])
    samp_out_dim = out_fh.createDimension("samples", out_data.shape[1])
    time_out_dim = out_fh.createDimension("time", out_data.shape[2])
    hemi_out_var = out_fh.createVariable("hemisphere", 'f', ("hemisphere",))
    samp_out_var = out_fh.createVariable("samples", samples.dtype, ("samples",))
    time_out_var = out_fh.createVariable("time", dates.dtype, ("time",))
    hemi_out_var[:] = numpy.array([0,1], 'f')
    samp_out_var[:] = samples
    time_out_var[:] = dates
    data_out_var = out_fh.createVariable(varname, out_data.dtype, ("hemisphere", "samples", "time"))
    data_out_var[:] = out_data[:]
    out_fh.close()    

#############################################################################

def calc_syn_GMSST_TS(run_type, ref_start, ref_end, monthly=True):
    out_name = "./" + run_type + "_gmsst_ts.nc"
    dates = numpy.array([1899 + float(x)/12 for x in range(0, 2424)], 'f')
    if not os.path.exists(out_name):
        # get the synthetic SST filename
        #
        ens_ts = numpy.zeros([2, samples.shape[0], dates.shape[0]], 'f')
        for a in samples:
            syn_sst_fname = get_syn_sst_filename(run_type,ref_start,ref_end,neofs,eof_year,int(a),ivm,monthly)
            sst_data = load_sst_data(syn_sst_fname, "sst")
            gmsst_nh = calc_GMSST(sst_data[:,:90,:],1)
            gmsst_sh = calc_GMSST(sst_data[:,90:,:],2)
            ens_ts[0,(a-samples[0])/10] = gmsst_nh
            ens_ts[1,(a-samples[0])/10] = gmsst_sh
        # save the timeseries
        save_ens_ts_file(out_name, ens_ts, samples, dates, "gmsst")
    else:
        ens_ts = load_data(out_name, "gmsst")
    
    return ens_ts.squeeze(), dates

#############################################################################

def calc_syn_SICEXT_TS(run_type, ref_start, ref_end, monthly=True):
    out_name = "./" + run_type + "_sicext_ts.nc"
    dates = numpy.array([1899 + float(x)/12 for x in range(0, 2412)], 'f')
    lats = numpy.array([90-x for x in range(0,180)], 'f')
    d_lon = 1.0
    mv = -1e30
    if not os.path.exists(out_name):
        # get the synthetic SIC filename
        #
        ens_ts = numpy.zeros([2, samples.shape[0], dates.shape[0]], 'f')
        for a in samples:
            syn_sic_fname = get_syn_sic_filename(run_type,ref_start,ref_end,neofs,eof_year,int(a),ivm,monthly)
            sic_data = load_sst_data(syn_sic_fname, "sic")
            sic_ext_nh, sic_ext_sh = calc_sea_ice_extent(sic_data, lats, d_lon, mv)
            ens_ts[0,(a-samples[0])/10] = sic_ext_nh
            ens_ts[1,(a-samples[0])/10] = sic_ext_sh
        # save the timeseries
        save_ens_ts_file(out_name, ens_ts, samples, dates, "sic")
    else:
        ens_ts = load_data(out_name, "sic")
    
    return ens_ts.squeeze(), dates

#############################################################################

def calc_HadISST_GMSST_TS():
    in_fname = get_HadISST_input_filename(400)
    hadisst = load_sst_data(in_fname, "sst")
    dates = numpy.array([1850 + float(x)/12 for x in range(0, hadisst.shape[0])], 'f')
    gmsst_hadisst_nh = calc_GMSST(hadisst[:,:90,:],1)
    gmsst_hadisst_sh = calc_GMSST(hadisst[:,90:,:],2)
    return gmsst_hadisst_nh, gmsst_hadisst_sh, dates

#############################################################################

def calc_HadISST_GMSST_SICEXT():
    in_fname = get_HadISST_input_filename(400)
    hadisst = load_sst_data(in_fname, "sic")
    dates = numpy.array([1850 + float(x)/12 for x in range(0, hadisst.shape[0])], 'f')
    lats = numpy.array([90-x for x in range(0,180)], 'f')
    d_lon = 1.0
    mv = -1.0e30
    sicext_hadisst_nh, sicext_hadisst_sh = calc_sea_ice_extent(hadisst,lats, d_lon,mv)
    return sicext_hadisst_nh, sicext_hadisst_sh, dates

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    hemi = 0
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:f:h:',
                               ['run_type=', 'ref_start=', 'ref_end=',
                                'hemi'])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--hemi', '-h']:
            hemi = int(val)
            
    ens_ts, dates = calc_syn_GMSST_TS(run_type, ref_start, ref_end, True)
    ens_ts = ens_ts.byteswap().newbyteorder()
    ens_sic_ts, dates_sic = calc_syn_SICEXT_TS(run_type, ref_start, ref_end, True)
    hadisst_nh, hadisst_sh, hadisst_dates = calc_HadISST_GMSST_TS()
    hadsic_nh, hadsic_sh, hadisst_dates = calc_HadISST_GMSST_SICEXT()
    # create the northern hemisphere plot
    plot_month=False
    if plot_month:
        fig = plt.figure()
        gs = gridspec.GridSpec(6,2)
        crow = 0
        ccol = 0
        sic_min = numpy.min(ens_sic_ts)
        sic_max = numpy.max(ens_sic_ts)
        sst_min = numpy.min(ens_ts)
        sst_max = numpy.max(ens_ts)
        for m in range(0,12):   # plot all twelve months
            # create subplot
            sp0 = plt.subplot(gs[crow,ccol])
            sp1 = sp0.twinx()
            # plot the ensemble
            for a in range(0, ens_ts.shape[1]):
                sp0.plot(dates[m::12], ens_ts[hemi,a,m::12], 'r')
                sp1.plot(dates_sic[m::12], ens_sic_ts[hemi,a,m::12], 'b')
            # plot hadisst
            if hemi == 0:
                sp0.plot(hadisst_dates[m::12], hadisst_nh[m::12], 'k', lw=2.0)
            else:
                sp0.plot(hadisst_dates[m::12], hadisst_sh[m::12], 'k', lw=2.0)
            if crow != 5:
                sp0.axes.get_xaxis().set_visible(False)
            if ccol == 0:
                sp1.axes.get_yaxis().set_visible(False)
            if ccol == 1:
                sp0.axes.get_yaxis().set_visible(False)
            sp0.set_ylim([sst_min, sst_max])
            sp1.set_ylim([sic_min, sic_max])
            sp0.set_xlim([dates[m::12][0], dates[m::12][-1]])
            crow += 1
            if crow == 6:
                ccol = 1
                crow = 0
    else:
        fig = plt.figure()
        sp0 = plt.subplot(111)
        sp1 = sp0.twinx()
        S = ens_ts.shape
        ens_sst_yr_mean = numpy.zeros([S[0], S[1], S[2]/12], 'f')
        ens_sic_yr_mean = numpy.zeros([S[0], S[1], S[2]/12], 'f')
        new_dates = dates[6::12]
        H = hadisst_nh.shape
        hadisst_nh_yr_mn = numpy.zeros([H[0]/12], 'f')
        hadisst_sh_yr_mn = numpy.zeros([H[0]/12], 'f')
        hadsic_nh_yr_mn = numpy.zeros([H[0]/12], 'f')
        hadsic_sh_yr_mn = numpy.zeros([H[0]/12], 'f')


        for y in range(0, S[2], 12):
            ens_sst_yr_mean[:,:,y/12] = numpy.mean(ens_ts[:,:,y:y+12], axis=2)
            ens_sic_yr_mean[:,:,y/12] = numpy.mean(ens_sic_ts[:,:,y:y+12], axis=2)
            
        for y in range(0, H[0], 12):
            hadisst_nh_yr_mn[y/12] = numpy.mean(hadisst_nh[y:y+12])
            hadisst_sh_yr_mn[y/12] = numpy.mean(hadisst_sh[y:y+12])
            hadsic_nh_yr_mn[y/12] = numpy.mean(hadsic_nh[y:y+12])
            hadsic_sh_yr_mn[y/12] = numpy.mean(hadsic_sh[y:y+12])

        sst_min = numpy.min(ens_sst_yr_mean, axis = 1)
        sst_max = numpy.max(ens_sst_yr_mean, axis = 1)
        
        sic_min = numpy.min(ens_sic_yr_mean, axis = 1)
        sic_max = numpy.max(ens_sic_yr_mean, axis = 1)
        
        print sst_max[hemi].shape, sst_min[hemi].shape, new_dates.shape
        
        sp0.fill_between(new_dates, sst_min[hemi], sst_max[hemi], facecolor='r', alpha=0.5, zorder=0)
        sp1.fill_between(new_dates, sic_min[hemi], sic_max[hemi], facecolor='b', alpha=0.5, zorder=0)
            
        for a in range(0, ens_ts.shape[1]):
            sp0.plot(new_dates, ens_sst_yr_mean[hemi,a], 'r', zorder=1)
            sp1.plot(new_dates, ens_sic_yr_mean[hemi,a], 'b', zorder=1)
            
        if hemi == 0:
            sp0.plot(hadisst_dates[6::12], hadisst_nh_yr_mn, 'k', lw=2.0, zorder=1)
            sp1.plot(hadisst_dates[6::12], hadsic_nh_yr_mn, 'k', lw=2.0, zorder=1)
        else:
            sp0.plot(hadisst_dates[6::12], hadisst_sh_yr_mn, 'k', lw=2.0, zorder=1)
            sp1.plot(hadisst_dates[6::12], hadsic_sh_yr_mn, 'k', lw=2.0, zorder=1)
    sp0.set_xlim([1850,2100])
    sp1.set_xlim([1850,2100])
    
    fig.set_size_inches(10,5)
    if hemi == 0:
        name = "NH_SST_SIC.png"
    else:
        name = "SH_SST_SIC.png"

    plt.savefig(name)