#! /usr/bin/env python  
#############################################################################
#
# Program : plot_CMIP5_reconstruction.py
# Author  : Neil Massey
# Purpose : Plot a timeseries of the GMSST of the CMIP5 ensemble plus the
#           reconstructed field of the EOFs and PC projections
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
from create_CMIP5_sst_anoms import get_concat_anom_sst_smooth_fname, get_concat_anom_sst_ens_mean_smooth_fname, get_start_end_periods
from create_HadISST_sst_anoms import get_HadISST_reference_fname, get_HadISST_smooth_fname, get_HadISST_residuals_fname, get_HadISST_monthly_residuals_fname, get_HadISST_annual_cycle_residuals_fname
from calc_CMIP5_EOFs import *
from cmip5_functions import calc_GMSST, load_data, reconstruct_field, load_sst_data
from filter_cmip5_members import read_cmip5_index_file
from create_HadISST_CMIP5_syn_SSTs import get_syn_sst_filename
from create_GMT_GMSST_anom_ts import get_gmt_gmsst_ts_fname

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from netcdf_file import *
import numpy

import pyximport
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
from running_gradient_filter import *

#############################################################################

def plot_likely_range(sp, ref_start, ref_end, run_n="mm"):
    # plot the AR5 fig 11.25 likely range
    # first calc in GMT
    Y0 = 2009.0
    Y1 = 2025.5

    grad0 = (0.3-0.16)/(Y1 - Y0)
    grad1 = (0.7-0.16)/(Y1 - Y0)

    gmt_min0 = grad0*(2016-Y0) + 0.16 - 0.1
    gmt_max0 = grad1*(2016-Y0) + 0.16 + 0.1
    gmt_min1 = grad0*(2035-Y0) + 0.16 - 0.1
    gmt_max1 = grad1*(2035-Y0) + 0.16 + 0.1

    # convert to gmsst using the values of slope and intercept computed
    # by regressing the tos anomaly onto tas anomaly in CMIP5 ensemble members
    
    slope = 0.669
    intercept = 0.017
    
    # get the reference value
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010
    hadisst_ref_fname = get_HadISST_reference_fname(histo_sy, hadisst_ey, ref_start, ref_end, run_n)
    hadisst_ref_sst = load_sst_data(hadisst_ref_fname, "sst")
    ref_gmsst = calc_GMSST(hadisst_ref_sst) -273.15
    
    gmsst_min0 = gmt_min0 * slope + intercept + ref_gmsst
    gmsst_max0 = gmt_max0 * slope + intercept + ref_gmsst
    gmsst_min1 = gmt_min1 * slope + intercept + ref_gmsst
    gmsst_max1 = gmt_max1 * slope + intercept + ref_gmsst

    l = sp.plot([2016,2035,2035,2016,2016],[gmsst_max0,gmsst_max1,gmsst_min1,gmsst_min0,gmsst_max0], 'k', lw=2.0, zorder=4)
    sp.plot([2016,2035], [(gmsst_max0+gmsst_min0)*0.5, (gmsst_max1+gmsst_min1)*0.5], 'k', lw=2.0, zorder=4)
    return l

#############################################################################

def plot_CMIP5_GMSST(sp, run_type, ref_start, ref_end, skip=1):
    # load the timeseries files
    if run_type == "likely":
        load_run_type = "rcp45"
    else:
        load_run_type = run_type
    out_name = get_gmt_gmsst_ts_fname(load_run_type, ref_start, ref_end)
    cmip5_tos = load_data(out_name, "tos")
    cmip5_tos = cmip5_tos.byteswap().newbyteorder()
    n_ens = cmip5_tos.shape[0]
    
    # load the reference file
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010
    hadisst_ref_fname = get_HadISST_reference_fname(histo_sy, hadisst_ey, ref_start, ref_end, "mm")
    hadisst_ref_sst = load_sst_data(hadisst_ref_fname, "sst")
    ref_gmsst = calc_GMSST(hadisst_ref_sst) - 273.15
    cmip5_tos += ref_gmsst
    
    passed_data = []
    for idx in range(0, n_ens, skip):
        if cmip5_tos[idx,0] < 1000:
            passed_data.append(cmip5_tos[idx])
    cmip5_tos = numpy.array(passed_data)
    n_ens = cmip5_tos.shape[0]
    
    # create the time axis
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    t_var = numpy.arange(histo_sy, rcp_ey+1)
    
    # smooth the data first
    sm_data = numpy.ones(cmip5_tos.shape, cmip5_tos.dtype) * 1e20
    for idx in range(0, n_ens, skip):
        sm_data[idx] = running_gradient_3D(cmip5_tos[idx].reshape([cmip5_tos[idx].shape[0],1,1]),40).squeeze()

    # loop over each ensemble member
    min = numpy.min(sm_data, axis=0)
    max = numpy.max(sm_data, axis=0)
    C = '#008888'
    sp.plot(t_var, min, C, lw=1.0, alpha=1.0, zorder=1)
    sp.plot(t_var, max, C, lw=1.0, alpha=1.0, zorder=1)
    sp.fill_between(t_var, min, max, facecolor=C, edgecolor=C, alpha=0.5, zorder=0)
    for idx in range(0, n_ens, skip):
        # smooth the data first
        if sm_data[idx,0] < 1000:
            l0 = sp.plot(t_var, sm_data[idx], C, lw=0.5, alpha=0.5, zorder=0)
        if idx == 0:
            l = l0
    return l

#############################################################################

def plot_syn_GMSST(sp, run_type, ref_start, ref_end, neofs, eof_year, varmode, skip=1):
    # create the time axis
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    if varmode == 2:
        t_var = numpy.arange(histo_sy, rcp_ey+1, 1.0/12)
    else:
        t_var = numpy.arange(histo_sy, rcp_ey+1)

    # create the storage so that we can create an envelope of min / max values
    n_ens = 100
    gmsst_vals = numpy.zeros([n_ens, t_var.shape[0]], 'f')
    # plot the reconstructed SSTs for all the samples
    for a in range(0, n_ens, skip):
        fname = get_syn_sst_filename(run_type, ref_start, ref_end, neofs, eof_year, a, varmode)
        sst_data = load_sst_data(fname, "sst")
        gmsst_vals[a] = calc_GMSST(sst_data) - 273.15

    min = numpy.min(gmsst_vals[::skip], axis=0)
    max = numpy.max(gmsst_vals[::skip], axis=0)
    C = '#FF0000'
    sp.plot(t_var, min, C, lw=1.0, alpha=1.0, zorder=3)
    sp.plot(t_var, max, C, lw=1.0, alpha=1.0, zorder=3)
    sp.fill_between(t_var, min, max, facecolor=C, edgecolor=C, alpha=0.5, zorder=2)
    for a in range(0, n_ens, skip):
        if a == 49 or a==4 or a==94:
            lw = 1.5
            C0 = '#440000'
        else:
            lw = 0.5
            C0 = '#AA0000'
        l = sp.plot(t_var, gmsst_vals[a], C0, lw=lw, alpha=1.0, zorder=1)
    return l
    
#############################################################################

def plot_HadISST2(sp, resids=False, run_n="mm"):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_fname = get_HadISST_smooth_fname(histo_sy, 2010, run_n)
    hadisst_data = load_sst_data(hadisst_fname, "sst")
    if resids:
        hadisst_resids_fname = get_HadISST_residuals_fname(histo_sy, 2010, run_n)
        hadisst_resids = load_sst_data(hadisst_resids_fname, 'sst')
    else:
        hadisst_resids = 0
    t_var = numpy.arange(histo_sy, histo_sy+hadisst_data.shape[0])
    gmsst = calc_GMSST(hadisst_data+hadisst_resids) - 273.15
    l = sp.plot(t_var, gmsst, 'k', lw=1.5, alpha=1.0, zorder=5)
    return l

#############################################################################

def plot_HadISST2_monthly(sp, resids=False, run_n="mm"):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_fname = get_HadISST_smooth_fname(histo_sy, 2010, run_n)
    hadisst_data = load_sst_data(hadisst_fname, "sst")
    if resids:
        hadisst_resids_fname = get_HadISST_monthly_residuals_fname(histo_sy, 2010, run_n)
        hadisst_resids = load_sst_data(hadisst_resids_fname, 'sst')
    else:
        hadisst_resids = 0
    t_var = numpy.arange(histo_sy, histo_sy+hadisst_data.shape[0], 1.0/12)
    hadisst_monthly = numpy.repeat(hadisst_data,12,axis=0)
    gmsst = calc_GMSST(hadisst_monthly+hadisst_resids) - 273.15
    l = sp.plot(t_var, gmsst, 'k', lw=1.5, alpha=1.0, zorder=2)
    return l

#############################################################################

def plot_syn_GMSST_grid_box(sp, run_type, ref_start, ref_end, neofs, eof_year, varmode, skip=1):
    # create the time axis
    lonp = 145
    latp = 32
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    if varmode == 2:
        t_var = numpy.arange(histo_sy, rcp_ey+1, 1.0/12)
        
        hadisst_ac_fname = get_HadISST_annual_cycle_residuals_fname(histo_sy, 2010, ref_start, ref_end)
        hadisst_ac = load_sst_data(hadisst_ac_fname, "sst")
        hadisst_ac = numpy.tile(hadisst_ac, [t_var.shape[0]/12,1,1])
    else:
        t_var = numpy.arange(histo_sy, rcp_ey+1)
    # create the storage so that we can create an envelope of min / max values
    n_ens = 100
    gmsst_vals = numpy.zeros([n_ens, t_var.shape[0]], 'f')
    # plot the reconstructed SSTs for all the samples
    for a in range(0, n_ens, skip):
        print a
        fname = get_syn_sst_filename(run_type, ref_start, ref_end, neofs, eof_year, a, varmode)
        sst_data = load_sst_data(fname, "sst")
        if varmode == 2:
            sst_data = sst_data - hadisst_ac
        gmsst_vals[a] =  sst_data[:,latp,lonp] - 273.15
    min = numpy.min(gmsst_vals[::skip], axis=0)
    max = numpy.max(gmsst_vals[::skip], axis=0)
    sp.plot(t_var, min, 'k', lw=1.0, alpha=1.0, zorder=1)
    sp.plot(t_var, max, 'k', lw=1.0, alpha=1.0, zorder=1)
    sp.fill_between(t_var, min, max, facecolor='#AAAAAA', edgecolor='#AAAAAA', alpha=0.5, zorder=0)
    for idx in range(0, n_ens, skip):
        l = sp.plot(t_var, gmsst_vals[idx], 'c', lw=0.5, alpha=1.0, zorder=0)
    return l

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    neofs = 0
    eof_year = 2050
    varintmode = 0
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:n:f:i:k:',
                               ['run_type=', 'ref_start=', 'ref_end=', 'neofs=', 'eof_year=',
                                'varint=', 'skip='])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--neofs', '-n']:
            neofs = int(val)
        if opt in ['--eof_year', '-f']:
            eof_year = int(val)
        if opt in ['--varint', '-i']:
            varintmode = int(val)
        if opt in ['--skip', '-k']:
            skip = int(val)

    if varintmode == 0:
        varmodestring = "varnone"
    elif varintmode == 1:
        varmodestring = "varyear"
    elif varintmode == 2:
        varmodestring = "varmon"

    out_name = "cmip5_"+run_type+"_"+str(ref_start)+"_"+str(ref_end)+"_"+varmodestring+"_recon.pdf"

    gs = gridspec.GridSpec(1,2)
    sp0 = plt.subplot(gs[0:2,:])
    l_synth = plot_syn_GMSST(sp0, run_type, ref_start, ref_end, neofs, eof_year, varintmode, skip)
    l_cmip5 = plot_CMIP5_GMSST(sp0, run_type, ref_start, ref_end, skip)
    if varintmode == 0:
        l_hadisst = plot_HadISST2(sp0, False)
    elif varintmode == 1:
        l_hadisst = plot_HadISST2(sp0, True)
    if varintmode == 2:
        l_hadisst = plot_HadISST2_monthly(sp0, True)
    l_likely = plot_likely_range(sp0, ref_start, ref_end)
    sp0.legend([l_cmip5[0], l_synth[0], l_hadisst[0]], ["CMIP5 ensemble", "Synthetic", "HadISST2"], loc=0)
    sp0.set_ylabel("GMSST $^\circ$C")
    sp0.set_xlabel("Year")
    sp0.set_xlim([1899,2100])
    f = plt.gcf()
    f.set_size_inches(15.0, 5.0)
        
    plt.savefig(out_name)
