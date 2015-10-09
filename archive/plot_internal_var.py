#! /usr/bin/env python  
#############################################################################
#
# Program : plot_internal_var.py
# Author  : Neil Massey
# Purpose : Plot the timeseries of GMSST anomalies achieved by the AR modelling
#           of the residuals obtained by subtracting the smoothed HadISST from
#           the unsmoothed HadISST
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
# Date    : 03/03/15
#
#############################################################################

import os, sys, getopt

from create_HadISST_CMIP5_syn_SSTs import create_yearly_intvar, create_monthly_intvar
from create_HadISST_sst_anoms import get_HadISST_residuals_fname, get_HadISST_monthly_residuals_fname, get_HadISST_annual_cycle_residuals_fname
from create_CMIP5_sst_anoms import get_start_end_periods
from cmip5_functions import calc_GMSST, load_sst_data
import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#############################################################################

def plot_yearly_var(run_type, ref_start, ref_end, n_samps):
    # create the storage
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    n_yrs = rcp_ey - histo_sy + 1
    gmsst = numpy.zeros([n_samps, n_yrs], 'f')
    
    for a in range(0, n_samps):
        # create the yearly internal variability for a samples
        yearly_var = create_yearly_intvar(run_type, ref_start, ref_end)
        # take the GMSST of the field
        gmsst[a] = calc_GMSST(yearly_var)
        
    # now plot
    t_var = numpy.arange(histo_sy, rcp_ey+1, 1.0)
    gs = gridspec.GridSpec(1,2)
    sp = plt.subplot(gs[0,:])
    for a in range(0, n_samps):
        sp.plot(t_var, gmsst[a], '#6666DD', lw=0.5, alpha=1.0, zorder=1)
    sp.set_xlim(histo_sy, rcp_ey)
    
    # plot the HadISST internal variability
    hadisst_fname = get_HadISST_residuals_fname(histo_sy,2010)
    hadisst_ssts = load_sst_data(hadisst_fname, "sst")
    hadisst_gmsst = calc_GMSST(hadisst_ssts)
    hadisst_t = numpy.arange(histo_sy, 2010+1, 1.0)
    sp.plot(hadisst_t, hadisst_gmsst, '#0000EE', lw=2.0, alpha=1.0, zorder=2)

    sp.set_ylabel("GMSST $^\circ$C")
    sp.set_xlabel("Year")
    
    f = plt.gcf()
    f.set_size_inches(15, 5)

    plt.savefig("internal_variability_yearly.pdf")
    
#############################################################################

def plot_monthly_var(run_type, ref_start, ref_end, n_samps, ac=False):
    monthly_var = create_monthly_intvar(run_type, ref_start, ref_end)
    # create the storage
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    n_mnths = (rcp_ey - histo_sy + 1) * 12
    gmsst = numpy.zeros([n_samps, n_mnths], 'f')
    
    for a in range(0, n_samps):
        # create the yearly internal variability for a samples
        monthly_var = create_monthly_intvar(run_type, ref_start, ref_end, ac=ac)
        # take the GMSST of the field
        gmsst[a] = calc_GMSST(monthly_var)
        
    # now plot
    t_var = numpy.arange(histo_sy, rcp_ey+1, 1.0/12)
    gs = gridspec.GridSpec(1,2)
    sp = plt.subplot(gs[0,:])
    for a in range(0, n_samps):
        sp.plot(t_var, gmsst[a], '#6666DD', lw=0.5, alpha=1.0, zorder=1)
    sp.set_xlim(1990, 2010)
    
    # plot the HadISST internal variability
    hadisst_fname = get_HadISST_monthly_residuals_fname(histo_sy,2010)
    hadisst_ssts = load_sst_data(hadisst_fname, "sst")
    
    if not ac:
        # load and subtract the annual cycle
        hadisst_ac_fname = get_HadISST_annual_cycle_residuals_fname(histo_sy, 2010, ref_start, ref_end)
        hadisst_ac = load_sst_data(hadisst_ac_fname, "sst")
        hadisst_ac = numpy.tile(hadisst_ac, [hadisst_ssts.shape[0]/12,1,1])
        hadisst_ssts = hadisst_ssts - hadisst_ac
        
    hadisst_gmsst = calc_GMSST(hadisst_ssts)
    hadisst_t = numpy.arange(histo_sy, 2010+1, 1.0/12)
    sp.plot(hadisst_t, hadisst_gmsst, '#0000EE', lw=2.0, alpha=1.0, zorder=2)

    sp.set_ylabel("GMSST $^\circ$C")
    sp.set_xlabel("Year")
    
    f = plt.gcf()
    f.set_size_inches(15, 5)
    if ac:
        outname = "internal_variability_monthly_ac.pdf"
    else:
        outname = "internal_variability_monthly.pdf"

    plt.savefig(outname)

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    sample = 100
    intvarmode = 0      # internal variability mode - 0 = none, 1 = yearly, 2 = monthly
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:i:a:',
                               ['run_type=', 'ref_start=', 'ref_end=', 'intvarmode=', 'samples='])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--intvar', '-i']:
            intvarmode = int(val)
        if opt in ['--samples', '-a']:
            sample = int(val)
    
    if intvarmode == 1:
        plot_yearly_var(run_type, ref_start, ref_end, sample)
    elif intvarmode == 2:
        plot_monthly_var(run_type, ref_start, ref_end, sample)