#! /usr/bin/env python  
#############################################################################
#
# Program : calc_HadISST_residual_EOFs.py
# Author  : Neil Massey
# Purpose : Calculate the EOFs of the residuals calculated by subtracting
#           the smoothed yearly mean HadISST data from the actual HadISST
#           data
# Inputs  : run_type  : rcp4.5 | rc8.5 | histo
#           ref_start : year to start reference period, 1850->2005
#           ref_end   : year to end reference period, 1850->2005
#           eof_year  : year to calculate EOFs at - e.g. 2050s
#           n_eofs    : number of EOFs / PCs to use
# Notes   : all reference values are calculated from the historical run_type
#           CMIP5 ensemble members are only included if their historical run 
#           includes the reference period
#           requires Andrew Dawsons eofs python libraries:
#            http://ajdawson.github.io/eofs/
# Output  : in the output/ directory filename is:
#            
# Date    : 26/02/15
#
#############################################################################

import os, sys, getopt
from create_HadISST_sst_anoms import *
from create_CMIP5_sst_anoms import get_start_end_periods
from calc_CMIP5_EOFs import save_pcs
from cmip5_functions import save_3d_file, load_data, load_sst_data, reconstruct_field, calc_GMSST
from cdo import *
import numpy
from netcdf_file import *
from eofs.standard import Eof
import matplotlib.pyplot as plt


#############################################################################

def get_HadISST_monthly_residual_EOFs_fname(histo_sy, histo_ey, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(run_n)+"_monthly_residual_EOFs.nc"
    return out_fname

#############################################################################

def get_HadISST_monthly_residual_PCs_fname(histo_sy, histo_ey, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(run_n)+ "_monthly_residual_PCs.nc"
    return out_fname

#############################################################################

def calc_HadISST_residual_EOFs(histo_sy, histo_ey, run_n):
    # load the already calculated residuals
    resid_fname = get_HadISST_residuals_fname(histo_sy, histo_ey, run_n)
    # open netcdf_file
    fh = netcdf_file(resid_fname, 'r')
    lats_var = fh.variables["latitude"]
    lons_var = fh.variables["longitude"]
    attrs = fh.variables["sst"]._attributes
    mv = attrs["_FillValue"]
    var = fh.variables["sst"]
    sst_data = numpy.ma.masked_equal(var[:], mv)

    # calculate the EOFs and PCs
    # take the eofs
    coslat = numpy.cos(numpy.deg2rad(lats_var[:])).clip(0., 1.)
    wgts = numpy.sqrt(coslat)[..., numpy.newaxis]
    eof_solver = Eof(sst_data, center=False, weights=wgts)
    pcs = eof_solver.pcs(npcs=None)
    eofs = eof_solver.eofs(neofs=None)

    # get the output names
    out_eofs_fname = get_HadISST_residual_EOFs_fname(histo_sy, histo_ey, run_n)
    out_pcs_fname  = get_HadISST_residual_PCs_fname(histo_sy, histo_ey, run_n)
    
    # save the eofs and pcs
    save_3d_file(out_eofs_fname, eofs, attrs, lats_var, lons_var)
    save_pcs(out_pcs_fname, pcs, attrs)
    fh.close()

#############################################################################

def calc_HadISST_monthly_residual_EOFs(histo_sy, histo_ey, ref_start, ref_end, run_n):
    # load the already calculated residuals
    resid_fname = get_HadISST_monthly_residuals_fname(histo_sy, histo_ey, run_n)
    # note that we don't have to subtract the annual cycle any more as the
    # residuals are with respect to a smoothed version of the monthly ssts
    
    resid_mon_fh = netcdf_file(resid_fname, 'r')
    sst_var = resid_mon_fh.variables["sst"]
    lats_var = resid_mon_fh.variables["latitude"]
    lons_var = resid_mon_fh.variables["longitude"]
    attrs = sst_var._attributes
    mv = attrs["_FillValue"]
    ssts = numpy.array(sst_var[:])
    sst_resids = numpy.ma.masked_less(ssts, -1000)
    
    # calculate the EOFs and PCs
    # take the eofs
    coslat = numpy.cos(numpy.deg2rad(lats_var[:])).clip(0., 1.)
    wgts = numpy.sqrt(coslat)[..., numpy.newaxis]
    eof_solver = Eof(sst_resids, center=True, weights=wgts)
    pcs = eof_solver.pcs(npcs=None)
    eofs = eof_solver.eofs(neofs=None)
    varfrac = eof_solver.varianceFraction(neigs=None)
    print varfrac[0:20], numpy.sum(varfrac[0:20])
    
    # get the output names
    out_eofs_fname = get_HadISST_monthly_residual_EOFs_fname(histo_sy, histo_ey, run_n)
    out_pcs_fname  = get_HadISST_monthly_residual_PCs_fname(histo_sy, histo_ey, run_n)
    
    # save the eofs and pcs
    save_3d_file(out_eofs_fname, eofs, attrs, lats_var, lons_var)
    out_pcs = pcs.reshape([pcs.shape[0],1,pcs.shape[1]])
    save_pcs(out_pcs_fname, out_pcs, attrs)
    resid_mon_fh.close()

#############################################################################

def plot_test_residuals(histo_sy, histo_ey, ref_start, ref_end, run_n):
    # load the yearly eofs and pcs
    yr_eof_fname = get_HadISST_residual_EOFs_fname(histo_sy, histo_ey, run_n)
    yr_eofs = load_sst_data(yr_eof_fname, "sst")
    yr_pcs_fname = get_HadISST_residual_PCs_fname(histo_sy, histo_ey, run_n)
    yr_pcs = load_data(yr_pcs_fname)
    
    # load the monthly eofs and pcs
    mn_eof_fname = get_HadISST_monthly_residual_EOFs_fname(histo_sy, histo_ey, run_n)
    mn_eofs = load_sst_data(mn_eof_fname, "sst")
    mn_pcs_fname = get_HadISST_monthly_residual_PCs_fname(histo_sy, histo_ey, run_n)
    mn_pcs = load_data(mn_pcs_fname)
    
    # load the smoothed hadisst data
#    smooth_fname  = get_HadISST_smooth_fname(histo_sy, histo_ey, run_n)
#    smooth_hadisst = load_sst_data(smooth_fname, "sst")
#    smooth_gmsst = calc_GMSST(smooth_hadisst)
#    smooth_gmsst = smooth_gmsst - numpy.mean(smooth_gmsst[1986-1899:2006-1899])
    
    # reconstruct the fields
    yr_resids = reconstruct_field(yr_pcs, yr_eofs, 20)
    mn_resids = reconstruct_field(mn_pcs, mn_eofs, 20)
        
    # calculate the gmsst
    yr_gmsst = calc_GMSST(yr_resids)
    mn_gmsst = calc_GMSST(mn_resids)
    
    # plot them
    yr_t = numpy.arange(1899,2011,1)
    mn_t = numpy.arange(1899,2011,1.0/12)
    
    sp = plt.subplot(111)
    sp.plot(yr_t, yr_gmsst, 'r', zorder=1)
    sp.plot(mn_t, mn_gmsst, 'k', zorder=0)
    sp.plot(yr_t, smooth_gmsst[:-1], 'b', lw=2.0)
    
    plt.savefig("hadisst_resids.pdf")

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_n = 400
    opts, args = getopt.getopt(sys.argv[1:], 's:e:n:',
                               ['ref_start=', 'ref_end=', 'runn='])

    for opt, val in opts:
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--runn', '-n']:
            run_n = int(val)

    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    histo_ey = 2010
#    calc_HadISST_residual_EOFs(histo_sy, histo_ey, run_n)
    calc_HadISST_monthly_residual_EOFs(histo_sy, histo_ey, ref_start, ref_end, run_n)
#    plot_test_residuals(histo_sy, histo_ey, ref_start, ref_end, run_n)