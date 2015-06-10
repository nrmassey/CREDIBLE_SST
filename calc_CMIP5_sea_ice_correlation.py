#! /usr/bin/env python  
#############################################################################
#
# Program : calc_sea_ice_correlation.py
# Author  : Neil Massey
# Purpose : Calculate the relationship between SST and SIC in the CMIP5
#           ensemble, over the periods 2006->2100 (i.e. forecast) in the
#           projection scenario and 1899->2005 in the historical scenario
# Inputs  : run_type  : rcp4.5 | rc8.5 | histo
# Output  : in the output/ directory filename is:
#            cmip5_warming_pattern_<run_type>_<ref_start>_<ref_end>_<warm>.nc
# Date    : 04/02/15
#
############################################################################# 

import os, sys, getopt
import numpy

from cmip5_functions import get_output_directory, get_cmip5_sic_fname, get_cmip5_tos_fname
from filter_cmip5_members import read_cmip5_index_file
from netcdf_file import *

############################################################################# 

def calc_sst_sic_corr(run_type):
    # calculate the correlation between the sea-ice concentration and the
    # sea-surface temperature in the CMIP5 simulations

    # get the index file
    ref_start = 1986
    ref_end   = 2005
    cmip5_idx = read_cmip5_index_file(run_type, ref_start, ref_end)

    # create the output bin
    sic_bw = 2.5
    tos_bw = 0.1
    tos_min = -2
    tos_max = 2
    tos_range = tos_max - tos_min
    n_tos = int(tos_range/tos_bw)
    n_sic = int((100+sic_bw)/sic_bw)
    out_bin_nh = numpy.zeros([n_tos, n_sic])
    out_bin_sh = numpy.zeros([n_tos, n_sic])
    sic_vals = numpy.array([float(x) * sic_bw for x in range(0, n_sic)], 'f')
    tos_vals = numpy.array([x * tos_bw + tos_min for x in range(0, n_tos)], 'f')

    # loop through each ensemble member
    for ens_mem in cmip5_idx:
        print ens_mem[0]
        # get the filenames for all the files we're using
        rcp_sic_fname  = get_cmip5_sic_fname(run_type, ens_mem[0], ens_mem[1])
        hist_sic_fname = get_cmip5_sic_fname("historical", ens_mem[0], ens_mem[1])
        rcp_tos_fname  = get_cmip5_tos_fname(run_type, ens_mem[0], ens_mem[1])
        hist_tos_fname = get_cmip5_tos_fname("historical", ens_mem[0], ens_mem[1])
        #
        if os.path.exists(rcp_sic_fname) and os.path.exists(rcp_tos_fname):
            # read the files in
            fh_sic_rcp  = netcdf_file(rcp_sic_fname, 'r')
            fh_tos_rcp  = netcdf_file(rcp_tos_fname, 'r')
            
            # start / end indices
            stidx = (2050 - 2006) * 12
            edidx = stidx + 12

            # get the variables from the files
            sic_rcp  = fh_sic_rcp.variables["sic"][stidx:edidx,:,:]
            tos_rcp  = fh_tos_rcp.variables["tos"][stidx:edidx,:,:]

            # get the missing value - assume it's the same for each file in
            # the (individual) model ensemble
            mv = 1000.0

            if sic_rcp.shape != tos_rcp.shape:
                continue

            for t in range(0, sic_rcp.shape[0]):
                for y in range(0, sic_rcp.shape[1]):
                    for x in range(0, sic_rcp.shape[2]):
                        cell_sic = sic_rcp[t,y,x]
                        cell_tos = tos_rcp[t,y,x]
                        # determine where the sea ice is > 0.0 and get the corresponding ssts
                        if abs(cell_tos) > mv:
                            continue
                        if cell_sic < 1.0:
                            continue
                        sic_idx = int(cell_sic/sic_bw + 0.5)
                        tos_idx = int(((cell_tos - 273.13) - tos_min) / tos_bw + 0.5)
                        if tos_idx >= 0 and sic_idx >=0 and tos_idx < n_tos and sic_idx < n_sic:
                            # split into hemispheres
                            if y < sic_rcp.shape[1] / 2:
                                out_bin_nh[tos_idx,sic_idx] += 1
                            else:
                                out_bin_sh[tos_idx,sic_idx] += 1

            fh_sic_rcp.close()
            fh_tos_rcp.close()
    # save the output file
    
    out_bin_nh[out_bin_nh == 0.0] = -1e20
    out_bin_sh[out_bin_sh == 0.0] = -1e20

    out_bin = out_bin_nh
    fname = "output/sea_ice_tos_corr_CMIP5_nh.nc"
    for i in range(0,2):
        out_fh = netcdf_file(fname, 'w')
        # create the dimensions
        sic_vals = numpy.array([x * sic_bw for x in range(0, n_sic)])
        tos_vals = numpy.array([x * tos_bw + tos_min for x in range(0, n_tos)])
        tos_out_dim = out_fh.createDimension("tos", tos_vals.shape[0])
        sic_out_dim = out_fh.createDimension("sic", sic_vals.shape[0])
        tos_out_var = out_fh.createVariable("tos", tos_vals.dtype, ("tos",))
        sic_out_var = out_fh.createVariable("sic", 'f', ("sic",))

        tos_out_var[:] = tos_vals
        sic_out_var[:] = sic_vals

        data_out_var = out_fh.createVariable("freq", out_bin.dtype, ("tos", "sic"))
        data_out_var._attributes = {"_FillValue" : -1e20, "missing_value" : -1e20}
        print data_out_var.shape, out_bin.shape
        data_out_var[:] = out_bin[:]

        out_fh.close()
        fname = "output/sea_ice_tos_corr_CMIP5_sh.nc"
        out_bin = out_bin_sh


############################################################################# 

if __name__ == "__main__":
    run_type = ""
    opts, args = getopt.getopt(sys.argv[1:], 'r:', ['run_type='])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val

    calc_sst_sic_corr(run_type)
