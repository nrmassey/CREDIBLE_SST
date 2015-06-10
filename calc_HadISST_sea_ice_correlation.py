#! /usr/bin/env python  
#############################################################################
#
# Program : calc_sea_ice_correlation.py
# Author  : Neil Massey
# Purpose : Calculate the relationship between SST and SIC in the CMIP5
#           ensemble, over the periods 2006->2100 (i.e. forecast) in the
#           projection scenario and 1899->2005 in the historical scenario
# Inputs  : 
# Output  : in the output/ directory filename is:
#            
# Date    : 04/02/15
#
############################################################################# 

import os, sys, getopt
import numpy

from cmip5_functions import get_output_directory, get_cmip5_sic_fname, get_cmip5_tos_fname
from calc_HadISST_ts_proj_GMSST import get_HadISST2_filepath
from filter_cmip5_members import read_cmip5_index_file
from netcdf_file import *

############################################################################# 

def calc_sst_sic_corr():
    # calculate the correlation between the sea-ice concentration and the
    # sea-surface temperature in the CMIP5 simulations

    # get HadISST file
    fname = get_HadISST2_filepath()
    

    # create the output bin
    sic_bw = 0.025
    tos_bw = 0.1
    tos_min = -2
    tos_max = 2
    tos_range = tos_max - tos_min
    n_tos = int(tos_range/tos_bw)
    n_sic = int((1+sic_bw)/sic_bw)
    out_bin_nh = numpy.zeros([n_tos, n_sic])
    out_bin_sh = numpy.zeros([n_tos, n_sic])
    sic_vals = numpy.array([float(x) * sic_bw for x in range(0, n_sic)], 'f')
    tos_vals = numpy.array([x * tos_bw + tos_min for x in range(0, n_tos)], 'f')

    # loop through each ensemble member
    #
    # read the files in
    fh_hadisst  = netcdf_file(fname, 'r')
    
    # get the variables from the files
    sic_hadisst = fh_hadisst.variables["sic"][:]
    tos_hadisst = fh_hadisst.variables["sst"][:]

    # get the missing value - assume it's the same for each file in
    # the (individual) model ensemble
    mv = 1000.0

    for t in range(0, 100):#sic_hadisst.shape[0]):
        for y in range(0, sic_hadisst.shape[1]):
            for x in range(0, sic_hadisst.shape[2]):
                cell_sic = sic_hadisst[t,y,x]
                cell_tos = tos_hadisst[t,y,x]
                # determine where the sea ice is > 0.0 and get the corresponding ssts
                if abs(cell_tos) > mv or abs(cell_sic) > mv:
                    continue
                if cell_sic == 0.0:
                    continue
                sic_idx = int(cell_sic/sic_bw)
                tos_idx = int(((cell_tos - 273.13) - tos_min) / tos_bw + 0.5)
                if tos_idx >= 0 and sic_idx >=0 and tos_idx < n_tos and sic_idx < n_sic:
                    # split into hemispheres
                    if y < sic_hadisst.shape[1] / 2:
                        out_bin_nh[tos_idx,sic_idx] += 1
                    else:
                        out_bin_sh[tos_idx,sic_idx] += 1

    fh_hadisst.close()
    
    # missing values
    out_bin_nh[out_bin_nh == 0.0] = -1e20
    out_bin_sh[out_bin_sh == 0.0] = -1e20
    
    # save the output file
    
    out_bin = out_bin_nh
    fname = "output/sea_ice_tos_corr_HadISST_nh.nc"
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
        data_out_var[:] = out_bin[:]

        out_fh.close()
        fname = "output/sea_ice_tos_corr_HadISST_sh.nc"
        out_bin = out_bin_sh

############################################################################# 

if __name__ == "__main__":
    calc_sst_sic_corr()
