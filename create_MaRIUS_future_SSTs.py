#! /usr/bin/env python  
#############################################################################
#
# Program : create_MaRIUS_future_SSTs.py
# Author  : Neil Massey
# Purpose : Create the MaRIUS future SSTs for the two periods 2020->2050
#           and 2070->2100.  Scenario is rcp8.5
# Inputs  : ref_start : year to start reference period, 1850->2005
#           ref_end   : year to end reference period, 1850->2005
# Notes   : all reference values are calculated from the historical run_type
#           CMIP5 ensemble members are only included if their historical run 
#           includes the reference period
#           requires Andrew Dawsons eofs python libraries:
#            http://ajdawson.github.io/eofs/
# Output  : in the output/ directory filename is:
#            
# Date    : 22/01/16
#
#############################################################################

import os, sys, getopt
from calc_CMIP5_EOFs import get_cmip5_EOF_filename, get_cmip5_proj_PC_scale_filename
from create_CMIP5_syn_PCs import get_syn_SST_PCs_filename
from create_CMIP5_sst_anoms import get_concat_anom_sst_ens_mean_smooth_fname, get_start_end_periods
from create_HadISST_sst_anoms import get_HadISST_smooth_fname, get_HadISST_monthly_reference_fname, get_HadISST_month_smooth_filename
from cmip5_functions import load_data, reconstruct_field, calc_GMSST, load_sst_data
from calc_HadISST_residual_EOFs import get_HadISST_monthly_residual_EOFs_fname, get_HadISST_monthly_residual_PCs_fname
from create_HadISST_CMIP5_syn_SSTs import *
import numpy
from create_HadISST_sst_anoms import get_HadISST_monthly_residuals_fname
from netcdf_file import *
from ARN import ARN
import pyximport
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
from zonal_smoother import *

#############################################################################

def get_Ma_output_directory(run_type, ref_start, ref_end, sy, ey):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_dir = "MaSST_"+run_type+"_"+str(sy)+"_"+str(ey)+\
              "_r"+str(ref_start)+"_"+str(ref_end)
    path = "/Users/Neil/Coding/MaRIUS_output/"
    if not os.path.exists(path+out_dir):
        os.mkdir(path+out_dir)
    return path + out_dir

#############################################################################

def get_Ma_output_name(run_type, ref_start, ref_end, sy, ey):
    out_name = "MaSST_"+run_type+"_"+str(sy)+"_"+str(ey)+\
              "_r"+str(ref_start)+"_"+str(ref_end)+".nc"
    return out_name

#############################################################################

def save_Ma_syn_SSTs(out_data, run_type, ref_start, ref_end, sy, ey):    
    # we require the time data and shape of the field - get this from the cmip5 ens mean file
    ens_mean_fname = get_concat_anom_sst_ens_mean_smooth_fname(run_type, ref_start, ref_end, True)
    fh2 = netcdf_file(ens_mean_fname, 'r')
    lon_var = fh2.variables["longitude"]
    lat_var = fh2.variables["latitude"]
    t_var = fh2.variables["time"]
    attrs = fh2.variables["sst"]._attributes

    # mask any nans
    mv = attrs["_FillValue"]
    out_data = numpy.ma.fix_invalid(out_data, fill_value=mv)
    # fix the lsm
    for t in range(1, out_data.shape[0]):
        out_data.data[t][out_data.data[0] == mv] = mv
    
    # smooth the data
    lat_data = lat_var[:].byteswap().newbyteorder().astype(numpy.float32)
    out_data = out_data.astype(numpy.float32)
    out_data = zonal_smoother(out_data, lat_data, 64, 90, mv)

    # get the output name
    out_dir = get_Ma_output_directory(run_type, ref_start, ref_end, sy, ey-1)
    out_name = out_dir + "/" + get_Ma_output_name(run_type, ref_start, ref_end, sy, ey-1)

    cmip5_sy = 1899
    t_vals = t_var[(sy-cmip5_sy)*12:(ey-cmip5_sy)*12]
    if ey == 2101:
        t_vals2 = numpy.zeros([t_vals.shape[0]+12], 'f')
        t_vals2[:-12] = t_vals[:]
        t_vals2[-12:] = t_vals[-12:] + (t_vals[12] - t_vals[0])
    else:
        t_vals2 = t_vals
    save_3d_file(out_name, out_data, lon_var, lat_var, attrs, t_vals2, t_var._attributes)
    fh2.close()
    print out_name

#############################################################################

def create_Ma_syn_SSTs(load_run_type, ref_start, ref_end, sy, ey):
    # determine which hadisst ensemble member to use
    hadisst_ens_members = [1059, 115, 1169, 1194, 1346, 137, 1466, 396, 400, 69]
    run_n = hadisst_ens_members[numpy.random.randint(0, len(hadisst_ens_members))]

    # load the CMIP5 ensemble mean timeseries
    # load the ensemble mean of the anomalies
    cmip5_ens_mean_anoms_fname = get_concat_anom_sst_ens_mean_smooth_fname(load_run_type, ref_start, ref_end, True)
    cmip5_ens_mean_anoms = load_sst_data(cmip5_ens_mean_anoms_fname, "sst")

    # sub set the anoms
    cmip5_sy = 1899
    cmip5_ens_mean_anoms = cmip5_ens_mean_anoms[(sy-cmip5_sy)*12:(ey-cmip5_sy)*12]

    if ey == 2101:
        # create 2101
        S = cmip5_ens_mean_anoms.shape
        cmip5_ens_mean_anoms2 = numpy.zeros([S[0]+12, S[1], S[2]], 'f')
        cmip5_ens_mean_anoms2[:S[0]] = cmip5_ens_mean_anoms
        cmip5_ens_mean_anoms2[-12:] = cmip5_ens_mean_anoms[-12:]
#        cmip5_ens_mean_anoms2[-24:-12] = cmip5_ens_mean_anoms[-12:]
        cmip5_ens_mean_anoms = cmip5_ens_mean_anoms2

    # load the hadisst reference
    n_repeats = cmip5_ens_mean_anoms.shape[0] / 12       # number of repeats = number of years
    hadisst_ac = create_hadisst_monthly_reference(run_type, ref_start, ref_end, n_repeats, run_n)
    # load the internal variability - we are only interested in the 30 year observed ones
    resid_fname = get_HadISST_monthly_residuals_fname(1899, 2010, 400)
    intvar = load_data(resid_fname, "sst")
    intvar = intvar[(1973-1899)*12:(2007-1899)*12]
    out_data = cmip5_ens_mean_anoms + hadisst_ac + intvar
    # save the synthetic ssts
    save_Ma_syn_SSTs(out_data, run_type, ref_start, ref_end, sy, ey)

#############################################################################

if __name__ == "__main__":
    # Note - this is all an adaptation of CREDIBLE SST functions
    ref_start = -1
    ref_end = -1
    run_type = "rcp85"
    intvarmode = 0      # internal variability mode - 0 = none, 1 = yearly, 2 = monthly
    monthly = True     # use the monthly EOFs / PCs ?
    opts, args = getopt.getopt(sys.argv[1:], 's:e:i:y:z:',
                               ['ref_start=', 'ref_end=', 
                                'start_year=', 'end_year='])

    for opt, val in opts:
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--start_year', '-y']:
            sy = int(val)
        if opt in ['--end_year', '-z']:
            ey = int(val)
    
    create_Ma_syn_SSTs(run_type, ref_start, ref_end, sy, ey)
