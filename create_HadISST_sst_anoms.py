#! /usr/bin/env python  
#############################################################################
#
# Program : create_HadISST_sst_anoms.py
# Author  : Neil Massey
# Purpose : Create the HadISST smoothed sst anomalies over the HadISST
#           period (1899->2014) and future RCP (2006->2100)
# Inputs  : run_type  : rcp4.5 | rc8.5 | histo
#           ref_start : year to start reference period, 1850->2005
#           ref_end   : year to end reference period, 1850->2005
#           run_type  : historical | rcp45 | rcp85
# Notes   : all reference values are calculated from the historical run_type
#           CMIP5 ensemble members are only included if their historical run 
#           includes the reference period
#           requires Andrew Dawsons eofs python libraries:
#            http://ajdawson.github.io/eofs/
# Output  : in the output/ directory filename is:
#            
# Date    : 18/02/15
#
#############################################################################

import os, sys, getopt
sys.path.append("../python_lib")
from create_CMIP5_sst_anoms import get_start_end_periods, save_3d_file
from cmip5_functions import calc_GMSST, load_sst_data

from netcdf_file import *
import numpy
from cdo import *
from running_gradient_filter import *

#############################################################################

def get_uname():
    uname = os.uname()
    if uname[0] == "Darwin":
        out_uname = "Darwin"
    elif uname[1] == "ouce-linux-01.ouce.ox.ac.uk" or \
         uname[1] == "ouce-linux-02.ouce.ox.ac.uk":
        out_uname = "Linux"
    return out_uname

#############################################################################

def get_HadISST_input_filename(run_n):
    uname = get_uname()
    if uname == "Darwin":
        hadisst_path = "/Users/Neil/ClimateData/HadISST2/"
    elif uname == "Linux":
        hadisst_path = "/soge-home/data_not_backed_up/surface/hadisst_2/1.0x1.0/monthly/nc/"
    path = hadisst_path+"HadISST.2.1.0.0_realisation_dec2010_"+str(run_n)+".nc"
    return path

#############################################################################

def get_HadISST_output_directory(histo_sy, histo_ey, run_n):
    uname = get_uname()
    if uname == "Darwin":
        out_base_dir = "/Users/Neil/Coding/CREDIBLE_output/output/"
    elif uname == "Linux":
        out_base_dir = "/soge-home/staff/coml0118/CREDIBLE_output/output/"
    out_dir = out_base_dir + "HadISST_" + str(histo_sy) + "_" + str(histo_ey) + "_" + str(run_n)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir
    
#############################################################################

def get_HadISST_smooth_fname(histo_sy, histo_ey, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(run_n)+"_smooth.nc"
    return out_fname

#############################################################################

def get_HadISST_month_smooth_filename(histo_sy, histo_ey, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(run_n)+"_monthly_smooth.nc"
    return out_fname

#############################################################################

def get_HadISST_residuals_fname(histo_sy, histo_ey, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(run_n)+"_residuals.nc"
    return out_fname

#############################################################################
    
def get_HadISST_monthly_residuals_fname(histo_sy, histo_ey, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(run_n)+"_monthly_residuals.nc"
    return out_fname
    
#############################################################################

def get_HadISST_annual_cycle_residuals_fname(histo_sy, histo_ey, ref_start, ref_end, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(ref_start)+"_"+str(ref_end)+"_"+str(run_n)+"_annual_cycle_residuals.nc"
    return out_fname
    
#############################################################################

def get_HadISST_reference_fname(histo_sy, histo_ey, ref_start, ref_end, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(ref_start)+"_"+str(ref_end)+"_"+str(run_n)+"_ref.nc"
    return out_fname

#############################################################################

def get_HadISST_monthly_reference_fname(histo_sy, histo_ey, ref_start, ref_end, run_n):
    out_dir = get_HadISST_output_directory(histo_sy, histo_ey, run_n)
    out_fname = out_dir+"/hadisst_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_"+str(ref_start)+"_"+str(ref_end)+"_"+str(run_n)+"_mon_smooth_ref.nc"
    return out_fname

#############################################################################

def create_HadISST_smoothed(histo_sy, histo_ey, run_n):
    # create a smoothed version of HadISST by first taking yearly means 
    # and then running the running mean / running gradient filter over these
    # yearly mean values
    
    # get the filenames
    in_fname = get_HadISST_input_filename(run_n)
    out_fname = get_HadISST_smooth_fname(histo_sy, histo_ey, run_n)
    
    # use cdo to calculate the yearly mean and return the temporary file
    cdo = Cdo()
    year_mean_file = cdo.yearmean(input=" -selyear,"+str(histo_sy)+"/"+str(histo_ey)+" "+in_fname)
    cdf_fh = netcdf_file(year_mean_file, 'r')
    # get the ssts, time variable, lon and lat variables
    sst_var  = cdf_fh.variables["sst"]
    time_var = cdf_fh.variables["time"]
    lon_var  = cdf_fh.variables["longitude"]
    lat_var  = cdf_fh.variables["latitude"]

    # have to byteswap the data out of a netcdf file
    sst_data = numpy.array(sst_var[:])
    sst_data = sst_data.byteswap().newbyteorder()
    # run the running gradient filter on this data
    P = 40
    mv = sst_var._attributes["_FillValue"]
    smoothed_data = running_gradient_3D(sst_data, P, mv)
    cdf_fh.close()
    # save the file
    save_3d_file(out_fname, smoothed_data, lon_var, lat_var, sst_var._attributes, time_var)
    cdf_fh.close()

#############################################################################

def create_HadISST_monthly_smoothed(histo_sy, histo_ey, run_n):
    # create the monthly smoothed version of HadISST2
    # this consists of applying the 40 year smoother to individual months
    # i.e. apply smoother to all Januaries, all Februaries etc.
    # load the data in
    in_fname = get_HadISST_input_filename(run_n)
    out_fname = get_HadISST_month_smooth_filename(histo_sy, histo_ey, run_n)
    
    # load the data - use cdo to select the year and create a temporary file
    cdo = Cdo()
    monthly_file = cdo.selyear(str(histo_sy)+"/"+str(histo_ey)+" "+in_fname)
    fh_m = netcdf_file(monthly_file, 'r')
    lon_var = fh_m.variables["longitude"]
    lat_var = fh_m.variables["latitude"]
    t_var   = fh_m.variables["time"]
    sst_var = fh_m.variables["sst"]
    P = 40
    mv = sst_var._attributes["_FillValue"]
    hadisst_sst = numpy.array(sst_var[:])
    hadisst_sst = hadisst_sst.byteswap().newbyteorder()

    # create the output - same shape as the input
    month_smoothed_hadisst_sst = numpy.zeros(hadisst_sst.shape, hadisst_sst.dtype)
    # now do the monthly smoothing
    for m in range(0,12):
        month_smoothed_hadisst_sst[m::12] = running_gradient_3D(hadisst_sst[m::12], P, mv)
    
    # save the file
    save_3d_file(out_fname, month_smoothed_hadisst_sst, lon_var, lat_var, sst_var._attributes, t_var)

    fh_m.close()

#############################################################################

def create_HadISST_residuals(histo_sy, histo_ey, run_n):
    # create the residuals achieved by subtracting the smoothed version of
    # HadISST from the yearly mean version
    smooth_fname  = get_HadISST_smooth_fname(histo_sy, histo_ey, run_n)
    resids_fname  = get_HadISST_residuals_fname(histo_sy, histo_ey, run_n)
    hadisst_fname = get_HadISST_input_filename(run_n)
    
    # use cdo to do the yearly mean and subtraction
    cdo = Cdo()
    cdo.sub(input=" -yearmean -selyear,"+str(histo_sy)+"/"+str(histo_ey)+" -selvar,sst "+hadisst_fname+" "+smooth_fname, 
            output=resids_fname)

#############################################################################

def create_HadISST_monthly_residuals(histo_sy, histo_ey, run_n):
    # create the residuals achieved by subtracting the monthly smoothed version 
    # of HadISST from the monthly mean version
    month_smooth_fname  = get_HadISST_month_smooth_filename(histo_sy, histo_ey, run_n)
    resids_fname  = get_HadISST_monthly_residuals_fname(histo_sy, histo_ey, run_n)
    hadisst_fname = get_HadISST_input_filename(run_n)
    
    # use cdo to do the subset and subtraction
    cdo = Cdo()
    cdo.sub(input=" -selyear,"+str(histo_sy)+"/"+str(histo_ey)+" -selvar,sst "+hadisst_fname+" "+month_smooth_fname,
            output=resids_fname)

#############################################################################

def create_HadISST_annual_cycle_residuals(histo_sy, histo_ey, ref_start, ref_end, run_n):
    # create an annual cycle of HadISST residuals by subtracting the 1986->2005 monthly 
    # mean of HadISST from the 1986->2005 mean of the 40 year smoothed yearly HadISST
    hadisst_fname = get_HadISST_input_filename(run_n)
    hadisst_year_smooth = get_HadISST_smooth_fname(histo_sy, histo_ey, run_n)
    ac_fname = get_HadISST_annual_cycle_residuals_fname(histo_sy, histo_ey, ref_start, ref_end, run_n)
    
    cdo = Cdo()
    cdo.sub(input=" -ymonmean -selyear,"+str(ref_start)+"/"+str(ref_end)+" -selvar,sst "+hadisst_fname+" " +\
                  " -yearmean -selyear,"+str(ref_start)+"/"+str(ref_end)+" -selvar,sst "+hadisst_year_smooth,
                 output=ac_fname)

#############################################################################

def create_HadISST_reference(histo_sy, histo_ey, ref_start, ref_end, run_n):
    in_fname = get_HadISST_input_filename(run_n)
    out_fname = get_HadISST_reference_fname(histo_sy, histo_ey, ref_start, ref_end, run_n)
    
    # use cdo to take the 1986->2005 (ref_start->ref_end) mean
    cdo = Cdo()
    cdo.timmean(input=" -selyear,"+str(ref_start)+"/"+str(ref_end)+" -selvar,sst "+in_fname+" ", output=out_fname)

#############################################################################

def create_HadISST_reference_SIC(histo_sy, histo_ey, ref_start, ref_end, run_n):
    in_fname = get_HadISST_input_filename(run_n)
    out_fname = get_HadISST_reference_fname(histo_sy, histo_ey, ref_start, ref_end, run_n)
    out_fname = out_fname[:-3] + "_sic.nc"
    # use cdo to take the 1986->2005 (ref_start->ref_end) mean
    cdo = Cdo()
    cdo.ymonmean(input=" -selyear,"+str(ref_start)+"/"+str(ref_end)+" -selvar,sic "+in_fname+" ", output=out_fname)

#############################################################################

def create_HadISST_monthly_smoothed_reference(histo_sy, histo_ey, ref_start, ref_end, run_n):
    in_fname = get_HadISST_month_smooth_filename(histo_sy, histo_ey, run_n)
    out_fname = get_HadISST_monthly_reference_fname(histo_sy, histo_ey, ref_start, ref_end, run_n)
    cdo = Cdo()
    cdo.ymonmean(input=" -selyear,"+str(ref_start)+"/"+str(ref_end)+" -selvar,sst "+in_fname+" ", output=out_fname)

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
            run_n = val
            

    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    histo_ey = 2010

    create_HadISST_reference_SIC(histo_sy, histo_ey, ref_start, ref_end, run_n)

    # Now that we have switched to being completely monthly based we need 3 files
    # Firstly, the monthly smoothed file
    create_HadISST_monthly_smoothed(histo_sy, histo_ey, run_n)
    # Secondly, the residuals, i.e. the monthly HadISST file minus the smoothed monthly file    
    create_HadISST_monthly_residuals(histo_sy, histo_ey, run_n)
    # Thirdly, the reference file.  This is the climatological monthly mean of the 1986->2005
    # period, given by ref_start and ref_end
    create_HadISST_monthly_smoothed_reference(histo_sy, histo_ey, ref_start, ref_end, run_n)

