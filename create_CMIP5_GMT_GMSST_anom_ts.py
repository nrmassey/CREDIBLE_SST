#! /usr/bin/env python  
#############################################################################
#
# Program : create_CMIP5_GMT_GMSST_anom_ts.py
# Author  : Neil Massey
# Purpose : Create concatenated global mean temperature (GMT) anomaly and 
#           global mean sea-surface temperature (GMSST) anomaly timeseries 
#           from the CMIP5 ensemble the two CMIP5 periods:
#           historical (1899->2005) and future RCP (2006->2100)
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
# Date    : 18/03/15
#
#############################################################################

import os, sys, getopt
from cmip5_functions import get_cmip5_tos_fname, get_cmip5_tas_fname, get_output_directory
from cmip5_functions import calc_GMSST
from create_CMIP5_sst_anoms import get_start_end_periods
from filter_cmip5_members import read_cmip5_index_file
from cdo import *
from netcdf_file import *
import numpy

#############################################################################

def get_grid_spacing(fname):
    # get the grid size
    cdo = Cdo()
    G = cdo.griddes(input=fname)
    for g in G:
        if "xsize" in g:
            xD = 360.0 / float(g.split('=')[-1])
        if "ysize" in g:
            yD = 180.0 / float(g.split('=')[-1])
    return xD, yD

#############################################################################

def create_tmp_anom_file(fname, ref_fname, sy, ey, var, monthly=False, lat=-1.0, lon=-1.0):
    cdo = Cdo()

    anom_string = " -fldmean "
    if lat != -1.0 and lon != -1.0:
#        xD,yD = get_grid_spacing(fname)
        anom_string += " -remapnn,lon="+str(lon)+"_lat="+str(lat)
    
    if not monthly:
        anom_string += " -yearmean "
    anom_string += " -selyear,"+str(sy)+"/"+str(ey)+\
                   " -selname,"+var+\
                   " " + fname +\
                   " " + ref_fname
    out_name = fname.split("/")[-1] + "_" + var + "_anom_tmp.nc"
    cdo.sub(input = anom_string, output = out_name)
    return out_name

#############################################################################

def create_tmp_ref_file(histo_fname, ref_start, ref_end, var, monthly=False, lat=-1.0, lon=-1.0):
    cdo = Cdo()
    if monthly:
        ref_anom_string = " -ymonmean"
    else:
        ref_anom_string = " -timmean"
    if lat != -1.0 and lon != -1.0:
#        xD,yD = get_grid_spacing(histo_fname)
        ref_anom_string += " -remapnn,lon="+str(lon)+"_lat="+str(lat)
        
    ref_anom_string += " -selyear," + str(ref_start)+"/"+str(ref_end) +\
                      " -selname," + var +\
                      " " + histo_fname
    out_name = var + "_ref_tmp.nc"
    cdo.fldmean(input = ref_anom_string, output = out_name)
    return out_name

#############################################################################

def get_gmt_gmsst_anom_ts_fname(run_type, ref_start, ref_end, monthly=False, lat=-1.0, lon=-1.0):
    out_dir = get_output_directory(run_type, ref_start, ref_end)
    out_name = out_dir + "/" + out_dir.split("/")[-1] + "_tos_tas_GM_ts"
    if monthly:
        out_name += "_mon"
    if lat != -1.0 and lon != -1.0:
        out_name += "_ll"+str(lat)+"N_"+str(lon)+"E"
    out_name += ".nc"
    return out_name
    
#############################################################################

def calc_GMT_GMSST_anom_ts(run_type, ref_start, ref_end, monthly=False, lat=-1.0, lon=-1.0):
    # get the filtered list of CMIP5 ensemble members
    cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)

    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()

    n_ens = len(cmip5_rcp_idx)
    if monthly:
        f = 12
    else:
        f = 1
    n_t = (rcp_ey - histo_sy + 1)*f
    all_tos = numpy.zeros([n_ens, n_t], 'f')
    all_tas = numpy.zeros([n_ens, n_t], 'f')
    t_attr = None
    t_vals = None
    
    for idx in range(0, n_ens):
        cdo = Cdo()
        print cmip5_rcp_idx[idx][0]
        # get the tos filenames for the rcp and historical simulation
        tos_rcp_fname = get_cmip5_tos_fname(run_type, cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])
        tos_histo_fname = get_cmip5_tos_fname("historical", cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])

        # get the tas filenames for the rcp and historical simulation
        tas_rcp_fname = get_cmip5_tas_fname(run_type, cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])
        tas_histo_fname = get_cmip5_tas_fname("historical", cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])

        # create the reference files
        tos_ref_fname = create_tmp_ref_file(tos_histo_fname, ref_start, ref_end, "tos", monthly, lat, lon)
        tas_ref_fname = create_tmp_ref_file(tas_histo_fname, ref_start, ref_end, "tas", monthly, lat, lon)
        
        # do the anomalies
        tos_histo_anom_fname = create_tmp_anom_file(tos_histo_fname, tos_ref_fname, histo_sy, histo_ey, "tos", monthly, lat, lon)
        tos_rcp_anom_fname = create_tmp_anom_file(tos_rcp_fname, tos_ref_fname, rcp_sy, rcp_ey, "tos", monthly, lat, lon)
        cdo.cat(input = tos_histo_anom_fname + " " + tos_rcp_anom_fname, output = "tos_temp.nc")
        
        tas_histo_anom_fname = create_tmp_anom_file(tas_histo_fname, tas_ref_fname, histo_sy, histo_ey, "tas", monthly, lat, lon)
        tas_rcp_anom_fname = create_tmp_anom_file(tas_rcp_fname, tas_ref_fname, rcp_sy, rcp_ey, "tas", monthly, lat, lon)
        cdo.cat(input = tas_histo_anom_fname + " " + tas_rcp_anom_fname, output = "tas_temp.nc")
                
        # read the temporary files in and add to the numpy array
        fh_tos = netcdf_file("tos_temp.nc")
        fh_tas = netcdf_file("tas_temp.nc")
        try:
            all_tos[idx] = fh_tos.variables["tos"][:].squeeze()
            all_tas[idx] = fh_tas.variables["tas"][:].squeeze()
        except:
            all_tos[idx] = 1e20
            all_tas[idx] = 1e20
            
        # get the time values / attributes
        if idx == 0:#n_ens-1:
            t_vals = numpy.array(fh_tos.variables["time"][:])
            t_attr = fh_tos.variables["time"]._attributes
            
        os.remove("tas_temp.nc")
        os.remove(tas_ref_fname)
        os.remove(tas_histo_anom_fname)
        os.remove(tas_rcp_anom_fname)
        os.remove("tos_temp.nc")
        os.remove(tos_ref_fname)
        os.remove(tos_histo_anom_fname)
        os.remove(tos_rcp_anom_fname)
        fh_tos.close()
        fh_tas.close()

    # clean up last
    
    # save the all tos / all tas file
    out_name = get_gmt_gmsst_anom_ts_fname(run_type, ref_start, ref_end, monthly, lat, lon)
    out_fh = netcdf_file(out_name, "w")
    # create dimensions and variables
    time_out_dim = out_fh.createDimension("time", t_vals.shape[0])
    time_out_var = out_fh.createVariable("time", t_vals.dtype, ("time",))
    ens_out_dim = out_fh.createDimension("ens", n_ens)
    ens_out_var = out_fh.createVariable("ens", 'f', ("ens",))
    tos_out_var = out_fh.createVariable("tos", all_tos.dtype, ("ens", "time",))
    tas_out_var = out_fh.createVariable("tas", all_tas.dtype, ("ens", "time",))
    # write out variables
    time_out_var._attributes = t_attr
    time_out_var[:] = t_vals[:]
    ens_out_var[:] = numpy.arange(0, n_ens)
    # write out data
    tos_out_var[:] = all_tos[:]
    tas_out_var[:] = all_tas[:]
    
    out_fh.close()

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    monthly = False
    lat = -1.0
    lon = -1.0
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:a:l:m',
                               ['run_type=', 'ref_start=', 'ref_end=', 
                                'longitude=', 'latitude=', 'monthly'])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True
        if opt in ['--latitude', '-a']:
            lat = float(val)
        if opt in ['--longitude', '-l']:
            lon = float(val)

    calc_GMT_GMSST_anom_ts(run_type, ref_start, ref_end, monthly, lat, lon)