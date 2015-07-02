#! /usr/bin/env python  
#############################################################################
#
# Program : create_CMIP5_sst_anoms.py
# Author  : Neil Massey
# Purpose : Create concatenated CMIP5 sst anomalies over the two CMIP5 periods:
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
# Date    : 10/02/15
#
#############################################################################

import os, sys, getopt
from cmip5_functions import get_cmip5_tos_fname, get_output_directory, get_cmip5_sic_fname
from filter_cmip5_members import read_cmip5_index_file, read_cmip5_model_mean_index_file
from cdo import *
from netcdf_file import *
import numpy

# uncomment this on local machine
#import pyximport
#pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
#from running_gradient_filter import *

#############################################################################

# start and end of rcp and historical periods

def get_start_end_periods():

    histo_sy = 1899
    histo_ey = 2005
    rcp_sy   = 2006
    rcp_ey   = 2100
    return histo_sy, histo_ey, rcp_sy, rcp_ey
    
#############################################################################

def create_tmp_ref_field(ref_fname, ref_start, ref_end, var, monthly):
    # calculate the temporary reference file
    tmp_ref_name = "tmp_ref_sst.nc"

    if monthly:
        anom_string = " -ymonmean "
    else:
        anom_string = " -timmean "
    anom_string += " -selyear," + str(ref_start) + "/" + str(ref_end) +\
                   " -selname,\"" + var + "\" " + ref_fname
    cdo = Cdo()
    tmp_ref_name = "tmp_ref_sst.nc"
    cdo.addc(0, input=anom_string, output=tmp_ref_name)
    
    return tmp_ref_name
    
#############################################################################

def get_calc_anom_string(tmp_ref_name, tgt_fname, 
                         tgt_start, tgt_end, var, monthly):
    anom_string = " -setmissval,-1e30 "+\
                  " -sub " +\
                  " -selyear," + str(tgt_start) + "/" + str(tgt_end) +\
                  " -selname,\"" + var + "\" " + tgt_fname +\
                  " " + tmp_ref_name + " "
    return anom_string

#############################################################################

def create_remapped_field(fname, tgt_start, tgt_end, var, monthly):
    lsm_grid = "/soge-home/staff/coml0118/LSM/hadisst_grid"
    cdo_string = " -remapbil," + lsm_grid
    if not monthly:
        cdo_string += " -yearmean"
    cdo_string += " -fillmiss "
    if var == "tos":
        cdo_string += " -setctomiss,0 "         # fix for MIROC model
        cdo_string += " -setrtomiss,1e3,1e20 "  # fix for IPSL model
    cdo_string += " -selyear," + str(tgt_start) + "/" + str(tgt_end) +\
                  " -selname,\"" + var + "\" " + fname
    return cdo_string

#############################################################################

def get_concat_output_path(run_type, ref_start, ref_end):
    out_path = "../CREDIBLE_output/output/"+run_type+"_"+str(ref_start)+"_"+str(ref_end)+"/concat_sst_anoms/"
    return out_path

#############################################################################

def get_concat_anom_sst_output_fname(idx0, idx1, run_type, ref_start, ref_end, monthly):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_name = idx0 + "_" + idx1 + "_" +\
               "histo" + "_" + str(histo_sy) + "_" + str(histo_ey) + "_" +\
               run_type + "_" + str(rcp_sy) + "_" + str(rcp_ey)
    if monthly:
        out_name += "_mon"
    out_name += ".nc"
    out_path = get_concat_output_path(run_type, ref_start, ref_end) + out_name
    return out_path
    
#############################################################################

def get_concat_anom_sst_smooth_fname(idx0, idx1, run_type, ref_start, ref_end, monthly):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_name = idx0 + "_" + idx1 + "_" +\
               "histo" + "_" + str(histo_sy) + "_" + str(histo_ey) + "_" +\
               run_type + "_" + str(rcp_sy) + "_" + str(rcp_ey) + "_sub_em_smooth"
    if monthly:
        out_name += "_mon"
    out_name += ".nc"
    out_path = get_concat_output_path(run_type, ref_start, ref_end) + out_name
    return out_path

#############################################################################

def get_concat_anom_sst_smooth_model_mean_fname(idx0, idx1, run_type, ref_start, ref_end):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_name = idx0 + "_" + idx1 + "_" +\
               "histo" + "_" + str(histo_sy) + "_" + str(histo_ey) + "_" +\
               run_type + "_" + str(rcp_sy) + "_" + str(rcp_ey) + "_mm_sub_em_smooth.nc"
    out_path = get_concat_output_path(run_type, ref_start, ref_end) + out_name
    return out_path

#############################################################################

def get_concat_anom_sst_ens_mean_fname(run_type, ref_start, ref_end, monthly):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_name = "cmip5_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_rcp45_"+str(rcp_sy)+"_"+str(rcp_ey)+"_ens_mean"
    if monthly:
        out_name += "_mon"
    out_name += ".nc"
    out_path = get_concat_output_path(run_type, ref_start, ref_end) + out_name
    return out_path

#############################################################################

def get_concat_anom_sst_ens_mean_smooth_fname(run_type, ref_start, ref_end, monthly):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_name = "cmip5_hist_"+str(histo_sy)+"_"+str(histo_ey)+"_rcp45_"+str(rcp_sy)+"_"+str(rcp_ey)+"_ens_mean_smooth"
    if monthly:
        out_name += "_mon"
    out_name += ".nc"
    out_path = get_concat_output_path(run_type, ref_start, ref_end) + out_name
    return out_path

#############################################################################

def get_concat_anom_sst_model_mean_fname(idx0, run_type, ref_start, ref_end, monthly):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_name = idx0 + "_mm_" +\
               "histo" + "_" + str(histo_sy) + "_" + str(histo_ey) + "_" +\
               run_type + "_" + str(rcp_sy) + "_" + str(rcp_ey)
    out_path = get_concat_output_path(run_type, ref_start, ref_end) + out_name
    return out_path

#############################################################################

def create_concat_sst_anoms_ens_mean(run_type, ref_start, ref_end, monthly):
    out_name = get_concat_anom_sst_ens_mean_fname(run_type, ref_start, ref_end, monthly)
    cdo = Cdo()
    cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)
    n_ens = len(cmip5_rcp_idx)
    input = ""
    
    for idx in range(0, n_ens):
        input += get_concat_anom_sst_output_fname(cmip5_rcp_idx[idx][0],
                                                  cmip5_rcp_idx[idx][1],
                                                  run_type, ref_start, ref_end,
                                                  monthly)
        input += " "
    cdo.ensmean(input=input, output=out_name)

#############################################################################

def create_concat_sst_anoms_ens_mean_smoothed(run_type, ref_start, ref_end, monthly):
    in_fname = get_concat_anom_sst_ens_mean_fname(run_type, ref_start, ref_end, monthly)
    out_fname = get_concat_anom_sst_ens_mean_smooth_fname(run_type, ref_start, ref_end, monthly)
    
    # load the input netcdf file
    in_fh = netcdf_file(in_fname)
    in_var = in_fh.variables["tos"]
    lon_var = in_fh.variables["longitude"]
    lat_var = in_fh.variables["latitude"]
    t_var = in_fh.variables["time"]
    mv = in_var._attributes["_FillValue"]
    
    in_data = in_var[:].byteswap().newbyteorder()
    P = 40
    smoothed_data = running_gradient_3D(in_data, P, mv)
    save_3d_file(out_fname, smoothed_data, lon_var, lat_var, in_var._attributes, t_var)
    in_fh.close()

#############################################################################

def create_concat_sst_anoms_model_means(run_type, ref_start, ref_end):
    # create the ensemble means of the models in the cmip5 archive
    cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)
    cidx = 1
    
    cdo = Cdo()
    
    while cidx < len(cmip5_rcp_idx):
        cmodel = cmip5_rcp_idx[cidx-1][0]
        cstring = get_concat_anom_sst_output_fname(cmodel,
                                                   cmip5_rcp_idx[cidx-1][1],
                                                   run_type, ref_start, ref_end)
        while cidx < len(cmip5_rcp_idx) and cmip5_rcp_idx[cidx][0] == cmodel:
            cstring += " " + get_concat_anom_sst_output_fname(cmodel,
                                                   cmip5_rcp_idx[cidx][1],
                                                   run_type, ref_start, ref_end)
            cidx += 1
        cidx += 1
        out_name = get_concat_anom_sst_model_mean_fname(cmodel, run_type, ref_start, ref_end)
        cdo.ensmean(input=cstring, output=out_name)

#############################################################################

def load_3d_file(fname, var_name="sst"):
    # get the longitude, latitude and attributes
    in_fh = netcdf_file(fname, "r")
    sst_var = in_fh.variables[var_name]
    sst_data = sst_var[:]
    # find lat and lon name
    for k in in_fh.variables.keys():
        if "lat" in k:
            lat_name = k
        if "lon" in k:
            lon_name = k
        if "time" in k:
            t_name = k
    lats_var = in_fh.variables[lat_name]
    lons_var = in_fh.variables[lon_name]
    t_var = in_fh.variables[t_name]

    # mask the array
    attrs = sst_var._attributes
    if "missing_value" in attrs.keys():
        mv = attrs["missing_value"]
    elif "_FillValue" in attrs.keys():
        mv = attrs["_FillValue"]
    sst_data = numpy.ma.masked_equal(sst_data, mv)
    return sst_data, lons_var, lats_var, attrs, t_var

#############################################################################

def save_3d_file(out_fname, out_data, out_lon_var, out_lat_var, out_attrs, out_t_var, out_vname="sst"):
    # open the file
    out_fh = netcdf_file(out_fname, "w")
    # create latitude and longitude dimensions - copy from the ens_mean file
    lon_data = numpy.array(out_lon_var[:])
    lat_data = numpy.array(out_lat_var[:])
    time_data = numpy.array(out_t_var[:])
    
    lon_out_dim = out_fh.createDimension("longitude", lon_data.shape[0])
    lat_out_dim = out_fh.createDimension("latitude", lat_data.shape[0])
    lon_out_var = out_fh.createVariable("longitude", lon_data.dtype, ("longitude",))
    lat_out_var = out_fh.createVariable("latitude", lat_data.dtype, ("latitude",))
    time_out_dim = out_fh.createDimension("time", time_data.shape[0])
    time_out_var = out_fh.createVariable("time", time_data.dtype, ("time",))

    lon_out_var[:] = lon_data
    lat_out_var[:] = lat_data
    time_out_var[:] = time_data
    
    lon_out_var._attributes = out_lon_var._attributes
    lat_out_var._attributes = out_lat_var._attributes
    time_out_var._attributes = out_t_var._attributes
    
    data_out_var = out_fh.createVariable(out_vname, out_data.dtype, ("time", "latitude", "longitude"))
    data_out_var[:] = out_data[:]
    data_out_var._attributes = out_attrs
    out_fh.close()  

#############################################################################

def create_concat_sst_anoms(run_type, ref_start, ref_end, start_idx, end_idx, monthly):
    # Build a time series of concatenated sst anomalies (wrt 1986->2005)
    # from 1899->2100

    # get the filtered set of cmip5 models / runs
    cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)
    n_ens = len(cmip5_rcp_idx)

    # create the cdo object
    cdo = Cdo()
    
    # variable name
    sst_var_name = "tos"
    sic_var_name = "sic"
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    
    for idx in range(start_idx, end_idx):
        print cmip5_rcp_idx[idx][0]
        # get the tos filenames for the rcp and historical simulation
        sst_rcp_fname = get_cmip5_tos_fname(run_type, cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])
        sst_histo_fname = get_cmip5_tos_fname("historical", cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])
        
        sic_rcp_fname = get_cmip5_sic_fname(run_type, cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])
        sic_histo_fname = get_cmip5_sic_fname("historical", cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1])
        
        if "HadGEM2-" in cmip5_rcp_idx[idx][0]:
            rcp_sy -= 1     # met office files run from 2005/12-> for rcp scenarios
        
        sst_rcp_remap_string   = create_remapped_field(sst_rcp_fname, rcp_sy, rcp_ey, sst_var_name, monthly)
        sst_histo_remap_string = create_remapped_field(sst_histo_fname, histo_sy, histo_ey, sst_var_name, monthly)
        sic_rcp_remap_string   = create_remapped_field(sic_rcp_fname, rcp_sy, rcp_ey, sic_var_name, monthly)
        sic_histo_remap_string = create_remapped_field(sic_histo_fname, histo_sy, histo_ey, sic_var_name, monthly)

        sst_rcp_remap_temp = cdo.addc(0,input=sst_rcp_remap_string)
        sst_histo_remap_temp = cdo.addc(0,input=sst_histo_remap_string)
        sic_rcp_remap_temp = cdo.addc(0,input=sic_rcp_remap_string)
        sic_histo_remap_temp = cdo.addc(0,input=sic_histo_remap_string)
        
        # cat the files together
        cdo.cat(input=sst_histo_remap_temp + " " + sst_rcp_remap_temp, output="tmp_sst.nc")
        cdo.cat(input=sic_histo_remap_temp + " " + sic_rcp_remap_temp, output="tmp_sic.nc")

        # fix the file to replace missing value in sst with -1.8 if sic > 0
        sst_fh = netcdf_file("tmp_sst.nc", 'r')
        sic_fh = netcdf_file("tmp_sic.nc", 'r')
        sst_var = sst_fh.variables[sst_var_name]
        sst_data = numpy.array(sst_var[:])
        lon_var = sst_fh.variables["lon"]
        lat_var = sst_fh.variables["lat"]
        t_var = sst_fh.variables["time"]
        sic_data = sic_fh.variables[sic_var_name][:]
        mv = sst_var._attributes["_FillValue"]
        
        # replace
#        for t in range(0, sic_data.shape[0]):
#            sic_data_idx = numpy.where(sic_data[t] > 1)
#            sst_data[t][sic_data_idx] = 273.15 - (1.8 * sic_data[t][sic_data_idx] * 0.01)
        sst_fh.close()
        sic_fh.close()
        
        # save the file
        save_3d_file("tmp_sst2.nc", sst_data, lon_var, lat_var, sst_var._attributes, t_var, sst_var_name)

        # add the lsm from hadisst
        lsm_path = "/soge-home/staff/coml0118/LSM/HadISST2_lsm.nc"
        cdo.add(input=" -smooth9 tmp_sst2.nc " + lsm_path, output="tmp_sst3.nc")

        # calculate the temporary reference file
        tmp_ref_name = create_tmp_ref_field("tmp_sst3.nc", ref_start, ref_end, sst_var_name, monthly)
        
        # calculate the timeseries of anomalies for both the historical
        # and RCP runs as running decadal means
        out_path = get_concat_anom_sst_output_fname(cmip5_rcp_idx[idx][0], 
                                                    cmip5_rcp_idx[idx][1], 
                                                    run_type, ref_start, ref_end,
                                                    monthly)
        anom_string = get_calc_anom_string(tmp_ref_name,
                                           "tmp_sst3.nc", histo_sy, rcp_ey, sst_var_name,
                                           monthly)
        cdo.addc(0, input=anom_string, output=out_path)
        os.remove("tmp_sst.nc")
        os.remove("tmp_sic.nc")
        os.remove("tmp_sst2.nc")
        os.remove("tmp_sst3.nc")
        os.remove("tmp_ref_sst.nc")

#############################################################################

def running_gradient_3D_monthly(data, P, mv):
    # run the running gradient on each month independently
    out_data = numpy.zeros(data.shape, data.dtype)
    for m in range(0,12):
        out_data[m::12] = running_gradient_3D(data[m::12], P, mv)
    return out_data

#############################################################################

def smooth_concat_sst_anoms(run_type, ref_start, ref_end, start_idx, end_idx, monthly):
    # Smooth the anomaly files created in the above function using the
    # running mean, running gradient filter
    # also subtract the ensemble mean
    # get the filtered set of cmip5 models / runs
    cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)
    n_ens = len(cmip5_rcp_idx)
    
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    
    # get the ensemble mean
    ens_mean_fname = get_concat_anom_sst_ens_mean_fname(run_type, ref_start, ref_end, monthly)
    ens_mean_fh = netcdf_file(ens_mean_fname)
    ens_mean = ens_mean_fh.variables["tos"][:].byteswap().newbyteorder()
    ens_mean_fh.close()
        
    for idx in range(start_idx, end_idx):
        print cmip5_rcp_idx[idx][0]
        concat_anom_fname = get_concat_anom_sst_output_fname(cmip5_rcp_idx[idx][0], 
                                                             cmip5_rcp_idx[idx][1],
                                                             run_type, ref_start, ref_end,
                                                             monthly)
                                                             
        # read the file in and extract the ssts
        sst_data, lons_var, lats_var, attrs, t_var = load_3d_file(concat_anom_fname, "tos")
        sst_data = sst_data.byteswap().newbyteorder()
        depart_from_ens_mean = sst_data - ens_mean
        P = 40
        mv = attrs["_FillValue"]
        if monthly:
            smoothed_data = running_gradient_3D_monthly(depart_from_ens_mean, P, mv)
        else:
            smoothed_data = running_gradient_3D(depart_from_ens_mean, P, mv)
        # save the data
        out_fname = get_concat_anom_sst_smooth_fname(cmip5_rcp_idx[idx][0], 
                                                     cmip5_rcp_idx[idx][1],
                                                     run_type, ref_start, ref_end,
                                                     monthly)
        save_3d_file(out_fname, smoothed_data, lons_var, lats_var, attrs, t_var)

#############################################################################

def smooth_concat_sst_anoms_model_means(run_type, ref_start, ref_end, start_idx, end_idx):
    # Smooth the anomaly files created in the above function using the
    # running mean, running gradient filter
    # also subtract the ensemble mean
    # get the filtered set of cmip5 models / runs
    cmip5_rcp_idx = read_cmip5_model_mean_index_file(run_type, ref_start, ref_end)
    n_ens = len(cmip5_rcp_idx)
    
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    
    # get the ensemble mean
    ens_mean_fname = get_concat_anom_sst_ens_mean_fname(run_type, ref_start, ref_end)
    ens_mean_fh = netcdf_file(ens_mean_fname)
    ens_mean = ens_mean_fh.variables["tos"][:].byteswap().newbyteorder()
    ens_mean_fh.close()
        
    for idx in range(start_idx, end_idx):
        print cmip5_rcp_idx[idx][0]
        concat_anom_fname = get_concat_anom_sst_output_fname(cmip5_rcp_idx[idx][0], 
                                                             cmip5_rcp_idx[idx][1],
                                                             run_type, ref_start, ref_end)
                                                             
        # read the file in and extract the ssts
        sst_data, lons_var, lats_var, attrs, t_var = load_3d_file(concat_anom_fname, "tos")
        sst_data = sst_data.byteswap().newbyteorder()
        depart_from_ens_mean = sst_data - ens_mean
        P = 40
        mv = attrs["_FillValue"]
        smoothed_data = running_gradient_3D(depart_from_ens_mean, P, mv)
        # save the data
        out_fname = get_concat_anom_sst_smooth_model_mean_fname(cmip5_rcp_idx[idx][0], 
                                                     cmip5_rcp_idx[idx][1],
                                                     run_type, ref_start, ref_end)
        save_3d_file(out_fname, smoothed_data, lons_var, lats_var, attrs, t_var)

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    start_idx = 0
    end_idx = 0
    monthly = False
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:i:j:m',
                               ['run_type=', 'ref_start=', 'ref_end=',
                                'st_idx=', 'ed_idx=', 'monthly'])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--st_idx', '-i']:
            start_idx = int(val)
        if opt in ['--ed_idx', '-j']:
            end_idx = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True

    model_mean = False
    if model_mean:
        create_concat_sst_anoms_model_means(run_type, ref_start, ref_end)
        smooth_concat_sst_anoms_model_means(run_type, ref_start, ref_end, start_idx, end_idx)
    else:
        # run this on ouce-linux-01
#        create_concat_sst_anoms(run_type, ref_start, ref_end, start_idx, end_idx, monthly)
        # copy files across then run this on local machine
#        create_concat_sst_anoms_ens_mean(run_type, ref_start, ref_end, monthly)
        # run this on local machine
#        create_concat_sst_anoms_ens_mean_smoothed(run_type, ref_start, ref_end, monthly)
        # run this on local machine
        smooth_concat_sst_anoms(run_type, ref_start, ref_end, start_idx, end_idx, monthly)
