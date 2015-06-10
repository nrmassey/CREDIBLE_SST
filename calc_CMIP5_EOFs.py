#! /usr/bin/env python  
#############################################################################
#
# Program : calc_CMIP5_EOFs.py
# Author  : Neil Massey
# Purpose : Calculate the EOFs of the Sea-Surface Temperature warming 
#           patterns for the CMIP5 ensembles that occur within a certain
#           decade. create_CMIP5_sst_anoms.py is required to run before
#           running this program.
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
# Date    : 16/02/15
#
#############################################################################

import os, sys, getopt
from cmip5_functions import get_cmip5_tos_fname, get_output_directory, save_3d_file, load_data
from filter_cmip5_members import read_cmip5_index_file, read_cmip5_model_mean_index_file
from create_CMIP5_sst_anoms import get_concat_anom_sst_smooth_fname, get_start_end_periods, get_concat_anom_sst_smooth_model_mean_fname

from netcdf_file import *
import numpy
from eofs.standard import Eof
import scipy.stats

import matplotlib.pyplot as plt

#############################################################################

def get_file_suffix(run_type, ref_sy, ref_ey, year):
    suff = run_type+"_"+str(ref_sy)+"_"+str(ref_ey)+"_y"+str(year)
    return suff

#############################################################################
    
def get_cmip5_EOF_filename(run_type, ref_sy, ref_ey, year, model_mean=False, monthly=False):
    if model_mean:
        suffix = "_mm"
    else:
        suffix = ""
    out_dir = get_output_directory(run_type, ref_sy, ref_ey, year) 
    fname = out_dir+"/cmip5_EOF_" + get_file_suffix(run_type, ref_sy, ref_ey, year) + suffix + "_eof"
    if monthly:
        fname += "_mon"
    fname += ".nc"
    return fname

#############################################################################

def get_cmip5_eigen_filename(run_type, ref_sy, ref_ey, year, model_mean=False, monthly=False):
    if model_mean:
        suffix = "_mm"
    else:
        suffix = ""
    out_dir = get_output_directory(run_type, ref_sy, ref_ey, year) 
    fname = out_dir+"/cmip5_EOF_" + get_file_suffix(run_type, ref_sy, ref_ey, year) + suffix + "_ev"
    if monthly:
        fname += "_mon"
    fname += ".nc"
    return fname

#############################################################################

def get_cmip5_PC_filename(run_type, ref_sy, ref_ey, year, model_mean=False, monthly=False):
    if model_mean:
        suffix = "_mm"
    else:
        suffix = ""
    out_dir = get_output_directory(run_type, ref_sy, ref_ey, year) 
    fname = out_dir+"/cmip5_EOF_" + get_file_suffix(run_type, ref_sy, ref_ey, year) + suffix + "_pc"
    if monthly:
        fname += "_mon"
    fname += ".nc"
    return fname

#############################################################################

def get_cmip5_proj_PC_filename(run_type, ref_sy, ref_ey, year, model_mean=False, monthly=False):
    if model_mean:
        suffix = "_mm"
    else:
        suffix = ""
    out_dir = get_output_directory(run_type, ref_sy, ref_ey, year) 
    fname = out_dir+"/cmip5_EOF_" + get_file_suffix(run_type, ref_sy, ref_ey, year) + suffix + "_proj_pc"
    if monthly:
        fname += "_mon"
    fname += ".nc"
    return fname

#############################################################################

def get_cmip5_proj_PC_scale_filename(run_type, ref_start, ref_end, eof_year, model_mean=False, monthly=False):
    out_dir = get_output_directory(run_type, ref_start, ref_end, eof_year)
    if model_mean:
        suffix = "_mm"
    else:
        suffix = ""
    fname = out_dir+"/cmip5_EOF_" + get_file_suffix(run_type, ref_start, ref_end, eof_year) + suffix + "_proj_pc_scale"
    if monthly:
        fname += "_mon"
    fname += ".nc"
    return fname

#############################################################################

def save_eofs(out_fname, out_data, out_attrs, in_lats, in_lons):
    # open the file
    out_fh = netcdf_file(out_fname, "w")
    # create latitude and longitude dimensions - copy from the ens_mean file
    lon_data = numpy.array(in_lons[:])
    lat_data = numpy.array(in_lats[:])
    lon_out_dim = out_fh.createDimension("longitude", lon_data.shape[0])
    lat_out_dim = out_fh.createDimension("latitude", lat_data.shape[0])
    lon_out_var = out_fh.createVariable("longitude", lon_data.dtype, ("longitude",))
    lat_out_var = out_fh.createVariable("latitude", lat_data.dtype, ("latitude",))
    ens_out_dim = out_fh.createDimension("ensemble_member", out_data.shape[1])
    ens_out_var = out_fh.createVariable("ensemble_member", numpy.dtype('i4'), ("ensemble_member",))
    mon_out_dim = out_fh.createDimension("month", out_data.shape[0])
    mon_out_var = out_fh.createVariable("month", numpy.dtype('i4'), ("month",))
    lon_out_var[:] = lon_data
    lat_out_var[:] = lat_data
    lon_out_var._attributes = in_lons._attributes
    lat_out_var._attributes = in_lats._attributes
    ens_out_var[:] = numpy.arange(0, out_data.shape[1])
    mon_out_var[:] = numpy.arange(0, out_data.shape[0])
    data_out_var = out_fh.createVariable("sst", out_data.dtype, ("month", "ensemble_member", "latitude", "longitude"))
    data_out_var[:] = out_data[:]
    data_out_var._attributes = out_attrs
    out_fh.close() 

#############################################################################

def save_eigenvalues(out_fname, out_data, out_attrs):
    # open the file
    out_fh = netcdf_file(out_fname, "w")
    mon_dim = out_fh.createDimension("month", out_data.shape[0])
    eig_out_dim = out_fh.createDimension("eigenvalue", out_data.shape[1])
    mon_var = out_fh.createVariable("month", numpy.dtype('i4'), ("month",))
    eig_out_var = out_fh.createVariable("eigenvalue", numpy.dtype('i4'), ("eigenvalue",))
    mon_var[:] = numpy.arange(0, out_data.shape[0])
    eig_out_var[:] = numpy.arange(0, out_data.shape[1])
    sst_out_var = out_fh.createVariable("sst", out_data.dtype, ("month", "eigenvalue",))
    sst_out_var[:] = out_data[:]
    sst_out_var._attributes = out_attrs
    out_fh.close()

#############################################################################

def save_pcs(out_fname, out_data, out_attrs):
    # open the file
    out_fh = netcdf_file(out_fname, "w")
    mon_dim = out_fh.createDimension("month", out_data.shape[0])
    ens_mem_dim = out_fh.createDimension("ensemble_member", out_data.shape[1])
    pc_out_dim  = out_fh.createDimension("principal_component", out_data.shape[2])
    ens_mem_var = out_fh.createVariable("ensemble_member", numpy.dtype('i4'), ("ensemble_member",))
    mon_var = out_fh.createVariable("month", numpy.dtype('i4'), ("month",))
    pc_out_var  = out_fh.createVariable("principal_component", numpy.dtype('i4'), ("principal_component",))
    mon_var[:] = numpy.arange(0, out_data.shape[0])
    ens_mem_var[:] = numpy.arange(0, out_data.shape[1])
    pc_out_var[:] = numpy.arange(0, out_data.shape[2])
    out_var = out_fh.createVariable("sst", out_data.dtype, ("month", "ensemble_member", "principal_component"))
    out_var[:] = out_data[:]
    out_fh.close()

#############################################################################

def save_pcs_ts(pc_fname, out_data, n_eofs, t_var, n_ens):
    # create the file with ensemble_member, eof_number and time
    # ensemble member is the unlimited dimension
    out_fh = netcdf_file(pc_fname, 'w')
    ens_mem_dim = out_fh.createDimension("ensemble_member", n_ens)
    pc_out_dim  = out_fh.createDimension("principal_component", n_eofs)
    t_out_dim   = out_fh.createDimension("time", t_var.shape[0])
    ens_mem_var = out_fh.createVariable("ensemble_member", numpy.dtype('i4'), ("ensemble_member",))
    pc_out_var  = out_fh.createVariable("principal_component", numpy.dtype('i4'), ("principal_component",))
    t_out_var   = out_fh.createVariable("time", t_var[:].dtype, ("time",))
    ens_mem_var[:] = numpy.arange(0, n_ens)
    pc_out_var[:]  = numpy.arange(0, n_eofs)
    t_out_var[:]   = t_var[:]
    t_out_var._attributes = t_var._attributes
    out_var = out_fh.createVariable("sst_pc", numpy.dtype('f4'), ("ensemble_member", "time", "principal_component",))
    out_var[:] = out_data
    out_fh.close()

#############################################################################

def save_pcs_scale(fname, offset, scale, t_var):
    out_fh = netcdf_file(fname, 'w')
    n_eofs = scale.shape[1]
    pc_out_dim  = out_fh.createDimension("principal_component", n_eofs)
    t_out_dim   = out_fh.createDimension("time", t_var.shape[0])
    pc_out_var  = out_fh.createVariable("principal_component", numpy.dtype('i4'), ("principal_component",))
    t_out_var   = out_fh.createVariable("time", t_var[:].dtype, ("time",))
    pc_out_var[:]  = numpy.arange(0, n_eofs)
    t_out_var[:]   = t_var[:]
    t_out_var._attributes = t_var._attributes
    scale_var  = out_fh.createVariable("sst_scale", numpy.dtype('f4'), ("time", "principal_component",))
    offset_var = out_fh.createVariable("sst_offset", numpy.dtype('f4'), ("time", "principal_component",))
    scale_var[:] = scale
    offset_var[:] = offset
    out_fh.close()

#############################################################################

def calc_EOFs(run_type, ref_start, ref_end, eof_year, model_mean=False, monthly=False):
    # get the filtered set of cmip5 models / runs
    if model_mean:
        cmip5_rcp_idx = read_cmip5_model_mean_index_file(run_type, ref_start, ref_end)
    else:
        cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)
    n_ens = len(cmip5_rcp_idx)
    
    # for each ensemble member get the smoothed anomalies that are also
    # anomalies from the ensemble mean anomaly
    # get the anomalies in the decade centred on the eof_year (-5/+4 inc.)
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    # calculate the start and end index into the netCDF files
    si = eof_year-5-histo_sy
    ei = eof_year+5-histo_sy
    if monthly:
        si *= 12
        ei *= 12
    
    # create the storage
    ensemble = numpy.ma.zeros([n_ens, ei-si, 180, 360], 'f')

    for idx in range(0, n_ens):
        if model_mean:
            concat_smooth_fname = get_concat_anom_sst_smooth_model_mean_fname(cmip5_rcp_idx[idx][0], 
                                                                   cmip5_rcp_idx[idx][1], 
                                                                   run_type, ref_start, ref_end,
                                                                   monthly)
        else:
            concat_smooth_fname = get_concat_anom_sst_smooth_fname(cmip5_rcp_idx[idx][0], 
                                                                   cmip5_rcp_idx[idx][1], 
                                                                   run_type, ref_start, ref_end,
                                                                   monthly)
        # open the netCDF file
        fh = netcdf_file(concat_smooth_fname, 'r')
        # get lon / lat / missing value
        if idx == 0:
            lons_var = fh.variables["longitude"]
            lats_var = fh.variables["latitude"]
            attrs = fh.variables["sst"]._attributes
            mv = attrs["_FillValue"]
        # get the data and add to the ensemble array
        var = fh.variables["sst"]
        data = var[si:ei]
        ensemble[idx] = numpy.ma.masked_equal(data, mv)
        fh.close()
    # now we have the option of monthly EOFs - return a list of Eof solvers    
    eof_solvers = []
    if monthly:
        for m in range(0,12):
            # get the monthly decadal data
            ens_mon_data = ensemble[:,m::12,:,:]
            # take the decadal mean
            ens_mon_dec_mn = numpy.mean(ens_mon_data, axis=1).squeeze()
            # take the EOFs
            coslat = numpy.cos(numpy.deg2rad(lats_var[:])).clip(0., 1.)
            wgts = numpy.sqrt(coslat)[..., numpy.newaxis]
            eof_solver = Eof(ens_mon_dec_mn, center=False, weights=wgts)
            eof_solvers.append(eof_solver)
    else:
        # do the decadal mean
        ensemble_dec_mn = numpy.mean(ensemble, axis=1).squeeze()
        # take the eofs
        coslat = numpy.cos(numpy.deg2rad(lats_var[:])).clip(0., 1.)
        wgts = numpy.sqrt(coslat)[..., numpy.newaxis]
        eof_solver = Eof(ensemble_dec_mn, center=False, weights=wgts)
        eof_solvers.append(eof_solver)
    
    return eof_solvers
    
#############################################################################

def calc_CMIP5_EOFs(run_type, ref_start, ref_end, eof_year, model_mean=False, monthly=False):

    # get the lats / lons from the first ensemble member
    if model_mean:
        cmip5_rcp_idx = read_cmip5_model_mean_index_file(run_type, ref_start, ref_end)
        concat_smooth_fname = get_concat_anom_sst_smooth_model_mean_fname(cmip5_rcp_idx[0][0], 
                                                               cmip5_rcp_idx[0][1], 
                                                               run_type, ref_start, ref_end,
                                                               monthly)
    else:
        cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)
        concat_smooth_fname = get_concat_anom_sst_smooth_fname(cmip5_rcp_idx[0][0], 
                                                               cmip5_rcp_idx[0][1], 
                                                               run_type, ref_start, ref_end,
                                                               monthly)
                                                               
    fh = netcdf_file(concat_smooth_fname, 'r')
    lons_var = fh.variables["longitude"]
    lats_var = fh.variables["latitude"]
    attrs = fh.variables["sst"]._attributes
    mv = attrs["_FillValue"]

    eof_solvers = calc_EOFs(run_type, ref_start, ref_end, eof_year, model_mean, monthly)

    # n_eofs = None - get all eofs
    n_eofs = None
    # get the principal components, eofs and eigenvalues
    pcs = []
    eofs = []
    evs = []
    for eof_solver in eof_solvers:
        pcs.append(eof_solver.pcs(pcscaling=0, npcs=n_eofs))
        eofs.append(eof_solver.eofs(eofscaling=0, neofs=n_eofs))
        evs.append(eof_solver.eigenvalues(neigs=n_eofs))
    # convert the lists to numpy arrays
    pcs = numpy.array(pcs)
    eofs = numpy.array(eofs)
    print eofs.shape
    evs = numpy.array(evs)
    # save the principal components
    pcs_fname = get_cmip5_PC_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    save_pcs(pcs_fname, pcs, attrs)
    # save the eigenvalues
    eig_fname = get_cmip5_eigen_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    save_eigenvalues(eig_fname, evs, attrs)
    # save the Eofs
    eof_fname = get_cmip5_EOF_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    save_eofs(eof_fname, eofs, attrs, lats_var, lons_var)

#############################################################################

def calc_CMIP5_PC_proj(run_type, ref_start, ref_end, eof_year, model_mean=False, monthly=False):
    # calculate the projection of the smoothed anomalies onto the EOFs of
    # the decadal mean smoothed anomalies of the eof_year, for all ensemble
    # members in CMIP5 for the historical and rcp scenarios
    
    # calculate the eof_solvers using the above function
    eof_solvers = calc_EOFs(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    
    # now loop over every ensemble member and project the smoothed anomalies 
    # onto the eofs
    # get the filtered set of cmip5 models / runs
    
    if model_mean:    
        cmip5_rcp_idx = read_cmip5_model_mean_index_file(run_type, ref_start, ref_end)
    else:
        cmip5_rcp_idx = read_cmip5_index_file(run_type, ref_start, ref_end)
    n_ens = len(cmip5_rcp_idx)
    
    neofs = None
    
    for idx in range(0, n_ens):
        print cmip5_rcp_idx[idx][0], cmip5_rcp_idx[idx][1]
        if model_mean:
            concat_smooth_fname = get_concat_anom_sst_smooth_model_mean_fname(cmip5_rcp_idx[idx][0], 
                                                               cmip5_rcp_idx[idx][1], 
                                                               run_type, ref_start, ref_end)
        else:
            concat_smooth_fname = get_concat_anom_sst_smooth_fname(cmip5_rcp_idx[idx][0], 
                                                               cmip5_rcp_idx[idx][1], 
                                                               run_type, ref_start, ref_end,
                                                               monthly)
        # open the netCDF file
        fh = netcdf_file(concat_smooth_fname, 'r')
        
        if idx == 0:
            t_var = fh.variables["time"]
            attrs = fh.variables["sst"]._attributes
            mv = attrs["_FillValue"]
        # get the data and add to the ensemble array
        var = fh.variables["sst"]
        sst_data = numpy.ma.masked_equal(var[:], mv)
        
        # create the output array
        if idx == 0:
            out_data = numpy.zeros([n_ens, sst_data.shape[0], n_ens], 'f')

        fh.close()

        # project the sst anomalies onto the 2050 EOFs
        # if monthly then project each timeseries of monthly data onto the monthly
        # eof from the monthly eof solver
        if monthly:
            for m in range(0, 12):
                mon_data = sst_data[m::12]
                mon_proj_pcs = eof_solvers[m].projectField(mon_data, neofs)
                out_data[idx,m::12,:] = mon_proj_pcs
        else:
            proj_pcs = eof_solvers[0].projectField(sst_data, neofs)
            out_data[idx,:,:] = proj_pcs

    out_name = get_cmip5_proj_PC_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    save_pcs_ts(out_name, out_data, n_ens, t_var, n_ens)
    print out_name

#############################################################################

def calc_CMIP5_PC_proj_scaling(run_type, ref_start, ref_end, eof_year, model_mean=False, monthly=False):
    # load the previously calculated PCs
    pc_fname = get_cmip5_proj_PC_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    pcs = load_data(pc_fname, "sst_pc")
    fh = netcdf_file(pc_fname, 'r')
    t_var = fh.variables["time"]
    
    # get the anomalies in the decade centred on the eof_year (-5/+4 inc.)
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    # calculate the start and end index into the netCDF files
    ri = eof_year-histo_sy
    if monthly:
        ri *= 12
    
    # For each year regress the pcs on the pcs in 2050 to determine the
    # relationship between the eof_year and the years in the time series
    npcs  = pcs.shape[2]
    n_t   = pcs.shape[1]
    offset = numpy.zeros([n_t, npcs], 'f')
    scale  = numpy.zeros([n_t, npcs], 'f')
    
    for t in range(0, n_t):
        tts_pcs = pcs[:,t,:].squeeze()
        # which month are we in?
        if monthly:
            mon = t % 12
        else:
            ref_pcs = pcs[:,ri,:].squeeze()
        for pc in range(0, npcs):
            # get the reference pcs in the eof_year - if monthly we want to get 12 
            # reference pcs
            if monthly:
                ref_pcs = pcs[:,ri+mon,:].squeeze()
            s, i, r, p, err = scipy.stats.linregress(ref_pcs[:,pc], tts_pcs[:,pc])
            scale [t,pc] = s
            offset[t,pc] = i
    
    # save the scalings
    out_name = get_cmip5_proj_PC_scale_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    save_pcs_scale(out_name, offset, scale, t_var)

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    eof_year = -1
    monthly = False
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:f:m',
                               ['run_type=', 'ref_start=', 'ref_end=',
                                'eof_year=', 'monthly'])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--eof_year', '-f']:
            eof_year = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True
    
    model_mean=False
    
#    calc_CMIP5_EOFs(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
#    calc_CMIP5_PC_proj(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    calc_CMIP5_PC_proj_scaling(run_type, ref_start, ref_end, eof_year, model_mean, monthly)