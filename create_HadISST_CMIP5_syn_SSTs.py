#! /usr/bin/env python  
#############################################################################
#
# Program : create_HadISST_CMIP5_syn_SSTs.py
# Author  : Neil Massey
# Purpose : Create a single synthetic SST from everything that has been
#           computed before hand
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
from calc_CMIP5_EOFs import get_cmip5_EOF_filename, get_cmip5_proj_PC_scale_filename
from create_CMIP5_syn_PCs import get_syn_SST_PCs_filename
from create_CMIP5_sst_anoms import get_concat_anom_sst_ens_mean_smooth_fname, get_start_end_periods
from create_HadISST_sst_anoms import get_HadISST_smooth_fname, get_HadISST_reference_fname, get_HadISST_month_smooth_filename
from cmip5_functions import load_data, reconstruct_field, calc_GMSST, load_sst_data
from calc_HadISST_residual_EOFs import get_HadISST_residual_PCs_fname, get_HadISST_residual_EOFs_fname, get_HadISST_monthly_residual_EOFs_fname, get_HadISST_monthly_residual_PCs_fname, get_HadISST_annual_cycle_residuals_fname
import numpy
from netcdf_file import *
from ARN import ARN
import pyximport
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
from zonal_smoother import *

#############################################################################

def get_output_directory(run_type, ref_start, ref_end, eof_year):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_dir = "HadISST_"+str(histo_sy)+"_"+str(histo_ey)+"_"+\
              run_type+"_"+str(rcp_sy)+"_"+str(rcp_ey)+\
              "_r"+str(ref_start)+"_"+str(ref_end)+\
              "_y"+str(eof_year)
    if not os.path.exists("../CREDIBLE_output/output/"+out_dir):
        os.mkdir("../CREDIBLE_output/output/"+out_dir)
    return out_dir

#############################################################################

def get_syn_sst_filename(run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, monthly):
     # build the filename for the synthetic SSTs
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    if intvarmode == 0:
        intvarstr = "varnone"
    elif intvarmode == 1:
        intvarstr = "varyear"
    elif intvarmode == 2:
        intvarstr = "varmon"
    out_dir = get_output_directory(run_type, ref_start, ref_end, eof_year)
    out_name = out_dir + "_f"+str(eof_year)+"_n"+str(neofs)+"_a"+str(sample)+"_"+intvarstr+"_ssts"
    if monthly:
        out_name += "_mon"
    out_name += ".nc"
    ppath = "../CREDIBLE_output/output/"+out_dir + "/" + intvarstr + "/sst/"
    if not os.path.exists(ppath):
        os.mkdir(ppath)
    return ppath + out_name

#############################################################################

def create_yearly_intvar(run_type, ref_start, ref_end, n_pcs=20, run_n=400):
    # load in the PCs and EOFs
    histo_sy = 1899
    histo_ey = 2010
    
    yearly_pc_fname = get_HadISST_residual_PCs_fname(histo_sy, histo_ey, run_n)
    yearly_pcs = load_data(yearly_pc_fname)
    
    yearly_eof_fname = get_HadISST_residual_EOFs_fname(histo_sy, histo_ey, run_n)
    yearly_eofs = load_sst_data(yearly_eof_fname, "sst")
    
    # get the number of years to predict the PCs for and create the
    # storage
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    n_yrs = rcp_ey - histo_sy
    predicted_pcs = numpy.zeros([n_yrs+1, n_pcs], yearly_pcs.dtype)
    
    # fit an AR process to the first ~20 pcs
    for pc in range(0, n_pcs):
        # create the model
        arn = ARN(yearly_pcs[:,pc])
        # fit the model to the data
        res = arn.fit()
        arp = res.k_ar
        # create a timeseries of predicted values
        predicted_pcs[:,pc] = arn.predict(res.params, noise='all', dynamic=True, start=arp, end=n_yrs+arp)
        
    # reconstruct the field and return
    # weights for reconstruction
    coslat = numpy.cos(numpy.deg2rad(numpy.arange(89.5, -90.5, -1)).clip(0., 1.))
    wgts = numpy.sqrt(coslat)[..., numpy.newaxis]
    # reconstruct field
    yearly_intvar = reconstruct_field(predicted_pcs, yearly_eofs[:n_pcs], n_pcs, wgts)
    return yearly_intvar

#############################################################################

def create_monthly_intvar(run_type, ref_start, ref_end, n_pcs=20, run_n=400):
    # load in the PCs and EOFs
    histo_sy = 1899
    histo_ey = 2010
    
    monthly_pc_fname = get_HadISST_monthly_residual_PCs_fname(histo_sy, histo_ey, run_n)
    monthly_pcs = load_data(monthly_pc_fname)
    
    monthly_eof_fname = get_HadISST_monthly_residual_EOFs_fname(histo_sy, histo_ey, run_n)
    monthly_eofs = load_sst_data(monthly_eof_fname, "sst")
    
    # load in the annual cycle
    resid_ac_fname = get_HadISST_annual_cycle_residuals_fname(histo_sy, histo_ey, ref_start, ref_end, run_n)
    resid_ac = load_sst_data(resid_ac_fname, "sst")
    
    # get the number of months to predict the PCs for and create the storage
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    n_mnths = 12*(rcp_ey - histo_sy)
    predicted_pcs = numpy.zeros([n_mnths+12, n_pcs], monthly_pcs.dtype)

    # fit an AR process to the first ~20 pcs
    for pc in range(0, n_pcs):
        # create the model
        arn = ARN(monthly_pcs[:,pc])
        # fit the model to the data
        res = arn.fit()
        arp = res.k_ar
        # create a timeseries of predicted values
        predicted_pcs[:,pc] = arn.predict(res.params, noise='all', dynamic=True, start=arp, end=n_mnths+arp+11)

    # reconstruct the field and return
    # weights for reconstruction
    coslat = numpy.cos(numpy.deg2rad(numpy.arange(89.5, -90.5, -1)).clip(0., 1.))
    wgts = numpy.sqrt(coslat)[..., numpy.newaxis]
    # reconstruct the field
    monthly_intvar = reconstruct_field(predicted_pcs, monthly_eofs[:n_pcs], n_pcs, wgts)
    return monthly_intvar
    
#############################################################################

def save_3d_file(out_fname, out_data, out_lon_var, out_lat_var, out_attrs, t_data, t_attrs, out_vname="sst"):
    # open the file
    out_fh = netcdf_file(out_fname, "w")
    # create latitude and longitude dimensions - copy from the ens_mean file
    lon_data = numpy.array(out_lon_var[:])
    lat_data = numpy.array(out_lat_var[:])
    
    lon_out_dim = out_fh.createDimension("longitude", lon_data.shape[0])
    lat_out_dim = out_fh.createDimension("latitude", lat_data.shape[0])
    lon_out_var = out_fh.createVariable("longitude", lon_data.dtype, ("longitude",))
    lat_out_var = out_fh.createVariable("latitude", lat_data.dtype, ("latitude",))
    time_out_dim = out_fh.createDimension("time", t_data.shape[0])
    time_out_var = out_fh.createVariable("time", t_data.dtype, ("time",))

    lon_out_var[:] = lon_data
    lat_out_var[:] = lat_data
    time_out_var[:] = t_data
    
    lon_out_var._attributes = out_lon_var._attributes
    lat_out_var._attributes = out_lat_var._attributes
    time_out_var._attributes = t_attrs
    
    data_out_var = out_fh.createVariable(out_vname, out_data.dtype, ("time", "latitude", "longitude"))
    data_out_var[:] = out_data[:]
    data_out_var._attributes = out_attrs
    out_fh.close()
    
#############################################################################

def fit_mean_to_likely(ens_mean, monthly=True):
    # scale the ensemble mean so that the GMSST matches the AR5 likely range
    # between 2016 and 2035.
    # first need the scaling factors to convert GMT to GMSST.  These are
    # calculated from a linear regression between the GMT and GMSST of the
    # CMIP5 models for the RCP4.5 and historical scenarios.

    # monthly scaling factor
    if monthly:
        M = 12
    else:
        M = 1

    # plot the AR5 fig 11.25 likely range
    # first calc in GMT
    grad_min = (0.3-0.16)/(2025.5-2009.0)
    grad_max = (0.7-0.16)/(2025.5-2009.0)

    # convert to gmsst using the values of slope and intercept computed
    # by regressing the tos anomaly onto tas anomaly in CMIP5 ensemble members
    
    slope = 0.669
    intercept = 0.017

    gmsst_min_2016 = (grad_min*(2016.0-2009.0)+0.16-1.0) * slope + intercept
    gmsst_max_2016 = (grad_max*(2016.0-2009.0)+0.16+1.0) * slope + intercept
    gmsst_min_2035 = (grad_min*(2035.0-2009.0)+0.16-1.0) * slope + intercept
    gmsst_max_2035 = (grad_max*(2035.0-2009.0)+0.16+1.0) * slope + intercept

    # calculate the middle of the likely range
    likely_gmsst_mean_2016 = (gmsst_min_2016+gmsst_max_2016) * 0.5
    likely_gmsst_mean_2035 = (gmsst_min_2035+gmsst_max_2035) * 0.5
    
    # what is the ratio of the gmsst ens_mean to gmsst_mean at 2016 and 2035
    ens_mean_2016 = ens_mean[(2016-1899)*M]
    ens_mean_2016 = numpy.reshape(ens_mean_2016, [1, ens_mean_2016.shape[0], ens_mean_2016.shape[1]])
    ens_mean_2035 = ens_mean[(2035-1899)*M]
    ens_mean_2035 = numpy.reshape(ens_mean_2035, [1, ens_mean_2035.shape[0], ens_mean_2035.shape[1]])
    gmsst_ratio_2016 = (likely_gmsst_mean_2016 / calc_GMSST(ens_mean_2016)).squeeze()
    gmsst_ratio_2035 = (likely_gmsst_mean_2035 / calc_GMSST(ens_mean_2035)).squeeze()
    # create a linear interpolation between 1.0 in 2005, the 2016 ratio and
    # the 2035 ratio - 2035 ratio will then be applied for rest of timeseries
    gmsst_scaling = numpy.ones([ens_mean.shape[0],1,1], 'f')
    # calculate the interpolated sections
    interp_section_2005_2016 = numpy.interp(numpy.arange(2005,2017,1.0/M), [2005,2016], [1.0, gmsst_ratio_2016])
    interp_section_2016_2035 = numpy.interp(numpy.arange(2017,2035,1.0/M), [2016,2035], [gmsst_ratio_2016, gmsst_ratio_2035])
    gmsst_scaling[(2005-1899)*M:(2017-1899)*M] = interp_section_2005_2016.reshape([interp_section_2005_2016.shape[0],1,1])
    gmsst_scaling[(2017-1899)*M:(2035-1899)*M] = interp_section_2016_2035.reshape([interp_section_2016_2035.shape[0],1,1])
    gmsst_scaling[(2035-1899)*M:] = gmsst_ratio_2035
    ens_mean = ens_mean * gmsst_scaling
    return ens_mean

#############################################################################

# The synthetic SSTs have two distinct time periods:
# 1. The HadISST2 period (1899->2010)
# 2. The CMIP5 period (2006->2100)
#
# To create the SSTs we create the SSTs for each individual time period and
# then interpolate between the two for the overlapping period (2006->2100)
# 
# The methods used to generate each one can be decomposed into:
# (HadISST period) HadISST monthly smoothed data + 
#                   HadISST internal variability (statistically generated)
# (CMIP5 period)   CMIP5 monthly smoothed ensemble mean + 
#                   CMIP5 monthly smoothed long term trend (anomalies from the ensemble mean) +
#                   HadISST 1986->2005 yearly reference pattern +
#                   HadISST internal variability +
#                   HadISST annual cycle
#
# To avoid discontinuity of the internal variability, we generate one long
# timeseries of this to add onto the results at the end

# functions follow to generate each individual component

#############################################################################

def create_hadisst_long_term_timeseries(monthly=True, run_n=400):
    # get the dates
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010
    
    # create the long term trend timeseries from the monthly smoothed hadisst data
    if monthly:
        hadisst_smoothed_fname = get_HadISST_month_smooth_filename(histo_sy, hadisst_ey, run_n)
    else:
        hadisst_smoothed_fname = get_HadISST_smooth_fname(histo_sy, hadisst_ey, run_n)
    
    hadisst_sst = load_sst_data(hadisst_smoothed_fname, "sst")
    
    return hadisst_sst
    
#############################################################################

def correct_cmip5_long_term_mean_timeseries(cmip5_ts, monthly=True, run_n=400):
    # correct the ensemble mean of the CMIP5 ensemble, this is achieved by subtracting the
    # difference between the 5 year mean of HadISST and the 5 year mean of the CMIP5 ensemble 
    # mean for 2006->2010 (the overlap period) from the CMIP5 ensemble mean
    # need to load in the hadisst values to enable the correction
   # get the dates
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010
    
    if monthly:
        hadisst_smoothed_fname = get_HadISST_month_smooth_filename(histo_sy, hadisst_ey, run_n)
    else:
        hadisst_smoothed_fname = get_HadISST_smooth_fname(histo_sy, hadisst_ey, run_n)
    hadisst_sst = load_sst_data(hadisst_smoothed_fname, "sst")

    # if monthly calculate the overlap indices in terms of months by multiplying by 12
    if monthly:
        # correct each month individually
        ovl_idx = (hadisst_ey - histo_sy) * 12      # start offset
        mon_correct = numpy.zeros([12, cmip5_ts.shape[1], cmip5_ts.shape[2]], 'f')
        for m in range(0, 12):
            cmip5_ens_mean_monmean = numpy.mean(cmip5_ts[ovl_idx-5+m:ovl_idx+m:12], axis=0)
            hadisst_monmean = numpy.mean(hadisst_sst[ovl_idx-5+m::12], axis=0)
            mon_correct[m] = hadisst_monmean - cmip5_ens_mean_monmean
        # tile and subtract from cmip5 timeseries
        n_repeats = cmip5_ts.shape[0] / 12
        cmip5_ens_mean_correction = numpy.tile(mon_correct, [n_repeats,1,1])
    else:
        ovl_idx = hadisst_ey - histo_sy    
        cmip5_ens_mean_timmean = numpy.mean(cmip5_ts[ovl_idx-5:ovl_idx], axis=0)
        hadisst_timmean = numpy.mean(hadisst_sst[ovl_idx-5:], axis=0)
        cmip5_ens_mean_correction = hadisst_timmean - cmip5_ens_mean_timmean
    return cmip5_ts + cmip5_ens_mean_correction
    
#############################################################################

def create_cmip5_long_term_mean_timeseries(run_type, ref_start, ref_end, monthly=True, run_n=400):
    # create the cmip5 timeseries of the cmip5 ensemble mean
    # This consists of the ens mean of the cmip5 anomalies from the reference
    # plus the HadISST reference
    
    # check which run type we should actually load.
    # Likely is rcp45 + adjustment
    if run_type == "likely":
        load_run_type = "rcp45"
    else:
        load_run_type = run_type
        
    # get the dates
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010

    # load the ensemble mean of the anomalies
    cmip5_ens_mean_anoms_fname = get_concat_anom_sst_ens_mean_smooth_fname(load_run_type, ref_start, ref_end, monthly=monthly)
    cmip5_ens_mean_anoms = load_sst_data(cmip5_ens_mean_anoms_fname, "sst")
        
    # if we have the likely scenario then fit the period between 2016 and 2035 to the likely
    # scenario from AR5 Ch11
    if run_type == "likely":
        cmip5_ens_mean_anoms = fit_mean_to_likely(cmip5_ens_mean_anoms, monthly)

    # load the HadISST reference pattern
    hadisst_ref_fname = get_HadISST_reference_fname(histo_sy, hadisst_ey, ref_start, ref_end, run_n)
    hadisst_ref_sst = load_sst_data(hadisst_ref_fname, "sst")

    # add it onto the ensemble mean anomalies
    cmip5_ens_mean = cmip5_ens_mean_anoms + hadisst_ref_sst

    return cmip5_ens_mean

#############################################################################

def create_cmip5_rcp_anomalies(run_type, ref_start, ref_end, eof_year, monthly=True):
    # create the time series of anomalies from the mean of the various 
    # samples in the CMIP5 ensemble
    # This spans the uncertainty of the GMT response to GHG forcing in CMIP5 

    if run_type == "likely":
        load_run_type = "rcp45"
    else:
        load_run_type = run_type

    # load the eof patterns in the eof_year
    eof_fname = get_cmip5_EOF_filename(load_run_type, ref_start, ref_end, eof_year, monthly=monthly)
    eofs = load_sst_data(eof_fname, "sst")
    
    # load the principle components for the eof_year
    syn_pc_fname  = get_syn_SST_PCs_filename(load_run_type, ref_start, ref_end, eof_year, monthly=monthly)
    syn_pc = load_data(syn_pc_fname, "sst")
    
    # load the timeseries of scalings and offsets to the pcs over the CMIP5 period
    proj_pc_scale_fname = get_cmip5_proj_PC_scale_filename(load_run_type, ref_start, ref_end, eof_year, monthly=monthly)
    proj_pc_scale  = load_data(proj_pc_scale_fname, "sst_scale")
    proj_pc_offset = load_data(proj_pc_scale_fname, "sst_offset")
    
    # corresponding weights that we supplied to the EOF function
    coslat = numpy.cos(numpy.deg2rad(numpy.arange(89.5, -90.5,-1.0))).clip(0., 1.)
    wgts = numpy.sqrt(coslat)[..., numpy.newaxis]

    # create the timeseries of reconstructed SSTs for just this sample
    # recreate the field - monthy by month if necessary
    if monthly:
        syn_sst_rcp = numpy.ma.zeros([proj_pc_scale.shape[0], eofs.shape[2], eofs.shape[3]], 'f')
        for m in range(0, 12):
            pc_ts = syn_pc[m,sample,:neofs] * proj_pc_scale[m::12,:neofs] + proj_pc_offset[m::12,:neofs]
            syn_sst_rcp[m::12] = reconstruct_field(pc_ts, eofs[m], neofs, wgts)
    else:
        pc_ts = syn_pc[0,sample,:neofs] * proj_pc_scale[:,:neofs] + proj_pc_offset[:,:neofs]
        syn_sst_rcp = reconstruct_field(pc_ts, eofs[0], neofs, wgts)
    return syn_sst_rcp

#############################################################################

def create_hadisst_monthly_reference(run_type, ref_start, ref_end, n_repeats, run_n=400):
    # create the annual cycle from hadisst, repeating it a number of times
    # so as to add it onto the CMIP5 timeseries
    
    # get the dates
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010

    # load in the monthly smoothed reference
    mon_smooth_name = get_HadISST_monthly_reference_fname(histo_sy, histo_ey, ref_start, ref_end, run_n)
    resid_ac = load_sst_data(mon_smooth_name, "sst")
    ac_tile = numpy.tile(resid_ac, [n_repeats,1,1])
    return ac_tile

#############################################################################

def create_hadisst_cmip5_long_term_timeseries(hadisst_ts, cmip5_ts, monthly=True):
    # create a long timeseries of the long term trend which has the hadisst
    # data for the first ~100 years and then the CMIP5 trends into the future
    
    # create the output - this is just the shape of the cmip5 timeseries as the
    # cmip5 timeseries also spans the historical timeperiod
    out_data = numpy.ma.zeros(cmip5_ts.shape, 'f')

    # get the dates
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010
    # get the monthly scalar for the indices
    if monthly:
        M = 12
    else:
        M = 1

    # assign the output data from hadisst_ey+1 to rcp_ey to the reconstructed sst
    out_data[(hadisst_ey-histo_sy)*M:] = cmip5_ts[(hadisst_ey-histo_sy)*M:]
    
    ovl_yr = 5 # overlap years
    # assign the output data from 0 to hadisst_ey-histo_sy-5 to the HadISST ssts
    out_data[:(hadisst_ey-histo_sy-ovl_yr)*M] = hadisst_ts[:(hadisst_ey-histo_sy-ovl_yr)*M]
    
    # now, over the 5 year period we want to interpolate between the HadISST
    # data and the data generated from the CMIP5 ensemble
    for y in range(0, ovl_yr):
        # calculate the weights
        w0 = float(ovl_yr-y)/ovl_yr      # HadISST2 weight
        w1 = float(y)/ovl_yr             # CMIP5 weight
        for mm in range(0, M):
            idx = (hadisst_ey-histo_sy-ovl_yr+y)*M + mm
            out_data[idx] = hadisst_ts[idx] * w0 + cmip5_ts[idx] * w1

    return out_data.astype(numpy.float32)

#############################################################################

def save_syn_SSTs(out_data, run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, monthly):
    if run_type == "likely":
        load_run_type = "rcp45"
    else:
        load_run_type = run_type

    # we require the length of the time series - get this from the proj_pc_scale_fname
    proj_pc_scale_fname = get_cmip5_proj_PC_scale_filename(load_run_type, ref_start, ref_end, eof_year, monthly=monthly)
    fh = netcdf_file(proj_pc_scale_fname, 'r')
    t_var = fh.variables["time"]
    
    # we require the shape of the field - get this from the cmip5 ens mean file
    ens_mean_fname = get_concat_anom_sst_ens_mean_smooth_fname(load_run_type, ref_start, ref_end, monthly=monthly)
    fh2 = netcdf_file(ens_mean_fname, 'r')
    lon_var = fh2.variables["longitude"]
    lat_var = fh2.variables["latitude"]
    attrs = fh2.variables["sst"]._attributes

    # mask any nans
    mv = attrs["_FillValue"]
    out_data = numpy.ma.fix_invalid(out_data, fill_value=mv)
    # fix the lsm
    for t in range(1, out_data.shape[0]):
        out_data.data[t][out_data.data[0] == mv] = mv
    
    # smooth the data
    lat_data = lat_var[:].byteswap().newbyteorder().astype(numpy.float32)

    out_data = zonal_smoother(out_data, lat_data, 64, 90, mv)

    # get the output name
    out_name = get_syn_sst_filename(run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, monthly)

    save_3d_file(out_name, out_data, lon_var, lat_var, attrs, t_var[:], t_var._attributes)
    fh.close()
    fh2.close()
    print out_name

#############################################################################

def create_syn_SSTs(run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, var_eofs, monthly):
    # determine which hadisst ensemble member to use
    hadisst_ens_members = [1059, 115, 1169, 1194, 1346, 137, 1466, 396, 400, 69]
    run_n = hadisst_ens_members[numpy.random.randint(0, len(hadisst_ens_members))]

    # get the hadisst trend
    hadisst_trend = create_hadisst_long_term_timeseries(monthly, run_n)
    # get the cmip5 trend
    cmip5_trend = create_cmip5_long_term_mean_timeseries(run_type, ref_start, ref_end, monthly, run_n)
    # create the sample from the distribution of the CMIP5 SST response to GHG forcing
    #syn_sst_rcp = create_cmip5_rcp_anomalies(run_type, ref_start, ref_end, eof_year, monthly)
    
    # cmip5 ssts are the sum of the ensemble mean trend and the deviation from the ensemble mean
    # monthly ssts have the hadisst annual cycle added onto them
    # create the hadisst annual cycle to add to the cmip5 projected SSTs, if monthly data is
    # required
    if monthly:
        n_repeats = syn_sst_rcp.shape[0] / 12       # number of repeats = number of years
        hadisst_ac = create_hadisst_monthly_reference(run_type, ref_start, ref_end, n_repeats, run_n)
        cmip5_sst = cmip5_trend + hadisst_ac
    else:
        cmip5_sst = cmip5_trend
    
    # adjust the cmip5 data
    cmip5_sst = correct_cmip5_long_term_mean_timeseries(cmip5_sst, monthly, run_n)
    # add the synthetic warming
    #cmip5_sst += syn_sst_rcp

    # create the interpolated / composite trend data
    out_data = create_hadisst_cmip5_long_term_timeseries(hadisst_trend, cmip5_sst, monthly)
    # create the synthetic internal variability
    #if monthly:
    #    intvar = create_monthly_intvar(run_type, ref_start, ref_end, n_pcs=var_eofs, run_n=run_n)
    #else:
    #    intvar = create_yearly_intvar(run_type, ref_start, ref_end, n_pcs=var_eofs, run_n=run_n)
    #out_data += intvar
    # save the synthetic ssts
    save_syn_SSTs(out_data, run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, monthly)

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""       # run type can be rcp45, rcp85, rcp26 or likely -
                        # fits ensemble mean to AR5 ch11 likely scenario for 2016->2035
    neofs = 0
    eof_year = 2050
    sample = 100
    intvarmode = 0      # internal variability mode - 0 = none, 1 = yearly, 2 = monthly
    monthly = False     # use the monthly EOFs / PCs ?
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:n:f:a:i:v:m',
                               ['run_type=', 'ref_start=', 'ref_end=', 'neofs=', 'eof_year=', 'sample=', 'intvarmode=',
                                'varneofs=', 'monthly'])

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
        if opt in ['--sample', '-a']:
            sample = int(val)
        if opt in ['--intvar', '-i']:
            intvarmode = int(val)
        if opt in ['--varneofs', '-v']:
            var_eofs = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True
    
    create_syn_SSTs(run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, var_eofs, monthly)
