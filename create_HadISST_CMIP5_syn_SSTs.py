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
from create_HadISST_sst_anoms import get_HadISST_smooth_fname, get_HadISST_reference_fname
from cmip5_functions import load_data, reconstruct_field, calc_GMSST, load_sst_data
from calc_HadISST_residual_EOFs import get_HadISST_residual_PCs_fname, get_HadISST_residual_EOFs_fname, get_HadISST_monthly_residual_EOFs_fname, get_HadISST_monthly_residual_PCs_fname, get_HadISST_annual_cycle_residuals_fname
import numpy
from netcdf_file import *
from ARN import ARN

#############################################################################

def get_output_directory(run_type, ref_start, ref_end, eof_year):
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    out_dir = "HadISST_"+str(histo_sy)+"_"+str(histo_ey)+"_"+\
              run_type+"_"+str(rcp_sy)+"_"+str(rcp_ey)+\
              "_r"+str(ref_start)+"_"+str(ref_end)+\
              "_y"+str(eof_year)
    if not os.path.exists("output/"+out_dir):
        os.mkdir("output/"+out_dir)
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
    if not os.path.exists("output/"+out_dir + "/" + intvarstr + "/"):
        os.mkdir("output/"+out_dir + "/" + intvarstr + "/")
    return "output/" + out_dir + "/" + intvarstr + "/" + out_name

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

def create_monthly_intvar(run_type, ref_start, ref_end, n_pcs=20, ac=True, run_n=400):
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
    # add the annual cycle back on - need to tile it to match the number of
    # reconstructed fields
    n = monthly_intvar.shape[0]
    if ac:
        ac_tile = numpy.tile(resid_ac, [n/12,1,1])
        monthly_intvar = monthly_intvar + ac_tile
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

def fit_mean_to_likely(ens_mean):
    # scale the ensemble mean so that the GMSST  matches the AR5 likely range
    # between 2016 and 2035.
    # first need the scaling factors to convert GMT to GMSST.  These are
    # calculated from a linear regression between the GMT and GMSST of the
    # CMIP5 models for the RCP4.5 and historical scenarios.

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
    ens_mean_2016 = ens_mean[2016-1899]
    ens_mean_2016 = numpy.reshape(ens_mean_2016, [1, ens_mean_2016.shape[0], ens_mean_2016.shape[1]])
    ens_mean_2035 = ens_mean[2035-1899]
    ens_mean_2035 = numpy.reshape(ens_mean_2035, [1, ens_mean_2035.shape[0], ens_mean_2035.shape[1]])
    gmsst_ratio_2016 = (likely_gmsst_mean_2016 / calc_GMSST(ens_mean_2016)).squeeze()
    gmsst_ratio_2035 = (likely_gmsst_mean_2035 / calc_GMSST(ens_mean_2035)).squeeze()
    # create a linear interpolation between 1.0 in 2005, the 2016 ratio and
    # the 2035 ratio - 2035 ratio will then be applied for rest of timeseries
    gmsst_scaling = numpy.ones([ens_mean.shape[0],1,1], 'f')
    # calculate the interpolated sections
    interp_section_2005_2016 = numpy.interp(numpy.arange(2005,2017), [2005,2016], [1.0, gmsst_ratio_2016])
    interp_section_2016_2035 = numpy.interp(numpy.arange(2017,2035), [2016,2035], [gmsst_ratio_2016, gmsst_ratio_2035])
    gmsst_scaling[2005-1899:2017-1899] = interp_section_2005_2016.reshape([interp_section_2005_2016.shape[0],1,1])
    gmsst_scaling[2017-1899:2035-1899] = interp_section_2016_2035.reshape([interp_section_2016_2035.shape[0],1,1])
    gmsst_scaling[2035-1899:] = gmsst_ratio_2035
    ens_mean = ens_mean * gmsst_scaling
    return ens_mean

#############################################################################

def create_syn_SSTs(run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, var_eofs, monthly):
    # we require:
    # 1. EOFs in the eof_year
    # 2. Synthetic PCs in the eof_year
    # 3. PC scalings over the required period
    # 4. Ensemble mean of the CMIP5 members over 1899->2100
    
    if run_type == "likely":
        load_run_type = "rcp45"
    else:
        load_run_type = run_type

    eof_fname = get_cmip5_EOF_filename(load_run_type, ref_start, ref_end, eof_year, monthly=monthly)
    eofs = load_sst_data(eof_fname, "sst")
    
    syn_pc_fname  = get_syn_SST_PCs_filename(load_run_type, ref_start, ref_end, eof_year, monthly=monthly)
    syn_pc = load_data(syn_pc_fname, "sst")
    
    proj_pc_scale_fname = get_cmip5_proj_PC_scale_filename(load_run_type, ref_start, ref_end, eof_year, monthly=monthly)
    proj_pc_scale  = load_data(proj_pc_scale_fname, "sst_scale")
    proj_pc_offset = load_data(proj_pc_scale_fname, "sst_offset")
    
    ens_mean_fname = get_concat_anom_sst_ens_mean_smooth_fname(load_run_type, ref_start, ref_end, monthly=monthly)
    ens_mean = load_sst_data(ens_mean_fname, "sst")
    
    # get the dates
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    hadisst_ey = 2010
    
    # we also require
    # 5. smoothed HadISST fields
    # 6. the HadISST 1986->2005 reference
    # Note: (20/03/2015) we have now calculated this for 10 different hadisst ensemble
    # members - (uniform) random sample from these ten
    
    hadisst_ens_members = [1059, 115, 1169, 1194, 1346, 137, 1466, 396, 400, 69]
    run_n = hadisst_ens_members[numpy.random.randint(0, len(hadisst_ens_members))]
    
    hadisst_smoothed_fname = get_HadISST_smooth_fname(histo_sy, hadisst_ey, run_n)
    hadisst_sst = load_sst_data(hadisst_smoothed_fname, "sst")
    hadisst_ref_fname = get_HadISST_reference_fname(histo_sy, hadisst_ey, ref_start, ref_end, run_n)
    hadisst_ref_sst = load_sst_data(hadisst_ref_fname, "sst")
    
    # corresponding weights that we supplied to the EOF function
    coslat = numpy.cos(numpy.deg2rad(numpy.arange(89.5, -90.5,-1.0))).clip(0., 1.)
    wgts = numpy.sqrt(coslat)[..., numpy.newaxis]

    # create the output
    out_data = numpy.ma.zeros(ens_mean.shape, 'f')
    
    # correct the ensemble mean of the CMIP5 ensemble, this is achieved by subtracting the
    # difference between the 5 year mean of HadISST and the 5 year mean of the CMIP5 ensemble 
    # mean for 2006->2010 (the overlap period) from the CMIP5 ensemble mean
    ovl_idx = hadisst_ey - histo_sy
    ens_mean_timmean = numpy.mean(ens_mean[ovl_idx-5:ovl_idx], axis=0) + hadisst_ref_sst
    hadisst_timmean = numpy.mean(hadisst_sst[ovl_idx-5:], axis=0)
    ens_mean_correction = hadisst_timmean - ens_mean_timmean
    ens_mean = ens_mean + ens_mean_correction

    # if we have the likely scenario then fit the period between 2016 and 2035 to the likely
    # scenario from AR5 Ch11
    if run_type == "likely":
        ens_mean = fit_mean_to_likely(ens_mean)

    # create the timeseries of reconstructed SSTs for just this sample
    # recreate the field - monthy by month if necessary
    if monthly:
        syn_sst_rcp = numpy.ma.zeros(ens_mean.shape, 'f')
        for m in range(0, 12):
            pc_ts = syn_pc[m,sample,:neofs] * proj_pc_scale[m::12,:neofs] + proj_pc_offset[m::12,:neofs]
            syn_sst_rcp[m::12] = reconstruct_field(pc_ts, eofs[m], neofs, wgts) + ens_mean[m::12] + hadisst_ref_sst
    else:
        pc_ts = syn_pc[0,sample,:neofs] * proj_pc_scale[0,:,:neofs] + proj_pc_offset[0,:,:neofs]
        syn_sst_rcp = reconstruct_field(pc_ts, eofs[0], neofs, wgts) + ens_mean + hadisst_ref_sst

    # assign the output data from hadisst_ey+1 to rcp_ey to the reconstructed sst
    out_data[hadisst_ey-histo_sy:] = syn_sst_rcp[hadisst_ey-histo_sy:]
    
    ovl_yr = 5 # overlap years
    if monthly:
        ovl_yr *= 12
    # assign the output data from 0 to hadisst_ey-histo_sy-20 to the HadISST ssts
    out_data[:hadisst_ey-histo_sy-ovl_yr] = hadisst_sst[:hadisst_ey-histo_sy-ovl_yr]
    
    # now, over the 20 year period we want to interpolate between the HadISST
    # data and the data generated from the CMIP5 ensemble
    for y in range(0, ovl_yr):
        # calculate the weights
        w0 = float(ovl_yr-y)/ovl_yr      # HadISST2 weight
        w1 = float(y)/ovl_yr      # CMIP5 weight
        idx = hadisst_ey-histo_sy-ovl_yr+y
        out_data[idx] = hadisst_sst[idx] * w0 + syn_sst_rcp[idx] * w1

    # create the internal variability / high frequency variability
    if intvarmode == 1:
        yearly_intvar = create_yearly_intvar(run_type, ref_start, ref_end, n_pcs=var_eofs, run_n=run_n)
        out_data = out_data + yearly_intvar
    elif intvarmode == 2:
        monthly_intvar = create_monthly_intvar(run_type, ref_start, ref_end, n_pcs=var_eofs, run_n=run_n)
        if monthly:     # no repetition of data needed if monthly variability used
            out_data += monthly_intvar
        else:
            out_data = numpy.repeat(out_data, 12, axis=0) + monthly_intvar

    # get the lon / lats / time / attributes from the projected sst scaling file
    # and the ensemble mean file
    fh = netcdf_file(proj_pc_scale_fname, 'r')
    t_var = fh.variables["time"]
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
    
    out_name = get_syn_sst_filename(run_type, ref_start, ref_end, neofs, eof_year, sample, intvarmode, monthly)
    
    # if this is the monthly internal variability then we need to produce a new
    # time dimensions - which involves writing the file out manually
    if intvarmode == 2 and not monthly:
        new_t_data = numpy.repeat(t_var[:], 12, axis=0)
        for t in range(0, new_t_data.shape[0]):
            new_t_data[t] = new_t_data[t] + (t % 12) * 30
        save_3d_file(out_name, out_data, lon_var, lat_var, attrs, new_t_data, t_var._attributes)
    else:
        save_3d_file(out_name, out_data, lon_var, lat_var, attrs, t_var[:], t_var._attributes)
    fh.close()
    fh2.close()
    print out_name

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