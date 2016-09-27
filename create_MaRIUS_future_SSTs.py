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
from create_CMIP5_sst_anoms import get_concat_anom_sst_ens_mean_smooth_fname, get_start_end_periods
from cmip5_functions import load_data, load_sst_data, get_output_directory
from create_HadISST_CMIP5_syn_SSTs import *
from calc_CMIP5_EOFs import get_file_suffix, get_cmip5_PC_filename, save_pcs
from create_HadISST_sst_anoms import get_HadISST_monthly_residuals_fname
from create_CMIP5_syn_PCs import fit_mvdc, fit_marginal, generate_synthetic_pcs

import numpy
#from netcdf_file import *
from scipy.io.netcdf import *
from ARN import ARN
import pyximport
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
from zonal_smoother import *

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri

from skewnorm import *

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

def get_Ma_output_name(run_type, ref_start, ref_end, sy, ey, ptile, pattern_number):
    out_name = "MaSST_"+run_type+"_"+str(sy)+"_"+str(ey)+\
              "_r"+str(ref_start)+"_"+str(ref_end)+"_p"+str(ptile)+"_n"+str(pattern_number)+".nc"
    return out_name

#############################################################################

def save_Ma_syn_SSTs(out_data, run_type, ref_start, ref_end, sy, ey, ptile, pn):
    # we require the time data and shape of the field - get this from the cmip5 ens mean file
    ens_mean_fname = get_concat_anom_sst_ens_mean_smooth_fname(run_type, ref_start, ref_end, True)
    fh2 = netcdf_file(ens_mean_fname, 'r', mmap=False)
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

    out_data = out_data.byteswap().newbyteorder().astype(numpy.float32)

    # get the output name
    out_dir = get_Ma_output_directory(run_type, ref_start, ref_end, sy, ey-1)
    out_name = out_dir + "/" + get_Ma_output_name(run_type, ref_start, ref_end, sy, ey-1, ptile, pn)

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

def get_Ma_syn_SST_PCs_filename(run_type, ref_start, ref_end, eof_year, ptile, monthly):
    out_dir = get_output_directory(run_type, ref_start, ref_end, eof_year) 
    fname = out_dir+"/MaSST_EOF_" + get_file_suffix(run_type, ref_start, ref_end, eof_year) + "_p"+str(ptile)+"_synth_pc"
    if monthly:
        fname += "_mon"
    fname += ".nc"
    return fname

#############################################################################

def calc_NA_index(SST_field):
    # calculate the North Atlantic SST field index, as described in 
    # Schaller et al. 2016
    # SST_field has shape 360x180
    # coordinates are: -40->0 lon, 70->50 lat for 1st sector, 50->30 for 2nd sector
    # indices are: 320->360 lon, 20->40 lat 1st sector, 40->60 for 2nd sector
    wts = numpy.cos(numpy.deg2rad(numpy.arange(89.5,-90.5,-1)))
    s_lon = 320
    e_lon = 360
    s1_lat = 20
    e1_lat = 40
    s2_lat = 30
    e2_lat = 60
    S1_field = SST_field[0,s1_lat:e1_lat,s_lon:e_lon,]
    S2_field = SST_field[0,s2_lat:e2_lat,s_lon:e_lon,]
    
    S1_avg = numpy.ma.average(numpy.ma.mean(S1_field, axis=1),axis=0,weights=wts[s1_lat:e1_lat])
    S2_avg = numpy.ma.average(numpy.ma.mean(S2_field, axis=1),axis=0,weights=wts[s2_lat:e2_lat])

    NA_idx = S2_avg - S1_avg
    return NA_idx

#############################################################################

def generate_Ma_large_sample_of_SSTs(pc_mvdc, eof_data, ens_mean, neofs):
    # now generate a number of samples and calculate just the mean sst from it
    n_big_sample = 10000
    sst_means_and_PCs = numpy.zeros([n_big_sample, neofs+2], 'f')
    # generate a load of PCs
    syn_PCs = generate_synthetic_pcs(pc_mvdc, n_big_sample)
    # reconstruct the field
    for spc in range(0, n_big_sample):
        this_syn_PCs = syn_PCs[spc] # just one set of PCs
        this_syn_PCs = this_syn_PCs.reshape([1, this_syn_PCs.shape[0]])
        recon_SST = reconstruct_field(this_syn_PCs, eof_data, neofs)
        recon_SST_gmsst = calc_GMSST(recon_SST+ens_mean)
        recon_NA_idx = calc_NA_index(recon_SST+ens_mean)
        sst_means_and_PCs[spc,0] = recon_SST_gmsst
        sst_means_and_PCs[spc,1] = recon_NA_idx
        sst_means_and_PCs[spc,2:] = this_syn_PCs.squeeze()
        
    return sst_means_and_PCs

#############################################################################

def sample_Ma_SSTs(sst_means_and_PCs, neofs, nsamps, ptile):
    # now we have a large distribution of SST warmings, choose a number (=nsamps)
    # so that the distribution is equally sampled
    sst_means_and_PCs = sst_means_and_PCs[sst_means_and_PCs[:,0].argsort()]
    # create the selected PCAs storage
    select_PCs = numpy.zeros([nsamps, neofs+2],'f')
    # derive the skew normal pdf from these sst means
    mu,std,alpha = skewnorm.fit(sst_means_and_PCs[:,0].squeeze())
    snpdf = skewnorm(mu,std,alpha)

    # new sampling strategy for MaRIUS - just sample at one percentile
    pts = numpy.zeros([nsamps], 'f')
    pts_per_pc = int(nsamps)
    pt_val = snpdf.ppf(ptile)

    # pick nsamps / len(ptiles) members from those which could satisfy being at a particular
    # percentile value
    # get where the GMSSTs bookend the percentile value
    I1 = numpy.where((sst_means_and_PCs[:,0] >= pt_val))
    if I1[0].shape[0] == 0:
        I1 = numpy.where((sst_means_and_PCs[:,0] <= pt_val))[0][-1]
    else:
        I1 = I1[0][0]
    # assign these PCAs to the selected PCAs
    for i in range(0,nsamps):
        if I1+i < sst_means_and_PCs.shape[0]:
            select_PCs[i,:] = sst_means_and_PCs[I1+i]
    return select_PCs
    
#############################################################################

def create_Ma_syn_SST_PCs(run_type, ref_start, ref_end, eof_year, neofs, ptile, model_mean=False, monthly=False):
    # load the PCs, EOFs for this year
    pcs_fname = get_cmip5_PC_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    pcs = load_data(pcs_fname, "sst")
    eof_fname = get_cmip5_EOF_filename(run_type, ref_start, ref_end, eof_year, model_mean, monthly)
    eofs = load_data(eof_fname, "sst")
    
    # load the smoothed ensemble mean
    ens_mean_fname = get_concat_anom_sst_ens_mean_smooth_fname(run_type, ref_start, ref_end, monthly)
    ens_mean = load_sst_data(ens_mean_fname, "sst")
    # we only need one ensemble mean - calculate decadal mean
    histo_sy, histo_ey, rcp_sy, rcp_ey = get_start_end_periods()
    ens_mean = ens_mean[eof_year-histo_sy]

    # transform pc data to R compatible format
    pcs = pcs.byteswap().newbyteorder()
    nsamps = 100
    nmons = pcs.shape[0]
    # create the return storage
    select_PCs = numpy.zeros([pcs.shape[0], nsamps, neofs+2], 'f')

    # now loop through each month pcs - if yearly mean then there will only be one
    for m in range(0, nmons):
         # fit a copula to the principle components
         pc_mvdc = fit_mvdc(pcs[m], neofs)
 
         # generate a large sample of GMSSTs and their corresponding PCs    
         sst_means_and_PCs = generate_Ma_large_sample_of_SSTs(pc_mvdc, eofs[m], ens_mean, neofs)
     
         # now sample the distribution to get nsamps number of PCs which
         # represent the distribution of GMSSTs
         select_PCs[m] = sample_Ma_SSTs(sst_means_and_PCs, neofs, nsamps, ptile)
    
    # sort the pcs based on the first pc for each of the percentiles
    sorted_select_PCs = numpy.zeros([nmons, 2, neofs], 'f')
    for m in range(0, nmons):
        # get the NA indices for this month
        na_idxs = select_PCs[m,:,1]
        # sort it and get the indices
        na_idxs_sort = numpy.argsort(na_idxs)
        # get the first and last in the list sorted by NA indices
        # - i.e. where the North Atlantic index is the most different
        # we just want the PCs now
        for e in range(0, neofs):
            sorted_select_PCs[m,0,e] = select_PCs[m,:,2+e][na_idxs_sort[0]]
            sorted_select_PCs[m,1,e] = select_PCs[m,:,2+e][na_idxs_sort[-1]]

    # we now have two sets of PCs - one at each end of the distribution of NA SST gradient for the desired percentile
    # save
    out_fname = get_Ma_syn_SST_PCs_filename(run_type, ref_start, ref_end, eof_year, ptile, monthly)
    # fix the missing value meta data
    out_attrs = {"missing_value" : 2e20}
    # save the selected PCAs
    save_pcs(out_fname, sorted_select_PCs, out_attrs)
    print out_fname

#############################################################################

def create_Ma_syn_SSTs(run_type, ref_start, ref_end, sy, ey, eof_year, neofs, ptile, monthly):

    # determine which hadisst ensemble member to use
    hadisst_ens_members = [1059, 115, 1169, 1194, 1346, 137, 1466, 396, 400, 69]
    run_n = hadisst_ens_members[numpy.random.randint(0, len(hadisst_ens_members))]

    # load the CMIP5 ensemble mean timeseries
    # load the ensemble mean of the anomalies
    cmip5_ens_mean_anoms_fname = get_concat_anom_sst_ens_mean_smooth_fname(run_type, ref_start, ref_end, monthly)
    cmip5_ens_mean_anoms = load_sst_data(cmip5_ens_mean_anoms_fname, "sst")

    # load the eof patterns in the eof_year
    eof_fname = get_cmip5_EOF_filename(run_type, ref_start, ref_end, eof_year, monthly=True)
    eofs = load_sst_data(eof_fname, "sst")
    
    # load the principle components for the eof_year
    syn_pc_fname  = get_Ma_syn_SST_PCs_filename(run_type, ref_start, ref_end, eof_year, ptile, monthly=True)
    syn_pc = load_data(syn_pc_fname, "sst")
    
    # load the timeseries of scalings and offsets to the pcs over the CMIP5 period
    proj_pc_scale_fname = get_cmip5_proj_PC_scale_filename(run_type, ref_start, ref_end, eof_year, monthly=True)
    proj_pc_scale  = load_data(proj_pc_scale_fname, "sst_scale")
    proj_pc_offset = load_data(proj_pc_scale_fname, "sst_offset")
    
    # corresponding weights that we supplied to the EOF function
    coslat = numpy.cos(numpy.deg2rad(numpy.arange(89.5, -90.5,-1.0))).clip(0., 1.)
    wgts = numpy.sqrt(coslat)[..., numpy.newaxis]

    # create the timeseries of reconstructed SSTs for just this sample
    # recreate the field - monthy by month
    # pattern number
    pn = 0

    nmons=12
    # sub set the mean anomalies and the proj_pc_scale and offset
    cmip5_sy = 1899
    si = (sy-cmip5_sy)*12
    ei = (ey-cmip5_sy)*12
    cmip5_ens_mean_anoms = cmip5_ens_mean_anoms[si:ei]

    if ey == 2101:
        # create 2101
        S = cmip5_ens_mean_anoms.shape
        cmip5_ens_mean_anoms2 = numpy.zeros([S[0]+12, S[1], S[2]], 'f')
        cmip5_ens_mean_anoms2[:S[0]] = cmip5_ens_mean_anoms
        cmip5_ens_mean_anoms2[-12:] = cmip5_ens_mean_anoms[-12:]
        cmip5_ens_mean_anoms = cmip5_ens_mean_anoms2

    proj_pc_scale = proj_pc_scale[si-12:ei]
    proj_pc_offset = proj_pc_offset[si-12:ei]
    syn_sst_rcp = numpy.ma.zeros([proj_pc_scale.shape[0], eofs.shape[2], eofs.shape[3]], 'f')
    #
    for pn in range(0, 2):  # two patterns per percentile
        for m in range(0, nmons):
            pc_ts = syn_pc[m,pn,:neofs] * proj_pc_scale[m::12,:neofs] + proj_pc_offset[m::12,:neofs]
            syn_sst_rcp[m::12] = reconstruct_field(pc_ts, eofs[m], neofs, wgts)

        # load the hadisst reference
        n_repeats = cmip5_ens_mean_anoms.shape[0] / 12       # number of repeats = number of years
        hadisst_ac = create_hadisst_monthly_reference(run_type, ref_start, ref_end, n_repeats, run_n)
        # load the internal variability - we are only interested in the 30 year observed ones
        resid_fname = get_HadISST_monthly_residuals_fname(1899, 2010, 400)
        intvar = load_data(resid_fname, "sst")
        intvar = intvar[(1973-1899)*12:(2007-1899)*12]
        print "cmip5_ens_mean_anoms ", cmip5_ens_mean_anoms.shape
        print "syn_sst_rcp ", syn_sst_rcp.shape
        print "hadisst_ac ", hadisst_ac.shape
        print "intvar ", intvar.shape
        out_data = cmip5_ens_mean_anoms + syn_sst_rcp + hadisst_ac + intvar
        # save the synthetic ssts
        save_Ma_syn_SSTs(out_data, run_type, ref_start, ref_end, sy, ey, ptile, pn)

#############################################################################

if __name__ == "__main__":
    # Note - this is all an adaptation of CREDIBLE SST functions
    ref_start = -1
    ref_end = -1
    run_type = "rcp85"
    intvarmode = 0      # internal variability mode - 0 = none, 1 = yearly, 2 = monthly
    monthly = True      # use the monthly EOFs / PCs ?
    eof_year = 2050
    neofs = 6
    opts, args = getopt.getopt(sys.argv[1:], 's:e:i:y:z:p:',
                               ['ref_start=', 'ref_end=', 
                                'start_year=', 'end_year=',
                                'ptile='])

    for opt, val in opts:
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
        if opt in ['--start_year', '-y']:
            sy = int(val)
        if opt in ['--end_year', '-z']:
            ey = int(val)
        if opt in ['--ptile', '-p']:
            ptile = float(val)

    fname = get_Ma_syn_SST_PCs_filename(run_type, ref_start, ref_end, eof_year, ptile, monthly)
    if not os.path.exists(fname):
        create_Ma_syn_SST_PCs(run_type, ref_start, ref_end, eof_year, neofs, ptile, model_mean=False, monthly=monthly)
    create_Ma_syn_SSTs(run_type, ref_start, ref_end, sy, ey, eof_year, neofs, ptile, monthly)
