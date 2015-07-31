#! /usr/bin/env python  
#############################################################################
#
# Program : create_CMIP5_syn_PCs.py
# Author  : Neil Massey
# Purpose : Create synthetic principle components for a particular eof_year
# Inputs  : run_type  : rcp4.5 | rc8.5 | histo
#           ref_start : year to start reference period, 1850->2005
#           ref_end   : year to end reference period, 1850->2005
#           eof_year  : year to take warming at
# Notes   : all reference values are calculated from the historical run_type
#           CMIP5 ensemble members are only included if their historical run 
#           includes the reference period
#           requires Andrew Dawsons eofs python libraries:
#            http://ajdawson.github.io/eofs/
# Output  : 
# Date    : 17/02/15
#
#############################################################################

import os, sys, getopt

from calc_CMIP5_EOFs import *
from create_CMIP5_sst_anoms import get_start_end_periods, get_concat_anom_sst_ens_mean_smooth_fname, get_start_end_periods
from cmip5_functions import calc_GMSST, load_data, reconstruct_field, load_sst_data

import numpy
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri

from skewnorm import *
import matplotlib.pyplot as plt

#############################################################################

def get_syn_SST_PCs_filename(run_type, ref_start, ref_end, eof_year, monthly):
    out_dir = get_output_directory(run_type, ref_start, ref_end, eof_year) 
    fname = out_dir+"/cmip5_EOF_" + get_file_suffix(run_type, ref_start, ref_end, eof_year) + "_synth_pc"
    if monthly:
        fname += "_mon"
    fname += ".nc"
    return fname
    
#############################################################################

def fit_marginal(pc_data, eof):
    # get the current eof from the data
    marginal_data = pc_data[:,eof].squeeze()
    # remove outliers > ns sd
    ns = 5
    mn = numpy.mean(marginal_data)
    std = numpy.std(marginal_data)    
    marginal_data[(marginal_data < mn - ns*std) | (marginal_data > mn + ns*std)] = mn
    
    x_data = numpy.arange(0, marginal_data.shape[0])
    # import sn - for skew normal distribution and MASS - for fitdistr
    sn = importr("sn")
    MASS = importr("MASS")
    # calculate first guesses - the mean of the absolute values and the standard deviation
    xi_guess = numpy.mean(numpy.abs(marginal_data))
    std_guess = numpy.std(marginal_data)
    # build the list of the starting values
    start_list = robjects.r("list(xi="+str(xi_guess)+",omega="+str(std_guess)+",alpha=0.0)")
    # fit the distribution
    fit_sn = MASS.fitdistr(marginal_data, sn.dsn, start_list, method="SANN")
    return fit_sn[0]

#############################################################################

def fit_mvdc(pc_data, n_eofs):
    # fit a copula with skew-normal marginals, and a Gumbel copula
    # import skew-normal and copula modules
    sn = importr("sn")
    cop = importr("copula")

    # create the parameter and marginal distribution list (of strings which are r objects)
    param_list_string = "list("
    margin_list_string = "c("
    start_list_string = "c("
    for n in range(0, n_eofs):
        m1 = fit_marginal(pc_data, n)
        param_list_string  += "list(xi="+str(m1[0])+",omega="+str(m1[1])+",alpha="+str(m1[2])+"), "
        margin_list_string += "\"sn\", "
        start_list_string  += str(m1[0]) + "," + str(m1[1]) + "," + str(m1[2]) + ","
    
    cop_param = 0.0
    
    param_list_string = param_list_string[:-2]+")"
    margin_list_string = margin_list_string[:-2]+")"
    start_list_string += str(cop_param) + ")"
    param_list = robjects.r(param_list_string)
    margin_list = robjects.r(margin_list_string)
    start_list = robjects.r(start_list_string)
    control = robjects.r("list(trace = FALSE, maxit = 10000)")
    
    # create the multivariate distribution with a gumbel copula
    pc_mvd = cop.mvdc(copula=cop.ellipCopula(family="normal", param=cop_param, dim=n_eofs), 
                      margins=margin_list,
                      paramMargins=param_list)
                     
    # now fit the data to get the copula
    marginal_data = pc_data[:,0:n_eofs]
    fit = cop.fitMvdc(marginal_data, pc_mvd, start=start_list, optim_control=control,
                      method = "Nelder")
    return pc_mvd

#############################################################################

def generate_synthetic_pcs(mvdc, n_samples):
    # mvdc is the multivariate distribution fitted by a copula using the
    # fit_mvdc routine above
    cop = importr("copula")
    syn_pcas = cop.rMvdc(n_samples, mvdc)
    return numpy.array(syn_pcas)

#############################################################################

def generate_large_sample_of_SSTs(pc_mvdc, eof_data, ens_mean, neofs):
    # now generate a number of samples and calculate just the mean sst from it
    n_big_sample = 10000
    sst_means_and_PCs = numpy.zeros([n_big_sample, neofs+1], 'f')
    # generate a load of PCs
    syn_PCs = generate_synthetic_pcs(pc_mvdc, n_big_sample)
    # reconstruct the field
    for spc in range(0, n_big_sample):
        this_syn_PCs = syn_PCs[spc] # just one set of PCs
        this_syn_PCs = this_syn_PCs.reshape([1, this_syn_PCs.shape[0]])
        recon_SST = reconstruct_field(this_syn_PCs, eof_data, neofs)
        recon_SST_gmsst = calc_GMSST(recon_SST+ens_mean)
        sst_means_and_PCs[spc,0] = recon_SST_gmsst
        sst_means_and_PCs[spc,1:] = this_syn_PCs.squeeze()
        
    return sst_means_and_PCs

#############################################################################

def sample_SSTs(sst_means_and_PCs, neofs, nsamps):
    # now we have a large distribution of SST warmings, choose a number (=nsamps)
    # so that the distribution is equally sampled
    sst_means_and_PCs = sst_means_and_PCs[sst_means_and_PCs[:,0].argsort()]
    # create the selected PCAs storage
    select_PCs = numpy.zeros([nsamps, neofs],'f')
    # derive the skew normal pdf from these sst means
    mu,std,alpha = skewnorm.fit(sst_means_and_PCs[:,0].squeeze())
    snpdf = skewnorm(mu,std,alpha)
    # sample strategy zero ensures that values are taken evenly across the
    # percentiles of the skew normal distribution
    # get the percentile values
    pts = numpy.arange(1.0/(nsamps+1),1.0,1.0/(nsamps+1))
    pt_vals = snpdf.ppf(pts)

    # pick one member from a number which could satisfy being at a particular
    # percentile value
    for ptp in range(0, pt_vals.shape[0]):
        # get where the GMSSTs bookend the percentile values
        I1 = numpy.where((sst_means_and_PCs[:,0] >= pt_vals[ptp]))
        if I1[0].shape[0] == 0:
            I1 = numpy.where((sst_means_and_PCs[:,0] <= pt_vals[ptp]))[0][-1]
        else:
            I1 = I1[0][0]
        # assign these PCAs to the selected PCAs
        select_PCs[ptp,:] = sst_means_and_PCs[I1][1:]
        
    return select_PCs
    
#############################################################################

def create_syn_SST_PCs(run_type, ref_start, ref_end, eof_year, neofs, nsamps, model_mean=False, monthly=False):
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
    # create the return storage
    select_PCs = numpy.zeros([pcs.shape[0], nsamps, neofs], 'f')
    # now loop through each month pcs - if yearly mean then there will only be one
    for m in range(0, pcs.shape[0]):
        # fit a copula to the principle components
        pc_mvdc = fit_mvdc(pcs[m], neofs)

        # generate a large sample of GMSSTs and their corresponding PCs    
        sst_means_and_PCs = generate_large_sample_of_SSTs(pc_mvdc, eofs[m], ens_mean, neofs)
    
        # now sample the distribution to get nsamps number of PCs which
        # represent the distribution of GMSSTs
        select_PCs[m] = sample_SSTs(sst_means_and_PCs, neofs, nsamps)
    
    # save
    out_fname = get_syn_SST_PCs_filename(run_type, ref_start, ref_end, eof_year, monthly)
    # fix the missing value meta data
    out_attrs = {"missing_value" : 2e20}
    # save the selected PCAs
    save_pcs(out_fname, select_PCs, out_attrs)
    print out_fname

#############################################################################

if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    neofs = 0
    eof_year = 2050
    nsamps = 100
    monthly = False
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:n:f:a:m',
                               ['run_type=', 'ref_start=', 'ref_end=', 'neofs=', 
                                'eof_year=', 'samples=', 'monthly'])

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
        if opt in ['--samples', '-a']:
            nsamps = int(val)
        if opt in ['--monthly', '-m']:
            monthly = True
    
    model_mean = False
    create_syn_SST_PCs(run_type, ref_start, ref_end, eof_year, neofs, nsamps, model_mean, monthly)