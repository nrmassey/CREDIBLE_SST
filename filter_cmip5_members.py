#! /usr/bin/env python  
#############################################################################
#
# Program : filter_cmip5_members.py
# Author  : Neil Massey
# Purpose : build a list of cmip5 members that include an historic file that
#           covers the reference period and an rcp file that covers the years
#           2006->2100
#           This will create an index file which will indicate 	
# Inputs  : run_type  : rcp4.5 | rcp8.5
#           ref_start : year to start reference period, 1850->2005
#           ref_end   : year to end reference period, 1850->2005
# Output  : in the output/ directory filename is:
#            cmip5_members_index.txt
# Date    : 11/10/14
#
############################################################################# 

import os, sys, getopt

from cmip5_functions import get_cmip5_path, get_cmip5_ensemble_members, get_output_directory
sys.path.append("/soge-home/staff/coml0118/python") # path to python cdo routines
from cdo import *

#############################################################################

def filter_on_date(cmip5_dict, start_date, end_date):
    # filter the cmip5 ensemble on whether the required dates are contained
    # within the files in the dictionary
    cdo = Cdo()
    for model in cmip5_dict.keys():
        new_runs = []
        for run in cmip5_dict[model]:
            # check whether the date is in the file
            dates = cdo.showyear(input = run[1])[0]
            date_in = (str(start_date) in dates) and (str(end_date) in dates)
            # only add to new model dict if the date is in
            if date_in:
                new_runs.append(run)
        # reassign this model's entry in the dictionary
        if new_runs != []:
            cmip5_dict[model] = new_runs
        else:
            del cmip5_dict[model]
            
    return cmip5_dict

#############################################################################

def model_and_run_in_dict(cmip5_dict, model, run):
    # return whether the model and run occur in the cmip5 dictionary
    is_in = model in cmip5_dict.keys()    
    if is_in:
        for cmip5_run in cmip5_dict[model]:
            is_in = (run == cmip5_run[0])
            if is_in:
                break
            
    return is_in

#############################################################################

def get_cmip5_index_filename(run_type, ref_start, ref_end):
    fname = get_output_directory(run_type, ref_start, ref_end, None)+\
            "/cmip5_" + run_type + "_" + str(ref_start) + "_" +\
            str(ref_end) + "_index_mapping.txt"
    return fname

#############################################################################

def get_cmip5_model_mean_index_filename(run_type, ref_start, ref_end):
    fname = get_output_directory(run_type, ref_start, ref_end, None)+\
            "/cmip5_" + run_type + "_" + str(ref_start) + "_" +\
            str(ref_end) + "_mm_index_mapping.txt"
    return fname

#############################################################################

def build_cmip5_index_file(run_type, ref_start, ref_end):
    if (run_type == "rcp45" or run_type == "rcp85"):
        rcp_start = 2006
        rcp_end = 2100
    else:
        rcp_start = 1899
        rcp_end = 2005

    # get the historic runs first for the ocean (tos)
    hist_tos = get_cmip5_ensemble_members("historical", "Omon", "tos")
    hist_tos = filter_on_date(hist_tos, ref_start, ref_end)

    # get the rcp runs for the ocean (tos)
    rcp_tos  = get_cmip5_ensemble_members(run_type, "Omon", "tos")
    rcp_tos = filter_on_date(rcp_tos, rcp_start, rcp_end)
    
    # get the historic runs for the atmosphere (tas)
    hist_tas = get_cmip5_ensemble_members("historical", "Amon", "tas")
    hist_tas = filter_on_date(hist_tas, ref_start, ref_end)

    # get the rcp runs for the atmosphere (tas)
    rcp_tas = get_cmip5_ensemble_members(run_type, "Amon", "tas")
    rcp_tas = filter_on_date(rcp_tas, rcp_start, rcp_end)
    
    # create the output file
    fname = get_cmip5_index_filename(run_type, ref_start, ref_end)
    fh = open(fname, "w")
    
    # now we want to build a list of model and run names that
    # occur in all four lists
    c_idx = 0
    for model in hist_tos.keys():
        for run in hist_tos[model]:
            in_all =  model_and_run_in_dict(rcp_tas, model, run[0])
            in_all &= model_and_run_in_dict(hist_tas, model, run[0])
            in_all &= model_and_run_in_dict(rcp_tos, model, run[0])
        
            if in_all:
                fh.write(model)
                fh.write(", ")
                fh.write(run[0])
                fh.write(", ")
                fh.write(str(c_idx))
                fh.write("\n")
                c_idx += 1
    fh.close()

#############################################################################

def read_idx_file(fname):
    fh = open(fname, "r")
    fhl = fh.readlines()
    cmip5_idx = []
    
    for l in fhl:
        ll = l.strip().split(", ")
        cmip5_idx.append(ll)
        
    fh.close()
    return cmip5_idx

#############################################################################

def read_cmip5_index_file(run_type, ref_start, ref_end):
    fname = get_cmip5_index_filename(run_type, ref_start, ref_end)
    cmip5_idx = read_idx_file(fname)
    return cmip5_idx

#############################################################################

def read_cmip5_model_mean_index_file(run_type, ref_start, ref_end):
    fname = get_cmip5_model_mean_index_filename(run_type, ref_start, ref_end)
    cmip5_idx = read_idx_file(fname)
    return cmip5_idx

#############################################################################

def build_cmip5_model_mean_index_file(run_type, ref_start, ref_end):
    cmip5_idxs = read_cmip5_index_file(run_type, ref_start, ref_end)
    cmodel = cmip5_idxs[0][0]
    model_list = [cmodel]
    for idx in cmip5_idxs[1:]:
        if idx[0] != cmodel:
            model_list.append(idx[0])
            cmodel = idx[0]
    # output
    out_name = get_cmip5_model_mean_index_filename(run_type, ref_start, ref_end)
    fh = open(out_name, 'w')
    mn = 0
    for model in model_list:
        fh.write(model)
        fh.write(", ")
        fh.write("mm")
        fh.write(", ")
        fh.write(str(mn))
        fh.write("\n")
        mn += 1
    fh.close()

#############################################################################


if __name__ == "__main__":
    ref_start = -1
    ref_end = -1
    run_type = ""
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:',
                               ['run_type', 'ref_start=', 'ref_end='])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)

    build_cmip5_index_file(run_type, ref_start, ref_end)
    build_cmip5_model_mean_index_file(run_type, ref_start, ref_end)
