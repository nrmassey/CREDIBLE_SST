#############################################################################
#
# Program : cmip5_functions.py
# Author  : Neil Massey
# Purpose : Common functions to get information about the CMIP5 ensembele
#           stored on SoGE / OUCE computers
# Date    : 19/06/14
#
#############################################################################

import os, sys
sys.path.append("/soge-home/staff/coml0118/python") # path to python cdo routines
from cdo import *
from scipy.io.netcdf import *
import numpy

#############################################################################

def get_cmip5_path():
    path = "/soge-home/data_not_backed_up/model/cmip5/"
    return path

#############################################################################

def get_output_directory(run_type, ref_sy, ref_ey, year=None):
    uname = os.uname()
    if uname[0] == "Darwin":
        out_dir_base = "/Users/Neil/Coding/CREDIBLE_output/output/"
    elif uname[1] == "ouce-linux-01.ouce.ox.ac.uk" or \
         uname[1] == "ouce-linux-02.ouce.ox.ac.uk":
        out_dir_base = "/soge-home/staff/coml0118/CREDIBLE_SST/output/"
        
    out_dir = out_dir_base + run_type+"_"+str(ref_sy)+"_"+str(ref_ey)
    if not year is None:
        out_dir += "_y"+str(year)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir

#############################################################################

def get_cmip5_tos_fname(run_type, model, run):
    path = get_cmip5_path()
    path += model + "/" + run_type + "/mon/Omon/" + run
    files = os.listdir(path)
    for f in files:
        if "tos" in f:
            return path + "/" + f

#############################################################################

def get_cmip5_tas_fname(run_type, model, run):
    path = get_cmip5_path()
    path += model + "/" + run_type + "/mon/Amon/" + run
    files = os.listdir(path)
    for f in files:
        if "tas" in f:
            return path + "/" + f

#############################################################################

def get_cmip5_sic_fname(run_type, model, run):
    path = get_cmip5_path()
    path += model + "/" + run_type + "/mon/OImon/" + run
    if os.path.exists(path):
        files = os.listdir(path)
        for f in files:
            if "sic" in f:
                return path + "/" + f
    else:
        return ""

#############################################################################

def get_cmip5_ensemble_members(run_type, model, variable):
    # run_type can either be historical, rcp45 or rcp85
    # get a list of the top directory at get_cmip5_path
    path = get_cmip5_path()
    top_dir = os.listdir(path)
    # going to return a dictionary of models from the archive
    cmip5_dict = {}
    for td in top_dir:
        # check that it is a directory
        if os.path.isdir(path+td):
            # list the next directory - should have 3 directories with
            # historical, rcp45 and rcp85 in - i.e. the run type
            t2d = os.listdir(path+td)
            for t2ad in t2d:
                if run_type in t2ad:
                    # check whether the full path exists
                    full_path = path+td+"/"+run_type+"/mon/"+model+"/"
                    if os.path.isdir(full_path):
                        # again list the directory
                        t3d = os.listdir(full_path) # t3d is ensemble member
                        # loop though each file in the directory
                        for t4d in t3d:
                            t5d = os.listdir(full_path+t4d)
                            # check whether the variable is in one of the filenames
                            for t6d in t5d:
                                if variable in t6d:
                                    file_path = full_path + t4d + "/" + t6d
                                    # is this the first time added to dictionary
                                    if not td in cmip5_dict.keys():
                                        cmip5_dict[td] = [(t4d, file_path)]
                                    else:
                                        cmip5_dict[td].append((t4d, file_path))
    return cmip5_dict
   
#############################################################################

def load_data(fname, var="sst"):
    # load the data - we can use the same function for the PCs, EOFs and
    # ens_mean as they all have the same variable name (sst)
    nc_fh = netcdf_file(fname, "r")
    data = numpy.array(nc_fh.variables[var][:])
    nc_fh.close()
    return data

#############################################################################

def load_sst_data(fname, var):
    fh = netcdf_file(fname, 'r')
    var = fh.variables[var]
    # mask the missing values
    mv = var._attributes["_FillValue"]
    data = numpy.ma.masked_equal(var[:], mv)
    fh.close()
    return data

#############################################################################

def get_missing_value(fname, var):
    fh = netcdf_file(fname, 'r')
    var = fh.variables[var]
    # mask the missing values
    mv = var._attributes["_FillValue"]
    fh.close()
    return mv

#############################################################################

def get_lons_lats_attrs(in_fname):
    in_fh = netcdf_file(in_fname, "r")
    sst_wp_var = in_fh.variables["sst"]
    sst_warming_patts = sst_wp_var[:]
    lats_var = in_fh.variables["latitude"]
    lons_var = in_fh.variables["longitude"]

    # mask the array
    attrs = sst_wp_var._attributes
    return lons_var, lats_var, attrs

#############################################################################

def reconstruct_field(pcs, EOFs, neofs, wgts=None):
    # reconstruct the field from the principal components and the EOFs
    # subset the pcs and EOFs to the number of eofs required
    new_pcs = pcs[:,:neofs]
    new_EOFs = EOFs[:neofs,:,:]
    flat_EOFs = numpy.reshape(new_EOFs, [new_EOFs.shape[0], new_EOFs.shape[1]*new_EOFs.shape[2]])
    data = numpy.dot(new_pcs, flat_EOFs)
    out_data = numpy.reshape(data, [data.shape[0], new_EOFs.shape[1], new_EOFs.shape[2]])
    if wgts is not None:
       out_data = out_data / wgts
    out_data = numpy.ma.masked_invalid(out_data)
    return out_data

#############################################################################

def calc_GMSST(field):
    wts = numpy.cos(numpy.deg2rad(numpy.arange(89.5,-90.5,-1)))
    GMSST = numpy.ma.average(numpy.ma.mean(field, axis=2),axis=1,weights=wts)
    return GMSST

#############################################################################

def reconstruct_and_calc_GMSST(PCAs, eofs, ens_mean, n_eofs):
    # reconstruct the field and calculate the GMSST from the reconstructed field
    recon_field = reconstruct_field(PCAs, eofs, n_eofs)
    recon_field += ens_mean
    GMSST = calc_GMSST(recon_field)
    return GMSST

#############################################################################

def save_3d_file(out_fname, out_data, out_attrs, in_lats, in_lons):
    # open the file
    out_fh = netcdf_file(out_fname, "w")
    # create latitude and longitude dimensions - copy from the ens_mean file
    lon_data = numpy.array(in_lons[:])
    lat_data = numpy.array(in_lats[:])
    lon_out_dim = out_fh.createDimension("longitude", lon_data.shape[0])
    lat_out_dim = out_fh.createDimension("latitude", lat_data.shape[0])
    lon_out_var = out_fh.createVariable("longitude", lon_data.dtype, ("longitude",))
    lat_out_var = out_fh.createVariable("latitude", lat_data.dtype, ("latitude",))
    ens_out_dim = out_fh.createDimension("time", out_data.shape[0])
    ens_out_var = out_fh.createVariable("time", lon_data.dtype, ("time",))
    lon_out_var[:] = lon_data
    lat_out_var[:] = lat_data
    lon_out_var._attributes = in_lons._attributes
    lat_out_var._attributes = in_lats._attributes
    ens_out_var[:] = [float(x) for x in range(0, out_data.shape[0])]
    data_out_var = out_fh.createVariable("sst", numpy.float32, ("time", "latitude", "longitude"))
    data_out_var[:] = out_data[:]
    data_out_var._attributes = out_attrs
    out_fh.close()    
