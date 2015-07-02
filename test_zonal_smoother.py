#! /usr/bin/env python

#############################################################################
#
# Program : test_zonal_smoother.py
# Author  : Neil Massey
# Purpose : Test the zonal smoother with some netcdf data
# Date    : 25/06/15
#
#############################################################################

import numpy
import pyximport
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
from zonal_smoother import zonal_smoother
from netcdf_file import *
from cmip5_functions import save_3d_file

#############################################################################

if __name__ == "__main__":
    nc_path = "/Users/Neil/Coding/CREDIBLE_output/output/HadISST_1899_2005_rcp45_2006_2100_r1986_2005_y2050/varmon/"
    nc_file = "HadISST_1899_2005_rcp45_2006_2100_r1986_2005_y2050_f2050_n6_a50_varmon_ssts_mon.nc"
    nc_fh = netcdf_file(nc_path+nc_file, "r")
    sst_var = nc_fh.variables["sst"]
    mv = sst_var._attributes["_FillValue"]
    sst_data = sst_var[:].byteswap().newbyteorder()
    lat_var = nc_fh.variables["latitude"]
    lat_data = lat_var[:].byteswap().newbyteorder()
    lon_var = nc_fh.variables["longitude"]
    lon_data = lon_var[:]
    
    power = 64
    max_ww = 90
    smooth_data = zonal_smoother(sst_data, lat_data, power, max_ww, mv)
    out_file = nc_path + nc_file[:-3] + "_smooth.nc"
    save_3d_file(out_file, smooth_data, sst_var._attributes, lat_var, lon_var)