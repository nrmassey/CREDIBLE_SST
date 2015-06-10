#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib as mpl
from cmip5_functions import load_data, get_output_directory
from netcdf_file import *
import numpy
import os, getopt, sys

import pyximport
pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
from running_gradient_filter import *

#############################################################################

def plot_tas_ar5(run_type, ref_start, ref_end):
    sp = plt.subplot(111)
    Y0 = 2009.0
    Y1 = 2025.5
    Y2 = 2035
    C = 0.16

    grad0 = (0.3-C)/(Y1 - Y0)
    grad1 = (0.7-C)/(Y1 - Y0)

    ym0 = grad0*(2016-Y0)+C - 0.1
    yx0 = grad1*(2016-Y0)+C + 0.1
    ym1 = grad0*(2035-Y0)+C - 0.1
    yx1 = grad1*(2035-Y0)+C + 0.1
    
    sp.plot([2016,2035,2035,2016,2016],[ym0,ym1,yx1,yx0,ym0], 'k', lw=2.0, zorder=3)
    t_var = numpy.arange(1899,2100+1)
    print t_var.shape

    for rcp in ["rcp26", "rcp45", "rcp85"]:
        if rcp == "rcp26":
            col = '#888888'
        if rcp == "rcp45":
            col = 'r'
        if rcp == "rcp85":
            col = 'c'
        out_dir = get_output_directory(rcp, ref_start, ref_end)
        out_name = out_dir + "/" + out_dir.split("/")[1] + "_tos_tas_GM_ts.nc"

        fh = netcdf_file(out_name, 'r')
        tas = numpy.array(fh.variables["tas"][:])
        smoothed_tas = numpy.zeros(tas.shape, 'f')
        
        n_ens = tas.shape[0]
        X = numpy.arange(1899,2101)
        c = 0
        for e in range(0, n_ens):
            if tas[e,0] < 1000:
                tas_e = tas[e].byteswap().newbyteorder()
                TAS = running_gradient_3D(tas_e.reshape(tas_e.shape[0],1,1), 10)
                smoothed_tas[c] = TAS.flatten()
                c += 1
        tas_min = numpy.min(smoothed_tas[:c], axis=0)
        tas_max = numpy.max(smoothed_tas[:c], axis=0)
        tas_5 = numpy.percentile(smoothed_tas[:c], 5, axis=0)
        tas_95 = numpy.percentile(smoothed_tas[:c], 95, axis=0)
        tas_50 = numpy.percentile(smoothed_tas[:c], 50, axis=0)
        sp.plot(t_var, tas_min, '-', c=col, lw=2.0, zorder=0, alpha=0.5)
        sp.plot(t_var, tas_max, '-', c=col, lw=2.0, zorder=0, alpha=0.5)
        sp.plot(t_var, tas_50, '-', c=col, lw=2.0, zorder=2, alpha=0.5)
        sp.fill_between(t_var, tas_5, tas_95, edgecolor=col, facecolor=col, zorder=1, alpha=0.5)

    plt.gca().set_ylim([-0.5,2.5])
    plt.gca().set_xlim([1986,2050])
    f = plt.gcf()
    f.set_size_inches(10.5, 5.0)
    plt.savefig("ar5_ch11_fig25.pdf")
    
#############################################################################

if __name__ == "__main__":
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 14}
    mpl.rc('font', **font)

    ref_start = -1
    ref_end = -1
    run_type = ""
    opts, args = getopt.getopt(sys.argv[1:], 'r:s:e:',
                               ['run_type=', 'ref_start=', 'ref_end='])

    for opt, val in opts:
        if opt in ['--run_type', '-r']:
            run_type = val
        if opt in ['--ref_start', '-s']:
            ref_start = int(val)
        if opt in ['--ref_end', '-e']:
            ref_end = int(val)
    
    plot_tas_ar5(run_type, ref_start, ref_end)