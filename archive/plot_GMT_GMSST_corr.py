#! /usr/bin/env python  
#############################################################################
#
# Program : plot_GMT_GMSST_corr.py
# Author  : Neil Massey
# Purpose : Plot a 2D histogram of the GMT vs GMSST in the CMIP5 runs
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
# Date    : 19/03/15
#
#############################################################################

import os, sys, getopt
from cmip5_functions import get_output_directory
from netcdf_file import *
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib as mpl
import scipy.stats

#############################################################################

def create_color_map(min_v, max_v):
    # this is the color scale from Neu et al 2013 (BAMS)
    d = 100
    nl = int(max_v - min_v + d) / d
    levels = [x*d for x in range(0, nl+1)]
    cmap = ["#ffffff", "#aaaaaa", "#666666", "#444444",
            "#002288", "#0044aa", "#0066cc", "#0088ff", 
            "#444400", "#888800", "#bbbb00", "#eeee00",
            "#440000", "#880000", "#aa1111", "#cc2222", "#ff3333"]
    ccmap, norm = col.from_levels_and_colors(levels, cmap, 'neither')
    return ccmap, norm, levels

#############################################################################

def plot_GMT_GMSST_corr(run_type, ref_start, ref_end):
    # get the netcdf file
    out_dir = get_output_directory(run_type, ref_start, ref_end)
    out_name = out_dir + "/" + out_dir.split("/")[1] + "_tos_tas_GM_ts.nc"

    fh = netcdf_file(out_name, 'r')
    tas = numpy.array(fh.variables["tas"][:])
    tos = numpy.array(fh.variables["tos"][:])
    
    # get the min / max values
    min_tas = numpy.min(tas[tas < 1000])
    max_tas = numpy.max(tas[tas < 1000])
    min_tos = numpy.min(tos[tos < 1000])
    max_tos = numpy.max(tos[tos < 1000])

    tas_range = [int(2*(min_tas - 0.5))/2, int(2*(max_tas + 0.5))/2]
    tos_range = [int(2*(min_tos - 0.5))/2, int(2*(max_tos + 0.5))/2]
    
    # create the storage
    d = 0.125
    freq_matrix = numpy.zeros([1.0/d*(tas_range[1]-tas_range[0])+1,
                               1.0/d*(tos_range[1]-tos_range[0])+1], 'f')
    # loop through to create the frequencies
    s = 0
    i = 0
    c = 0
    for e in range(0, tas.shape[0]):
        if tas[e,0] >= 1000 or tos[e,0] >= 1000:
            continue
        for t in range(0, tas.shape[1]):
            tas_v = tas[e,t]
            tos_v = tos[e,t]
            tas_i = int((tas_v - tas_range[0]) / d)
            tos_i = int((tos_v - tos_range[0]) / d)
            freq_matrix[tas_i, tos_i] += 1
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(tas[e], tos[e])
        s += slope
        i += intercept
        c += 1
    s /= c
    i /= c
    # plot the frequency
    cmap, norm, levels = create_color_map(numpy.min(freq_matrix), numpy.max(freq_matrix))
    X = numpy.arange(tas_range[0], tas_range[1]+d, d)
    Y = numpy.arange(tos_range[0], tos_range[1]+d, d)
    plt.pcolormesh(X, Y, freq_matrix.T, cmap=cmap, norm=norm)
    # calculate linear regression
    L = X*s + i
    plt.plot(X, L, 'k-', lw=2.0)
    
    plt.gca().set_xlabel("GMT anomaly at surface $^\circ$C")
    plt.gca().set_ylabel("GMSST anomaly $^\circ$C")
    title = "Slope: %.3f" %s
    title += "    Intercept: %.3f" %i
    plt.gca().set_title(title)
    plt.gca().set_xlim(numpy.min([tas_range[0], tos_range[0]]),
                       numpy.max([tas_range[1], tos_range[1]]))
    plt.gca().set_ylim(numpy.min([tas_range[0], tos_range[0]]),
                       numpy.max([tas_range[1], tos_range[1]]))

    fig = plt.gcf()
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='uniform',
                                   extend='neither')

    plt.savefig(out_name[:-3] + ".pdf", bbox_inches="tight")
    print out_name[:-3] + ".pdf"
    fh.close()

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
    
    plot_GMT_GMSST_corr(run_type, ref_start, ref_end)
