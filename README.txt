# Order to run programs to generate SST scenarios

1.  ./filter_cmip5_members.py       -r <run_type> -s <ref_start> -e <ref_end>
	# This creates a list of CMIP5 ensemble members that have an historic run covering the
	# reference period and a rcp run (run_type=rcp45 or run_type=rcp85) covering the period
	# 2006->2100 for both tos and tas variables.

2.  ./calc_warming_patterns.py      -r <run_type> -s <ref_start> -e <ref_end> -y <year> -v <var>
	# This creates a warming pattern (for either var="tos" or var="tas") which is the decadal
	# mean centred around year (start=year-4, end=year+5) with the mean of the reference
	# period from the corresponding historical run removed.  This is done for each CMIP5
	# ensemble member identified in 1.

3.  ./calc_SST_warming_EOFs.py      -r <run_type> -s <ref_start> -e <ref_end> -y <year>
	# The EOFs, PCs and ensemble mean of the warming patterns for the variable tos are
	# produced by this program.

4.  ./calc_SST_projected_PCs.py     -r <run_type> -s <ref_start> -e <ref_end> -y <year> -n <n_eofs> -f <eof_year>
	# This calculates the pseudo PCs by projecting the warming patterns for years other than
	# the eof_year onto the EOFs for the eof_year.  Using these pseudo PCs and the EOFs for
	# eof_year, the warming patterns for multiple years can be reconstructed (but there might
	# be an error)

5.  ./calc_SST_PCs_scalings.py      -r <run_type> -s <ref_start> -e <ref_end> -y <year_start> -z <year_end> -f <eof_year> -p <period> -n <n_eofs>
	# This calculates the scalings that are required to apply to the actual PCs in the eof_year
	# to get the pseudo PCs calculated in step 4.
	
6.  ./gen_syn_SST_PCs.py            -r <run_type> -s <ref_start> -e <ref_end> -y <year> -n <n_eofs> -a <n_samples> -t <sample strategy>

# After this step we have a set of EOFs (modes of warming) and a set of
# synthetic PCAs (contribution of modes of warming to overall SST warming) 
# that are sampled so as to represent the GMSST distribution well.
# These synthetic PCAs and EOFs can be combined to produce a number of warming
# patterns at a particular date.

# To generate a timeseries of SST warmings we can generate a timeseries of PCAs
# which are based on the cmip5 ensemble.  We first need to project the SST
# warmings over the cmip5 timeseries onto the EOFs for our reference year to generate 
# a series of projected PCAs over time.
# These projected PCAs show the evolution of the contribution of the modes of
# warming (the EOFs) as they change over time.  They are only based on the cmip5
# ensemble and not the synthetic PCAs.

5.  ./project_SST_warming_PCs.py    -r <run_type> -s <ref_start> -e <ref_end> -w <warming>|-y <year> -e <n_eofs> -f <eof_year>

# Do this for every decade of interest range(2010, 2100, 10) to produce the 
# projected PCAs.  We can then use these trajectories to modify the synthetic PCAs
# to generate timeseries of synthetic PCAs which will then create timeseries of
# synthetic SSTs.
