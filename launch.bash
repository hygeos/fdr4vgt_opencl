#!/bin/bash

dirname='/mnt/ceph/proj/FDR4VGT/input/probav'
fileout='/data/smaccl/output/toc.nc'
config='./fdr4vgt/probav_config.cfg'
demfile='/archive2/data/DEM/GLOBE/GTOPO30_DZ_MLUT.nc'
merra_aer='/archive2/data/MERRA2/aer_extinction/2014/MERRA2_400.tavg1_2d_aer_Nx.20141111.nc4'
merra_p2='/archive2/data/MERRA2/surf_pression_water_vapor/2014/MERRA2_400.tavg1_2d_slv_Nx.20141111.nc4'
smac_dir='/archive/proj/C3S/'
aer_file='./ANCILLARY/Aerosol_model_fraction.txt'
python fdr4vgt/process.py $dirname $demfile $merra_aer $merra_p2 $smac_dir $aer_file $fileout $config
