import os
import glob
import time
import pyasdf
import noise_module
from mpi4py import MPI

'''
this scripts takes the h5 file from script S2 (cc function for day x)
and get the spectrum of the coda part of the cc and store them in a
new HDF5 files for computing C3 functions in script S4
'''

t0=time.time()
#----------some common variables here----------
#CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_deconv'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'

CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_opt'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations.txt'
CCFDIR_C3 = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3'

flag = True
vel = [1,3]
maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq

if rank == 0:
    #----check the directory of STACK----
    if os.path.exists(CCFDIR_C3)==False:
        os.mkdir(CCFDIR_C3)

    #-----other variables to share-----
    locs = pd.read_csv(locations)
    sta  = list(locs.iloc[:]['station'])
    pairs= noise_module.get_station_pairs(sta)
    ccfs = glob.glob(os.path.join(CCFDIR,'*.h5'))
    splits = len(pairs)
else:
    locs,sta,ccfs,splits=[None for _ in range(4)]

locs   = comm.bcast(locs,root=0)
pairs  = comm.bcast(pairs,root=0)
ccfs   = comm.bcast(ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        iday = day[ii]
#-----loop through each station pair-----

#---------record its spectrum--------
