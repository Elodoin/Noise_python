import os
import glob
import time
import pyasdf
import pandas as pd
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
C3DIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3'

flag  = False
vmin  = 1.5
vmax  = 3
wcoda = 500
maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    #----check the directory of STACK----
    if os.path.exists(C3DIR)==False:
        os.mkdir(C3DIR)

    #-----other variables to share-----
    locs = pd.read_csv(locations)
    daily_ccfs = glob.glob(os.path.join(C3DIR,'*.h5'))
    splits = len(daily_ccfs)
else:
    locs,daily_ccfs,splits=[None for _ in range(4)]

locs   = comm.bcast(locs,root=0)
daily_ccfs   = comm.bcast(daily_ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        dayfile = daily_ccfs[ii]
        sta  = list(locs.iloc[:]['station'])

        with pyasdf.ASDFDataSet(dayfile,mpi=False,mode='r') as ds:
            data_types = ds.auxiliary_data.list()

            #-----loop through each station pair-----
            for ii in range(len(sta)-1):
                for jj in range(ii+1,len(sta)):

                    #----source and receiver information----
                    source = sta[ii]
                    receiver = sta[jj]
                    indx = sta.index(source)
                    slat = locs.iloc[indx]['latitude']
                    slon = locs.iloc[indx]['longitude']
                    netS = locs.iloc[indx]['network']
                    indx = sta.index(receiver)
                    rlat = locs.iloc[indx]['latitude']
                    rlon = locs.iloc[indx]['longitude']
                    netR = locs.iloc[indx]['network']

                    #----calculate window for cutting the ccfs-----

                    #------construct the data_type and path lists for X-A, and X-B ccfs-----

                    #---------a for loop through all 4 lag windonw--------


                    #---------stacking---------


#---------record its spectrum--------
