import os
import glob
import time
import pyasdf
import numpy as np
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
wcoda = 500
maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq
comp1 = ['EHE','EHN','EHZ']
comp2 = ['HNE','HNN','HNU']

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
    sta  = locs.iloc[:]['station']
    pairs= noise_module.get_station_pairs(sta)
    daily_ccfs = glob.glob(os.path.join(C3DIR,'*.h5'))
    splits = len(daily_ccfs)
else:
    locs,pairs,daily_ccfs,splits=[None for _ in range(4)]

locs   = comm.bcast(locs,root=0)
pairs  = comm.bcase(pairs,root=0)
daily_ccfs   = comm.bcast(daily_ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        dayfile = daily_ccfs[ii]
        sta  = list(locs.iloc[:]['station'])

        with pyasdf.ASDFDataSet(dayfile,mpi=False,mode='r') as ds:
            data_types = ds.auxiliary_data.list()

            for ii in range(len(pairs)):

                #----source and receiver information----
                source,receiver = pairs[ii][0],pairs[ii][1]
                indx1 = sta.index(source)
                slat = locs.iloc[indx1]['latitude']
                slon = locs.iloc[indx1]['longitude']
                netS = locs.iloc[indx1]['network']
                if netS == 'E' or netS == 'OK':
                    compS = comp2[2]
                else:
                    compS = comp1[2]

                indx2 = sta.index(receiver)
                rlat = locs.iloc[indx2]['latitude']
                rlon = locs.iloc[indx2]['longitude']
                netR = locs.iloc[indx2]['network']
                if netR == 'E' or netR == 'OK':
                    compR = comp2[2]
                else:
                    compR = comp1[2]

                #----calculate window for cutting the ccfs-----
                dist = noise_module.get_distance(slon,slat,rlon,rlat)
                t1,t2 = noise_module.get_coda_window(dist,vmin,maxlag,wcoda)

                for ista in sta:
                    indx = sta.index(ista)
                    net = locs.iloc[indx]['network']
                    if indx == indx1 or indx == indx2:
                        continue
                    
                    #------ready to find the ccfs between ista and (S,R)-----
                    if net == 'E' or net == 'OK':
                        comp = comp2
                    else:
                        comp = comp1
                    
                    #---------loop through the components-----------
                    for icomp in range(len(comp)):

                        #-----find the index of the data_type and path------
                        if indx > indx1:
                            data_indx = indx*3+icomp
                            path_indx1 = indx1*3-indx*3-1+icomp
                            path_indx2 = indx2*3-indx*3-1+icomp
                            paths = ds.auxiliary_data[data_types[data_indx]].list()

                            SS_data = ds.auxiliary_data[data_types[data_indx]][paths[path_indx1]].data[:]
                            SR_data = ds.auxiliary_data[data_types[data_indx]][paths[path_indx2]].data[:]
                        elif indx > indx2:
   
