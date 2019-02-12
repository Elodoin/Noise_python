import os
import glob
import time
import pyasdf
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI

'''
this scripts takes the ASDF file outputed by script S2 (cc function for day x)
and get the spectrum of the coda part of the cc and store them in a
new HDF5 files for computing C3 functions in script S4
'''

t0=time.time()
#----------some common variables here----------
#CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_deconv'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'

CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_opt'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations_small.txt'
C3DIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3'

flag  = True
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
    sta  = sorted(locs.iloc[:]['station'])
    print(sta)
    pairs= noise_module.get_station_pairs(sta)
    daily_ccfs = glob.glob(os.path.join(CCFDIR,'*.h5'))
    splits = len(daily_ccfs)
else:
    locs,pairs,daily_ccfs,splits=[None for _ in range(4)]

locs   = comm.bcast(locs,root=0)
pairs  = comm.bcast(pairs,root=0)
daily_ccfs   = comm.bcast(daily_ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        dayfile = daily_ccfs[ii]
        if flag:
            print('work on day %s' % dayfile)
        sta  = sorted(locs.iloc[:]['station'])
        tt   = np.arange(-maxlag/dt+1, maxlag/dt)*dt

        with pyasdf.ASDFDataSet(dayfile,mpi=False,mode='r') as ds:
            data_types = ds.auxiliary_data.list()

            for ii in range(len(pairs)):

                #----source and receiver information----
                source,receiver = pairs[ii][0],pairs[ii][1]
                if flag:
                    print('doing C3 for %dth station pair: %s %s' % (ii,source,receiver))

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

                if flag:
                    print('infor read for the station pair! Only do Z component now')

                #----calculate window for cutting the ccfs-----
                dist = noise_module.get_distance(slon,slat,rlon,rlat)
                t1,t2 = noise_module.get_coda_window(dist,vmin,maxlag,wcoda)
                if flag:
                    print('interstation distance %f and time window of [%f %f]' %(dist,t1,t2))

                for ista in sta:
                    indx = sta.index(ista)
                    net = locs.iloc[indx]['network']
                    if indx == indx1 or indx == indx2:
                        continue
                    
                    if flag:
                        print('virtural source %s' % ista)
                    
                    #------ready to find the ccfs between ista and (S,R)-----
                    if net == 'E' or net == 'OK':
                        comp = comp2
                    else:
                        comp = comp1
                    
                    #---------loop through the components-----------
                    for icomp in range(len(comp)):
                        if flag:
                            print('Source %s %d, Receiver %s %d, Virtual %s %d %s' % (source,indx1,receiver,indx2,ista,indx,comp[icomp]))

                        #-----find the index of the data_type and path----
                        #------for situation of A-> (C,F)------
                        if indx < indx1:
                            dtype_indx1 = indx*3+icomp
                            dtype_indx2 = indx*3+icomp
                            path_indx1 = (indx1-indx-1)*3+2
                            path_indx2 = (indx2-indx-1)*3+2
                            paths1 = ds.auxiliary_data[data_types[dtype_indx1]].list()
                            paths2 = ds.auxiliary_data[data_types[dtype_indx2]].list()

                            SS_data = ds.auxiliary_data[data_types[dtype_indx1]][paths1[path_indx1]].data[:]
                            SR_data = ds.auxiliary_data[data_types[dtype_indx2]][paths2[path_indx2]].data[:]
                        
                        #------for situation of E->(C,F)-------
                        elif indx < indx2:
                            dtype_indx1 = indx1*3+2
                            dtype_indx2 = indx*3+icomp
                            path_indx1 = (indx-indx1-1)*3+icomp
                            path_indx2 = (indx2-indx-1)*3+icomp
                            paths1 = ds.auxiliary_data[data_types[dtype_indx1]].list()
                            paths2 = ds.auxiliary_data[data_types[dtype_indx2]].list()

                            SS_data = ds.auxiliary_data[data_types[dtype_indx1]][paths1[path_indx1]].data[:]
                            SS_data = SS_data[::-1]
                            SR_data = ds.auxiliary_data[data_types[dtype_indx2]][paths2[path_indx2]].data[:]

                        #------for situation of G->(C,F)-------
                        else:
                            dtype_indx1 = indx1*3+2
                            dtype_indx2 = indx2*3+2
                            path_indx1 = (indx-indx1-1)*3+icomp
                            path_indx2 = (indx-indx2-1)*3+icomp
                            paths1 = ds.auxiliary_data[data_types[dtype_indx1]].list()
                            paths2 = ds.auxiliary_data[data_types[dtype_indx2]].list()

                            SS_data = ds.auxiliary_data[data_types[dtype_indx1]][paths1[path_indx1]].data[:]
                            SS_data = SS_data[::-1]
                            SR_data = ds.auxiliary_data[data_types[dtype_indx2]][paths2[path_indx2]].data[:]
                            SR_data = SR_data[::-1]

                        if flag:    
                            print('dtype1 %s path1 %s' % (data_types[dtype_indx1],paths1[path_indx1]))
                            print('dtype2 %s path2 %s' % (data_types[dtype_indx2],paths2[path_indx2]))

                        #--------begin FFT-------
                        #cc=noise_module.C3_process(SS_data,SR_data)