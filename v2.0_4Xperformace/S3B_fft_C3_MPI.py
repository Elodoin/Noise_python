import os
import sys
import glob
import time
import scipy
import pyasdf
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI
from scipy.fftpack.helper import next_fast_len

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

CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF'
C3DIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3'

flag  = True
vmin  = 1.5
wcoda = 1200
maxlag = 2000
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
    daily_ccfs = glob.glob(os.path.join(CCFDIR,'*.h5'))
    splits = len(daily_ccfs)
else:
    daily_ccfs,splits = [None for _ in range(2)]

daily_ccfs   = comm.bcast(daily_ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

#--------MPI loop through each day---------
for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        t00=time.time()
        dayfile = daily_ccfs[ii]
        tt   = np.arange(-maxlag/dt+1, maxlag/dt)*dt
        if flag:
            print('work on day %s' % dayfile)

        #------------dayfile contains everything needed----------
        with pyasdf.ASDFDataSet(dayfile,mpi=False,mode='r') as ds:
            data_types = ds.auxiliary_data.list()

            #------make a list of stations-------
            ncomp = len(comp1)
            sta = []
            net = []
            for ii in range(len(data_types)//ncomp):
                sta.append(data_types[ii*ncomp].split('s')[1])
                net.append(data_types[ii*ncomp].split('s')[0])
            pairs = noise_module.get_station_pairs(sta)

            #-------make a crutial estimate on the memory need--------
            ccfs_size = 2*int(maxlag*downsamp_freq)
            memory_size = len(pairs)*ncomp*ncomp*4*ccfs_size/1024/1024/1024
            if memory_size > 20:
                raise MemoryError('Memory exceeds 10 GB! No enough memory to load them all once!')

            #------cc_array holds all ccfs and npairs tracks the number of pairs for each source-----
            cc_array = np.zeros((len(pairs)*ncomp*ncomp,ccfs_size),dtype=np.float32)
            cc_npair = np.zeros(len(data_types),dtype=np.int16)
            cc_dist  = np.zeros(len(pairs)*ncomp*ncomp),dtype=np.float32)
            c3_tt    = np.zeros((len(pairs)*ncomp*ncomp,2),dtype=np.float32)
            k=0

            #-------load everything here to avoid repeating read HDF5 files------
            for ii in range(len(data_types)):
                data_type=data_types[ii]
                tpaths = ds.auxiliary_data[data_type].list()
                cc_npair[ii] = len(tpaths)

                for jj in range(len(tpaths)):
                    tpath = tpaths[jj]
                    cc_array[k][:]=ds.auxiliary_data[data_type][tpath].data[:]
                    cc_dist[k] = ds.auxiliary_data[data_type][tpath].parameters['dist']
                    c3_tt[k][0],c3_tt[k][1] = noise_module.get_coda_window(dist,vmin,maxlag,wcoda)
                    if flag:
                        print('interstation distance %f and time window of [%f %f]' %(cc_dist[k],c3_tt[k][0],c3_tt[k][1]))
                    k+=1

            if k!= len(pairs)*ncomp*ncomp:
                raise ValueError('the size of cc_array seems not right [%d vs %d]' % (k,len(pairs)*ncomp*ncomp))

            exit()
            #-----loop each station pair-----
            for ii in range(len(pairs)):
                source,receiver = pairs[ii][0],pairs[ii][1]
                if flag:
                    print('doing C3 for %dth station pair: [%s %s]' % (ii,source,receiver))

                #----load source + receiver information----
                indx1 = sta.index(source)
                netS  = net[indx1]
                #-------only deal with vertical------
                if netS == 'E' or netS == 'OK':
                    compS = comp2[2]
                else:
                    compS = comp1[2]

                indx2 = sta.index(receiver)
                netR  = net[indx2]
                if netR == 'E' or netR == 'OK':
                    compR = comp2[2]
                else:
                    compR = comp1[2]

                if flag:
                    print('infor read for the station pair! Only do Z component now')

                #-----------initialize some variables-----------
                Nfft  = int(next_fast_len(int(wcoda)/dt+1)))
                npair = 0
                cc_P = np.zeros(Nfft,dtype=np.complex64)
                cc_N = cc_P
                cc_final = cc_P

                #------loop through all virtual sources------
                for ista in sta:
                    indx = sta.index(ista)
                    net = net[indx]
                    if indx == indx1 or indx == indx2:
                        continue
                    if flag:
                        print('virtural source %s' % ista)
                    
                    #------use all components of virtual source-----
                    if net == 'E' or net == 'OK':
                        comp = comp2
                    else:
                        comp = comp1
                    
                    #---------loop through all 3 components-----------
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
                            path_indx2 = (indx2-indx-1)*3+2
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

                        #--------cast all processing into C3-process function-------
                        ccp,ccn=noise_module.C3_process(SS_data,SR_data,Nfft,t1,t2,tt)
                        cc_P+=ccp
                        cc_N+=ccn
                        npair+=1
                    
                    if flag:
                        print('moving to next virtual source')

                #-------stack contribution from all virtual source------
                cc_P = cc_P/npair
                cc_N = cc_N/npair
                cc_final = 0.5*cc_P + 0.5*cc_N
                cc_final = np.real(scipy.fftpack.ifft(cc_final, Nfft))
                if flag:
                    print('start to ouput to HDF5 file')

                #------ready to write into HDF5 files-------
                c3_h5 = dayfile
                crap  = np.zeros(cc_final.shape)

                if not os.path.isfile(c3_h5):
                    with pyasdf.ASDFDataSet(c3_h5,mpi=False,mode='w') as ds:
                        pass 

                with pyasdf.ASDFDataSet(c3_h5,mpi=False,mode='a') as ccf_ds:
                    parameters = {'dt':dt, 'maxlag':maxlag, 'wcoda':wcoda, 'vmin':vmin}

                    #------save the time domain cross-correlation functions-----
                    path = '_'.join([netR,receiver,compS])
                    new_data_type = netS+'s'+source+'s'+compR
                    crap = cc_final
                    ccf_ds.add_auxiliary_data(data=crap, data_type=new_data_type, path=path, parameters=parameters)

            t10=time.time()
            print('one station takes %f s to compute' % (t10-t00))

t111 = time.time()
print('C3 takes %f s in total' % (t111-t0))

comm.barrier()
if rank == 0:
    sys.exit()