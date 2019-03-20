import os
import sys
import glob
import numpy as np
import scipy
import obspy
import matplotlib.pyplot as plt
import noise_module
import time
import pyasdf
import pandas as pd
from mpi4py import MPI


'''
this script loop through the days by using MPI and compute cross-correlation functions for each station-pair at that
day when there are overlapping time windows. (Nov.09.2018)

optimized to run ~5 times faster by 1) making smoothed spectrum of the source outside of the receiver loop; 2) taking 
advantage of the linearality of ifft to average the spectrum first before doing ifft in cross-correlaiton functions, 
and 3) sacrifice storage (by 1.5 times) to improve the I/O speed (by 4 times). 
Thanks to Zhitu Ma for thoughtful discussions.  (Jan,28,2019)

new updates include 1) remove the need of input station.lst by listing available HDF5 files, 2) make use of the inventory
for lon, lat information, 3) add new parameters to HDF5 files needed for later CC steps and 4) make data_types and paths
in the same format (Feb.15.2019). 

add the functionality of auto-correlations (Feb.22.2019). Note that the auto-cc is normalizing each station to its Z comp.

modify the structure of ASDF files to make it more flexable for later stacking and matrix rotation (Mar.06.2019)

modify to read the fft for all stations in the right beginning to save time (Mar.14.2019)
'''

ttt0=time.time()

rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
FFTDIR = os.path.join(rootpath,'FFT')
CCFDIR = os.path.join(rootpath,'CCF/test')

#-----some control parameters------
flag=True              #output intermediate variables and computing times
auto_corr=False         #include single-station auto-correlations or not
smooth_N=10             #window length for smoothing the spectrum amplitude
num_seg=4
downsamp_freq=20
dt=1/downsamp_freq
cc_len=3600
step=1800
maxlag=800              #enlarge this number if to do C3
method='deconv'
start_date = '2011_03_01'
end_date   = '2011_03_25'
inc_days   = 1

if auto_corr and method=='coherence':
    raise ValueError('Please set method to decon: coherence cannot be applied when auto_corr is wanted!')

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

#-------form a station pair to loop through-------
if rank ==0:
    if not os.path.isdir(CCFDIR):
        os.mkdir(CCFDIR)

    sfiles = sorted(glob.glob(os.path.join(FFTDIR,'*.h5')))
    day = noise_module.get_event_list(start_date,end_date,inc_days)
    splits = len(day)
else:
    splits,sfiles,day = [None for _ in range(3)]

#------split the common variables------
splits = comm.bcast(splits,root=0)
day    = comm.bcast(day,root=0)
sfiles   = comm.bcast(sfiles,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        iday = day[ii]

        tt=time.time()
        #-----------get parameters of Nfft and Nseg--------------
        with pyasdf.ASDFDataSet(sfiles[0],mpi=False,mode='r') as ds:
            data_types = ds.auxiliary_data.list()
            paths      = ds.auxiliary_data[data_types[0]].list()
            Nfft = ds.auxiliary_data[data_types[0]][paths[0]].parameters['nfft']
            Nseg = ds.auxiliary_data[data_types[0]][paths[0]].parameters['nseg']
        ncomp = len(data_types)
        nsta  = len(sfiles)

        #----double check the ncomp parameters by opening a few stations------
        for ii in range(1,4):
            with pyasdf.ASDFDataSet(sfiles[ii],mpi=False,mode='r') as ds:
                data_types = ds.auxiliary_data.list()
                if len(data_types) > ncomp:
                    ncomp = len(data_types)
                    print('first station of %s misses other components' % (sfiles[0]))

        #----loop through each data segment-----
        nhours = int(np.ceil(Nseg/num_seg))
        for iseg in range(num_seg):
            
            #---index for the data chunck---
            sindx1 = iseg*nhours
            if iseg==num_seg-1:
                nhours = Nseg-iseg*nhours
            sindx2 = sindx1+nhours

            if nhours==0 or nhours <0:
                raise ValueError('nhours<=0, please double check')

            if flag:
                print('working on %dth segments of the daily FFT'% iseg)

            #-------make a crutial estimate on memory needed for the FFT of all stations: defaultly using float32--------
            memory_size = nsta*ncomp*Nfft/2*nhours*8/1024/1024/1024
            if memory_size > 8:
                raise MemoryError('Memory exceeds 8 GB! No enough memory to load them all once!')

            print('initialize the array ~%3.1f GB for storing all cc data' % (memory_size))

            #---------------initialize the array-------------------
            cc_array = np.zeros((nsta*ncomp,nhours*Nfft//2),dtype=np.complex64)
            cc_std   = np.zeros((nsta*ncomp,nhours),dtype=np.float32)
            cc_flag  = np.zeros((nsta),dtype=np.int16)
            k=0

            ttl = time.time()
            #-----loop through all stations------
            for ifile in range(len(sfiles)):
                tfile = sfiles[ifile]

                with pyasdf.ASDFDataSet(tfile,mpi=False,mode='r') as ds:
                    data_types = ds.auxiliary_data.list()

                    #-----when some components are missing------
                    if len(data_types) < ncomp:
                        cc_flag[ifile]=1

                        for icomp in data_types:
                            if icomp[-1]=='E':
                                iindx = 0
                            elif icomp[-1]=='N':
                                iindx = 1
                            else:
                                iindx = 2
                            tpaths = ds.auxiliary_data[icomp].list()

                            if iday in tpaths:
                                if flag:
                                    print('find %dth data chunck for station %s day %s' % (iseg,tfile.split('/')[-1],iday))
                                indx = ifile*ncomp+iindx
                                data  = ds.auxiliary_data[icomp][iday].data[sindx1:sindx2,:]
                                cc_array[indx][:]= data.reshape(data.size)
                                std   = ds.auxiliary_data[icomp][iday].parameters['std']
                                cc_std[indx][:]  = std[sindx1:sindx2]
                    else:

                        #-----good orders when all components are available-----
                        for ii in range(len(data_types)):
                            icomp = data_types[ii]
                            tpaths = ds.auxiliary_data[icomp].list()
                            if iday in tpaths:
                                if flag:
                                    print('find %dth data chunck for station %s day %s' % (iseg,tfile.split('/')[-1],iday))
                                indx = ifile*ncomp+ii
                                data  = ds.auxiliary_data[icomp][iday].data[sindx1:sindx2,:]
                                cc_array[indx][:]= data.reshape(data.size)
                                std   = ds.auxiliary_data[icomp][iday].parameters['std']
                                cc_std[indx][:]  = std[sindx1:sindx2]

            tt0 = time.time()
            print('loading all FFT takes %6.4fs' % (tt0-ttl))

            #------loop I of each source-----
            for isource in range(len(sfiles)-1):
                source = sfiles[isource]
                staS = source.split('/')[-1].split('.')[1]
                netS = source.split('/')[-1].split('.')[0]
                if flag:
                    print('source: %s %s' % (staS,netS))

                with pyasdf.ASDFDataSet(source,mpi=False,mode='r') as fft_ds_s:

                    #-------get lon and lat information from inventory--------
                    temp = fft_ds_s.waveforms.list()
                    invS = fft_ds_s.waveforms[temp[0]]['StationXML']
                    lonS = invS[0][0].longitude
                    latS = invS[0][0].latitude
                    if flag:
                        print('source coordinates: %8.2f %8.2f' % (lonS,latS))

                    #----loop II of each component for source A------
                    data_types_s = fft_ds_s.auxiliary_data.list()
                    
                    #---assume Z component lists the last in data_types of [E,N,Z/U]-----
                    if auto_corr:

                        auto_corr_flag = True
                        if data_types_s[-1]=='Z' or data_types_s[-1]=='U':
                            auto_indx = isource*ncomp+2
                        else:
                            print('no Z component for autocorrelation for %s'%source)
                            auto_corr_flag = False
                    
                    for icompS in range(len(data_types_s)):
                        if flag:
                            print("reading source %s for day %s" % (staS,icompS))
                        
                        data_type_s = data_types_s[icompS]
                        path_list_s = fft_ds_s.auxiliary_data[data_type_s].list()

                        #---missing a component---
                        if cc_flag[isource]:
                            if data_type_s[-1]=='E':
                                cc_indx = isource*ncomp
                            elif data_type_s[-1]=='N':
                                cc_indx = isource*ncomp+1
                            else:
                                cc_indx = isource*ncomp+2
                        
                        #-------or not-------
                        else:
                            cc_indx = isource*ncomp+icompS

                        #-----iday exists for source A---
                        if iday in path_list_s:

                            t1=time.time()
                            
                            fft1 = cc_array[cc_indx][:]
                            source_std = cc_std[cc_indx][:]
                            
                            t2=time.time()
                            #-----------get the smoothed source spectrum for decon later----------
                            if method == 'deconv':

                                #-----normalize single-station cc to z component-----
                                if auto_corr and auto_corr_flag:
                                    fft_temp = cc_array[auto_indx][:]
                                    temp     = noise_module.moving_ave(np.abs(fft_temp),smooth_N)
                                else:
                                    temp = noise_module.moving_ave(np.abs(fft1),smooth_N)

                                #--------think about how to avoid temp==0-----------
                                try:
                                    sfft1 = np.conj(fft1)/temp**2
                                except ValueError:
                                    raise ValueError('smoothed spectrum has zero values')

                            elif method == 'coherence':
                                temp = noise_module.moving_ave(np.abs(fft1),smooth_N)
                                try:
                                    sfft1 = np.conj(fft1)/temp
                                except ValueError:
                                    raise ValueError('smoothed spectrum has zero values')

                            elif method == 'raw':
                                sfft1 = fft1
                            sfft1 = sfft1.reshape(nhours,Nfft//2)

                            t3=time.time()
                            if flag:
                                print('read S %6.4fs, smooth %6.4fs' % ((t2-t1), (t3-t2)))

                            #-----------now loop III for each receiver B----------
                            if auto_corr and auto_corr_flag:
                                tindex = isource
                            else:
                                tindex = isource+1

                            for ireceiver in range(tindex,len(sfiles)):
                                receiver = sfiles[ireceiver]
                                staR = receiver.split('/')[-1].split('.')[1]
                                netR = receiver.split('/')[-1].split('.')[0]
                                if flag:
                                    print('receiver: %s %s' % (staR,netR))
                                
                                with pyasdf.ASDFDataSet(receiver,mpi=False,mode='r') as fft_ds_r:

                                    #-------get lon and lat information from inventory--------
                                    temp = fft_ds_r.waveforms.list()
                                    invR = fft_ds_r.waveforms[temp[0]]['StationXML']
                                    lonR = invR[0][0].longitude
                                    latR = invR[0][0].latitude
                                    if flag:
                                        print('receiver coordinates: %8.2f %8.2f' % (lonR,latR))

                                    #-----loop IV of each component for receiver B------
                                    data_types_r = fft_ds_r.auxiliary_data.list()

                                    for icompR in range(len(data_types_r)):
                                        data_type_r = data_types_r[icompR]
                                        path_list_r = fft_ds_r.auxiliary_data[data_type_r].list()

                                        #---missing a component---
                                        if cc_flag[ireceiver]:
                                            if data_type_r[-1]=='E':
                                                cc_indx = ireceiver*ncomp
                                            elif data_type_r[-1]=='N':
                                                cc_indx = ireceiver*ncomp+1
                                            else:
                                                cc_indx = ireceiver*ncomp+2
                                        
                                        #-------or not-------
                                        else:
                                            cc_indx = ireceiver*ncomp+icompR

                                        #----if that day exists for receiver B----
                                        if iday in path_list_r:

                                            t4=time.time()
                                            fft2= cc_array[cc_indx][:]
                                            fft2=fft2.reshape(nhours,Nfft//2)
                                            receiver_std = cc_std[cc_indx][:]
                                            t5=time.time()

                                            #---------- check the existence of earthquakes ----------
                                            rec_ind = np.where(receiver_std < 10)[0]
                                            sou_ind = np.where(source_std < 10)[0]

                                            #-----note that Hi-net has a few mi-secs differences to Mesonet in terms starting time-----
                                            bb=np.intersect1d(sou_ind,rec_ind)
                                            if len(bb)==0:
                                                continue

                                            t6=time.time()
                                            corr=noise_module.optimized_correlate1(sfft1[bb,:],fft2[bb,:],\
                                                    np.round(maxlag),dt,Nfft,len(bb),method)
                                            t7=time.time()

                                            #---------------keep daily cross-correlation into a hdf5 file--------------
                                            cc_aday_h5 = os.path.join(CCFDIR,iday+'.h5')
                                            crap   = np.zeros(corr.shape)

                                            if not os.path.isfile(cc_aday_h5):
                                                with pyasdf.ASDFDataSet(cc_aday_h5,mpi=False) as ccf_ds:
                                                    pass 

                                            with pyasdf.ASDFDataSet(cc_aday_h5,mpi=False) as ccf_ds:
                                                parameters = noise_module.optimized_cc_parameters(dt,maxlag,str(method),lonS,latS,lonR,latR)

                                                #-----------make a universal change to component-----------
                                                if data_type_r[-1]=='U' or data_type_r[-1]=='Z':
                                                    compR = 'Z'
                                                elif data_type_r[-1]=='E':
                                                    compR = 'E'
                                                elif data_type_r[-1]=='N':
                                                    compR = 'N' 

                                                if data_type_s[-1]=='U' or data_type_s[-1]=='Z':
                                                    compS = 'Z'
                                                elif data_type_s[-1]=='E':
                                                    compS = 'E'
                                                elif data_type_s[-1]=='N':
                                                    compS = 'N' 

                                                #------save the time domain cross-correlation functions-----
                                                path = netR+'s'+staR+'s'+compR+str(iseg)
                                                data_type = netS+'s'+staS+'s'+compS

                                                crap = corr
                                                ccf_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                                            t8=time.time()
                                            if flag:
                                                print('read R %6.4fs, cc %6.4fs, write cc %6.4fs'% ((t5-t4),(t7-t6),(t8-t7)))
            
        tt1 = time.time()
        print('it takes %6.4fs to process %d hours of data [%d segment] in step 2' % (tt1-tt,nhours,num_seg))


ttt1=time.time()
print('all step 2 takes %6.4fs'%(ttt1-ttt0))

comm.barrier()
if rank == 0:
    sys.exit()
