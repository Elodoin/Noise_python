import os
import sys
import glob
from datetime import datetime
import numpy as np
import scipy
import obspy
import matplotlib.pyplot as plt
import noise_module
import time
import pyasdf
import pandas as pd
import itertools
from obspy.io.sac.sactrace import SACTrace
from mpi4py import MPI


'''
this script reads from the h5 file for each station (containing all pre-processed and fft-ed traces) and then
computes the cross-correlations between each station-pair at an overlapping time window.

this version is implemented with MPI (Nov.09.2018)

this optimized version runs 5 times faster than the previous one by 1) pulling the prcess of making smoothed spectrum
of the source outside of the receiver loop, 2) take advantage the linear relationship of ifft to average the spectrum
first before doing ifft in cross-correlaiton functions and 3) sacrifice the disk memory (by 1.5 times) to improve the 
I/O speed (by 4 times)  (Jan,20,2019)
'''

#ttt=time.time()
#------some useful absolute paths-------
#FFTDIR = '/n/flashlfs/mdenolle/KANTO/DATA/FFT'
#STACKDIR = '/n/flashlfs/mdenolle/KANTO/DATA/STACK'
#FFTDIR = '/n/regal/denolle_lab/cjiang/FFT1'
#STACKDIR = '/n/regal/denolle_lab/cjiang/STACK1'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'

FFTDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT_10Hz'
CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF'
STACKDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/STACK_10Hz'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations.txt'
tcomp  = ['EHZ','EHE','EHN','HNU','HNE','HNN']


#-----some control parameters------
data_type = 'FFT'
save_day = True
downsamp_freq=20
dt=1/downsamp_freq
freqmin=0.05
freqmax=4
cc_len=3600
step=1800
maxlag=800
#method='deconv'
method='raw'
#method='coherence'

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------


#-------form a station pair to loop through-------
if rank ==0:
    tfiles = glob.glob(os.path.join(FFTDIR,'*.h5'))
    tfiles = sorted(tfiles)
    pairs  = list(itertools.combinations(tfiles,2))
    locs = pd.read_csv(locations)
    sta  = list(locs.iloc[:]['station'])
    splits = len(pairs)
else:
    splits,pairs,locs,sta = [None for _ in range(4)]

#------split the common variables------
splits = comm.bcast(splits,root=0)
pairs  = comm.bcast(pairs,root=0)
locs   = comm.bcast(locs,root=0)
sta    = comm.bcast(sta,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        t10=time.time()
        source,receiver = pairs[ii][0],pairs[ii][1]
        print('source '+source.split('/')[-1]+' receiver '+receiver.split('/')[-1]+' rank '+str(rank))

        t0=time.time()
        fft_h5   = source
        fft_ds_s = pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r')
        fft_h5   = receiver
        fft_ds_r = pyasdf.ASDFDataSet(fft_h5, mpi=False, mode='r')
        
        #-------get source information------
        tt0=time.time()
        net_sta_s = fft_ds_s.waveforms.list()[0]
        staS = net_sta_s.split('.')[1]
        netS = net_sta_s.split('.')[0]
        tt1=time.time()

        tindx = sta.index(staS)
        slat  = locs.iloc[tindx]['latitude']
        slon  = locs.iloc[tindx]['longitude']
        path_list_s = fft_ds_s.auxiliary_data[data_type].list()

        #------get receiver information--------
        net_sta_r = fft_ds_r.waveforms.list()[0]
        staR = net_sta_r.split('.')[1]
        netR = net_sta_r.split('.')[0]

        tindx = sta.index(staR)
        rlat  = locs.iloc[tindx]['latitude']
        rlon  = locs.iloc[tindx]['longitude']
        path_list_r = fft_ds_r.auxiliary_data[data_type].list()

        #---------initialize index and stacking arrays to save sac files---------
        indx = np.zeros(shape=(len(tcomp),len(tcomp)),dtype=np.int16)       # indx contains the comp information for each station pair
        tlen = int((2*maxlag)/dt+1)
        ncorr = np.zeros(shape=(indx.size,tlen),dtype=np.float32)

        t1=time.time()
        print('prepare source+receiver takes '+str(t1-t0)+' s; read station '+str(tt1-tt0))
        
        #---------loop through each component of the source------
        for jj in range(len(path_list_s)):
            t2=time.time()
            paths = path_list_s[jj]
            compS = fft_ds_s.auxiliary_data[data_type][paths].parameters['component']

            #-----------get the parameter of Nfft-----------
            Nfft = fft_ds_s.auxiliary_data[data_type][paths].parameters['nfft']
            Nseg = fft_ds_s.auxiliary_data[data_type][paths].parameters['nseg']
            
            fft1= fft_ds_s.auxiliary_data[data_type][paths].data[:,:Nfft//2] 
            source_std = fft_ds_s.auxiliary_data[data_type][paths].parameters['std']

            tt0=time.time()
            #-----get smoothed spectrum of source for decon-type cross correlations----
            sfft1 = noise_module.running_abs_mean(np.abs(fft1.reshape(fft1.size,)),10)
            sfft1 = sfft1.reshape(Nseg,Nfft//2)
            t3=time.time()
            print('calculate smooth spec takes '+str(t3-tt0))

            #-------day information------
            tday  = paths[-10:]

            for compR in tcomp:
                #------ find the corresponding path for the receiver for that day ----------
                tpath = '_'.join(['fft',netR,staR,compR,tday])

                #-------if it exists-------        
                if tpath in path_list_r:
                    pathr = tpath
                    #print(str(pathr))
    
                    t4=time.time()
                    fft2= fft_ds_r.auxiliary_data[data_type][pathr].data[:,:Nfft//2]
                    receiver_std = fft_ds_r.auxiliary_data[data_type][pathr].parameters['std']
                    t5=time.time()

                    #---------- check the existence of earthquakes ----------
                    rec_ind = np.where(receiver_std < 10)[0]
                    sou_ind = np.where(source_std < 10)[0]

                    #-----note that Hi-net has a few mi-secs differences to Mesonet in terms starting time-----
                    bb,indx1,indx2=np.intersect1d(sou_ind,rec_ind,return_indices=True)
                    indx1=sou_ind[indx1]
                    indx2=rec_ind[indx2]
                    if (len(indx1)==0) | (len(indx2)==0):
                        continue

                    t6=time.time()
                    #-----------do daily cross-correlations now-----------
                    #corr,tcorr=noise_module.correlate(fft1[indx1,:Nfft//2],fft2[indx2,:Nfft//2],np.round(maxlag),dt,Nfft,method)
                    corr,tcorr=noise_module.optimized_correlate(fft1[indx1,:Nfft//2],fft2[indx2,:Nfft//2],\
                            sfft1[indx1,:Nfft//2],np.round(maxlag),dt,Nfft,method)
                    t7=time.time()
                    print('source read '+str(t3-t2)+'s, receiver '+str(t5-t4)+'s & len '+str(len(indx1))+', cross takes '+str(t7-t6)+'s')

                    #--------find the index to store data--------
                    indx[tcomp.index(compS)][tcomp.index(compR)]=1
                    nindx=tcomp.index(compS)*len(tcomp)+tcomp.index(compR)

                    #--------linear stackings---------
                    if corr.ndim==2:
                        y = np.mean(corr,axis=0)
                    elif corr.ndim==1:
                        y = corr
                    else:
                        continue
                    ncorr[nindx][:] = ncorr[nindx][:] + y

                    #-------pws stacking--------
                    #y2 = noise_module.pws(np.array(corr),2,1/dt,5.)
                    
                    if save_day:
                        t8=time.time()
                        #---------------keep the daily cross-correlation into a hdf5 file--------------
                        tsource = os.path.join(CCFDIR,netS+"."+staS)
                        if os.path.exists(tsource)==False:
                            os.mkdir(tsource)
                        
                        fft_h5 = os.path.join(tsource,netS +"." + staS + "." + netR + "." + staR + '.h5')
                        crap   = np.zeros(corr.shape)
                        if not os.path.isfile(fft_h5):
                            with pyasdf.ASDFDataSet(fft_h5,mpi=False) as ds:
                                pass 
                        #else:
                        #    print([netS+"."+staS+"."+netR+"."+staR+'.h5', 'Already exists',obspy.UTCDateTime()])

                        with pyasdf.ASDFDataSet(fft_h5,mpi=False) as ccf_ds:
                            parameters = {'dt':dt, 'maxlag':maxlag, 'method':str(method)}

                            #------save the time domain cross-correlation functions-----
                            path = '_'.join(['ccfs',str(method),netS,staS,netR,staR,compS,compR,tday])
                            new_data_type = compS+compR
                            crap = corr
                            ccf_ds.add_auxiliary_data(data=crap, data_type=new_data_type, path=path, parameters=parameters)

                            #------save the freq domain cross-correlation functions for future C3-----
                            #path = '_'.join(['ccfs.f',str(method),netS,staS,netR,staR,compS,compR,tday])
                            #crap = np.squeeze(scorr)
                            #ccf_ds.add_auxiliary_data(data=crap, data_type=new_data_type, path=path, parameters=parameters)

                        del ccf_ds, crap, parameters, path, corr,tcorr
                        t9=time.time()
                        print('save sac takes '+str(t9-t8))
 
        #-------ready to write into files---------
        indx_array = np.where(indx==1)

        #-------first dim is for source and second for receiver-----
        for jj in range(len(indx_array[0])):
            compS = tcomp[indx_array[0][jj]]
            compR = tcomp[indx_array[1][jj]]
            dir1 = '.'.join([netS,staS])
            dir2 = '_'.join([netS,staS,netR,staR,compS,compR])
            nindx = indx_array[0][jj]*len(tcomp)+indx_array[1][jj]

            #------------make sure the directory exists----------
            if os.path.exists(os.path.join(STACKDIR,dir1))==False:
                os.mkdir(os.path.join(STACKDIR,dir1))    

            #--------save two stacked traces as SAC files---------
            filename1 = os.path.join(STACKDIR,dir1,dir2+'_ls.SAC')

            sac1 = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,
                            delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=ncorr[nindx][:])

            sac1.write(filename1,byteorder='big')

        #del pcorr
        del fft_ds_s, fft_ds_r, path_list_r, path_list_s, fft1, fft2, source_std, receiver_std, ncorr

        t11 = time.time()
        print('it takes '+str(t11-t10)+' s to process a station pair in step 2')

#ttt1=time.time()
#print('cc takes '+str(ttt1-ttt)+' s')

comm.barrier()
if rank == 0:
    sys.exit()