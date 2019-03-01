import os
import sys
import glob
import numpy as np
from scipy.fftpack import ifft
import obspy
import time
import pandas as pd
from mpi4py import MPI
import h5py
from numba import jit
import matplotlib.pyplot as plt


'''
this script uses the day as the outmost loop and then computes the cross-correlations between each station-pair at 
that day for overlapping time window.

this version is implemented with MPI (Nov.09.2018)

this optimized version runs 5 times faster than the previous one by 1) pulling the prcess of making smoothed spectrum
of the source outside of the receiver loop, 2) take advantage the linear relationship of ifft to average the spectrum
first before doing ifft in cross-correlaiton functions and 3) sacrifice the disk memory (by 1.5 times) to improve the 
I/O speed (by 4 times)  (Jan,28,2019)
'''

@jit('complex64[:](complex64[:],complex64[:])')
def optimized_correlate(fft1_smoothed_abs,fft2):
    '''
    Optimized version of the correlation functions: put the smoothed 
    source spectrum amplitude out of the inner for loop. 
    It also takes advantage of the linear relationship of ifft, so that
    stacking in spectrum first to reduce the total number of times for ifft,
    which is the most time consuming steps in the previous correlation function  
    '''
    Nfft = fft2.shape[1]*2
    nwin = fft2.shape[0]

    #------convert all 2D arrays into 1D to speed up--------
    tcorr = np.zeros(Nfft//2,dtype=np.complex64)
    k = 0
    for ii in range(nwin):
        for jj in range(Nfft//2):
            tcorr[jj]+=fft1_smoothed_abs[ii][jj]*fft2[ii][jj]
        k+=1

    tcorr = tcorr/k
    ncorr = np.zeros(Nfft,dtype=np.complex64)
    ncorr[:Nfft//2] = tcorr[:Nfft//2]
    ncorr[-(Nfft//2)+1:]=np.conj(ncorr[1:(Nfft//2)])[::-1]
    ncorr[0]=0.
    
    return ncorr


@jit('float32[:](float32[:],int16)')
def moving_ave(A,N):
    '''
    Numba compiled function to do running smooth average.
    N is the the half window length to smooth
    A and B are both 1-D arrays (which runs faster compared to 2-D operations)
    '''
    #A = np.r_[A[:N],A,A[-N:]]
    B = np.zeros(A.shape,A.dtype)
    
    tmp=0.
    for pos in range(A.size):
        # do summing only once
        if pos <N or pos > A.size-N:
            B[pos]=A[pos]
        elif pos==N:
            for i in range(-N,N+1):
                tmp+=A[pos+i]
            B[pos]=tmp/(2*N+1)
        else:
            tmp=tmp-A[pos-N-1]+A[pos+N]
            B[pos]=tmp/(2*N+1)
        if B[pos]==0:
            B[pos]=1
    return B



ttt0=time.time()
#------some useful absolute paths-------
FFTDIR = '/Users/chengxin/Documents/Harvard/Julia/julia_noise_shared/data_test/FFT'
CCFDIR = '/Users/chengxin/Documents/Harvard/Julia/julia_noise_shared/data_test/CCF/python'
locations = '/Users/chengxin/Documents/Harvard/Julia/julia_noise_shared/data_test/locations.txt'
events    = '/Users/chengxin/Documents/Harvard/Julia/julia_noise_shared/data_test/events.txt'

#-----some control parameters------
save_day = False
downsamp_freq=20
dt=1/downsamp_freq
freqmin=0.05
freqmax=4
cc_len=3600
step=1800
maxlag=800
method='deconv'
#method='raw'
#method='coherence'

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

#-------form a station pair to loop through-------
if rank ==0:
    daylists = pd.read_csv(events)
    day  = list(daylists.iloc[:]['days'])
    splits = len(day)
else:
    splits,day = [None for _ in range(2)]

#------split the common variables------
splits = comm.bcast(splits,root=0)
day    = comm.bcast(day,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        iday = day[ii]

        time_ifft=0.
        time_mul=0.
        time_read=0.
        time_write=0.
        time_flip=0.
        time_source=0.
        n_ifft=0
        n_mul=0
        n_read=0
        n_write=0
        n_flip=0
        n_source=0

        sources = sorted(glob.glob(os.path.join(FFTDIR,'*.h5')))

        #------loop I of each source-----
        for isource in range(len(sources)):

            fft_ds_s = h5py.File(sources[isource],'r')
            list_s    = list(fft_ds_s.keys())
            
            #------loop II of each component-----
            for icompS in range(len(list_s)//2):

                staS = list_s[icompS*2].split('_')[2]
                netS = list_s[icompS*2].split('_')[1]
                compS = list_s[icompS*2].split('_')[3]

                t1=time.time()
                Nfft = fft_ds_s[list_s[icompS*2+1]].attrs['nfft']
                Nseg = fft_ds_s[list_s[icompS*2+1]].attrs['nseg']    

                fft1 = np.zeros(shape=(Nseg,Nfft//2),dtype=np.complex64)     
                fft1 = fft_ds_s[list_s[icompS*2+1]][:,:Nfft//2]+1j*fft_ds_s[list_s[icompS*2]][:,:Nfft//2]
                source_std = fft_ds_s[list_s[icompS*2+1]].attrs['std']
                    
                #-----------get the smoothed source spectrum for decon-----------
                temp = moving_ave(np.abs(fft1.reshape(fft1.size,)),10)  
                sfft1 = np.conj(fft1.reshape(fft1.size,))/temp**2
                sfft1 = sfft1.reshape(Nseg,Nfft//2)
                t2=time.time()
                n_source+=1
                time_source+=(t2-t1)

                #-----------loop III of each receiver B----------
                for ireceiver in range(isource+1,len(sources)):
                    
                    fft_ds_r = h5py.File(sources[ireceiver],'r')
                    list_r    = list(fft_ds_r.keys())

                    #-----loop IV of each component for receiver B------
                    for icompR in range(len(list_r)//2):

                        staR = list_r[icompR*2].split('_')[2]
                        netR = list_r[icompR*2].split('_')[1]
                        compR = list_r[icompR*2].split('_')[3]
  
                        t1=time.time()
                        fft2 = np.zeros(shape=(Nseg,Nfft//2),dtype=np.complex64)  
                        fft2 = fft_ds_r[list_r[icompR*2+1]][:,:Nfft//2]+1j*fft_ds_r[list_r[icompR*2]][:,:Nfft//2]
                        receiver_std = fft_ds_r[list_r[icompR*2+1]].attrs['std']
                        t2=time.time()
                        n_read+=1
                        time_read+=(t2-t1)

                        #---------- check the existence of earthquakes ----------
                        rec_ind = np.where(receiver_std < 10)[0]
                        sou_ind = np.where(source_std < 10)[0]

                        #-----note that Hi-net has a few mi-secs differences to Mesonet in terms starting time-----
                        bb,indx1,indx2=np.intersect1d(sou_ind,rec_ind,return_indices=True)
                        indx1=sou_ind[indx1]
                        indx2=rec_ind[indx2]

                        if (len(indx1)==0) | (len(indx2)==0):
                            continue

                        t1=time.time()
                        corr=optimized_correlate(sfft1[indx1,:],fft2[indx2,:])
                        t2=time.time()
                        n_mul+=1
                        time_mul+=(t2-t1)

                        t1=time.time()
                        corr = np.real(ifft(corr, Nfft, axis=0))
                        t2=time.time()
                        n_ifft+=1
                        time_ifft+=(t2-t1)

                        t1=time.time()
                        maxlag_sample = downsamp_freq*maxlag
                        indx = Nfft-maxlag_sample-1
                        ncorr = np.concatenate((corr[indx:-1],corr[:maxlag_sample+1]))
                        t2=time.time()
                        n_flip+=1
                        time_flip+=(t2-t1)

                        #---------------keep daily cross-correlation into a hdf5 file--------------
                        t1=time.time()
                        cc_aday_h5 = os.path.join(CCFDIR,iday+'.h5')
                        crap   = np.zeros(corr.shape,dtype=np.float32)

                        if not os.path.isfile(cc_aday_h5):
                            with h5py.File(cc_aday_h5,"w") as ccf_ds:
                                pass 

                        with h5py.File(cc_aday_h5,"a") as ccf_ds:
                            parameters = {'dt':dt, 'maxlag':maxlag}
                            crap = ncorr
                            path = '_'.join([staS,compS,staR,compR])
                            #print('work on ',n_write,' ',path)
                            tmp=ccf_ds.create_dataset(path,data=np.real(crap),shape=np.shape(crap),dtype=np.float32)
                            for key in parameters.keys():
                                tmp.attrs[key]=parameters[key]
                        t2=time.time()
                        n_write+=1
                        time_write+=t2-t1
        
        print("total in read: ",n_read,' ',time_read)
        print("total in mul: ",n_mul,' ',time_mul)
        print("total in write: ",n_write,' ',time_write)
        print("total in ifft: ",n_ifft,' ',time_ifft)
        print("total in source: ",n_source,' ',time_source)
        print("total in flipping: ",n_flip,' ',time_flip)

ttt1=time.time()
print('all step 2 takes '+str(ttt1-ttt0))

comm.barrier()
if rank == 0:
    sys.exit()
