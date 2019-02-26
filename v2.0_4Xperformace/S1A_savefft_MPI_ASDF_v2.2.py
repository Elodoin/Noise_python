import os
import sys
import glob
from datetime import datetime
import numpy as np
import scipy
from scipy.fftpack.helper import next_fast_len
from obspy.signal.filter import bandpass
import obspy
import matplotlib.pyplot as plt
import noise_module
import time
import pyasdf
import pandas as pd
from mpi4py import MPI

'''
this script pre-processs the noise data for each single station using the parameters given below 
and stored the whitened and nomalized fft trace for each station in ASDF format. 
- C.Jiang, T.Clements, M.Denolle (Nov.09.2018)

updated to handle SAC, MiniSeed and ASDF formate inputs (Feb.20.2019). 
'''

t00=time.time()

#------absolute path parameters-------
rootpath  = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO'
FFTDIR = os.path.join(rootpath,'FFT/test')
event = os.path.join(rootpath,'data_download/*.h5')
resp_dir = os.path.join(rootpath,'DATA/resp')

#------input file types: make sure it is asdf--------
asdf  = True
tag   = 'raw_recordings'

#-----some control parameters------
prepro=False                #preprocess the data?
to_whiten=False             #whiten the spectrum?
time_norm=False             #normalize in time?
flag=False                  #print intermediate variables and computing time

#-----assume response has been removed in downloading process-----
rm_resp_spectrum=False       #remove response using spectrum?
rm_resp_inv=False           #remove response using inventory

pre_filt=[0.04,0.05,4,6]
downsamp_freq=20
dt=1/downsamp_freq
cc_len=3600
step=1800
freqmin=0.05  
freqmax=4
norm_type='running_mean'


#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

#-------make sure only one input type is selected--------
if not asdf:
    raise ValueError('make sure the input files are in ASDF format!')

if rank == 0:
    #----define common variables----
    tdir = sorted(glob.glob(event))
    splits = len(tdir)
else:
    splits,tdir = [None for _ in range(2)]

#-----broadcast the variables------
splits = comm.bcast(splits,root=0)
tdir = comm.bcast(tdir,root=0)
extra = splits % size

    #--------MPI: loop through each station--------
for ista in range (rank,splits+size-extra,size):
    #time.sleep(rank/1000)

    if ista<splits:
        t10=time.time()
        #----loop through each day on each core----
        for jj in range (len(tdir)):
            
            #---------think about reading mini-seed files here---------
            with pyasdf.ASDFDataSet(tdir[jj],mode='r') as ds:
                temp = ds.waveforms.list()
                station = temp[0].split('.')[1]
                network = temp[0].split('.')[0]

                if flag:
                    print("working on station %s " % station)

                #------get traces and station inventory------
                tfiles = ds.waveforms[temp[0]][tag]
                inv1 = ds.waveforms[temp[0]]['StationXML']
                if (not inv1[0][0].latitude) or (not inv1[0][0].longitude):
                    raise ValueError('no station information in inventory! double check!')

                #---------construct a pd structure for fft_parameter functions later----------
                locs = pd.DataFrame([[inv1[0][0].latitude,inv1[0][0].longitude,inv1[0][0].elevation]],\
                    columns=['latitude','longitude','elevation'])

                #----loop through each channel----
                for source in tfiles:
                    comp = source.stats.channel
                    
                    if flag:
                        print("working on trace %s" % source)
                    
                    if prepro:
                        t0=time.time()
                        source = noise_module.process_raw(source, downsamp_freq)
                        t1=time.time()
                        if flag:
                            print("prepro takes %f s" % (t1-t0))
                    
                    #----remove instrument response using extracted files-----
                    if rm_resp_spectrum:
                        t0=time.time()
                        if not os.path.isdir(resp_dir):
                            raise IOError ('repsonse spectrum folder %s not exist' % resp_dir)

                        if source.stats.npts!=downsamp_freq*24*cc_len:
                            print('Next! Extraced response file not match SAC file length')
                            continue

                        source = noise_module.resp_spectrum(source,resp_dir,downsamp_freq,station)
                        if not source:
                            continue
                        source.data=bandpass(source.data,freqmin,freqmax,downsamp_freq,corners=4,zerophase=False)
                        t1=time.time()
                        if flag:
                            print("remove instrument takes %f s" % (t1-t0))
                    
                    #-----using inventory------
                    elif rm_resp_inv:
                        source.data=noise_module.remove_resp(source.data,source.stats,inv1)

                    #----------variables to define days with earthquakes----------
                    all_madS = noise_module.mad(source.data)
                    all_stdS = np.std(source.data)
                    if all_madS==0 or all_stdS==0:
                        print("continue! madS or stdS equeals to 0 for %s" % source)
                        continue

                    trace_madS = []
                    trace_stdS = []
                    nonzeroS = []
                    nptsS = []
                    source_slice = obspy.Stream()

                    #--------break a continous recording into pieces----------
                    t0=time.time()
                    for ii,win in enumerate(source.slide(window_length=cc_len, step=step)):
                        win.detrend(type="constant")
                        win.detrend(type="linear")
                        trace_madS.append(np.max(np.abs(win.data))/all_madS)
                        trace_stdS.append(np.max(np.abs(win.data))/all_stdS)
                        nonzeroS.append(np.count_nonzero(win.data)/win.stats.npts)
                        nptsS.append(win.stats.npts)
                        win.taper(max_percentage=0.05,max_length=20)
                        source_slice.append(win)
                    del source
                    
                    t1=time.time()
                    if flag:
                        print("breaking records takes %f s"%(t1-t0))

                    if len(source_slice) == 0:
                        print("No traces for %s " % source)
                        continue

                    source_params= np.vstack([trace_madS,trace_stdS,nonzeroS]).T
                    del trace_madS, trace_stdS, nonzeroS

                    #---------seems un-necesary for data already pre-processed with same length (zero-padding)-------
                    N = len(source_slice)
                    NtS = np.max(nptsS)
                    dataS_t= np.zeros(shape=(N,2))
                    dataS = np.zeros(shape=(N,NtS),dtype=np.float32)
                    for ii,trace in enumerate(source_slice):
                        dataS_t[ii,0]= source_slice[ii].stats.starttime-obspy.UTCDateTime(1970,1,1)# convert to dataframe
                        dataS_t[ii,1]= source_slice[ii].stats.endtime -obspy.UTCDateTime(1970,1,1)# convert to dataframe
                        dataS[ii,0:nptsS[ii]] = trace.data
                        if ii==0:
                            dataS_stats=trace.stats

                    #------check the dimension of the dataS-------
                    if dataS.ndim == 1:
                        axis = 0
                    elif dataS.ndim == 2:
                        axis = 1

                    Nfft = int(next_fast_len(int(dataS.shape[axis])))

                    #-----to whiten or not------
                    if to_whiten:
                        t0=time.time()
                        source_white = noise_module.whiten(dataS,dt,freqmin,freqmax)
                        t1=time.time()
                        if flag:
                            print("spectral whitening takes %f s"%(t1-t0))
                    else:
                        source_white = scipy.fftpack.fft(dataS, Nfft, axis=axis)

                    #------to normalize in time or not------
                    if time_norm:
                        t0=time.time()   
                        white = np.real(scipy.fftpack.ifft(source_white, Nfft, axis=axis)) #/ Nt

                        if norm_type == 'one_bit': 
                            white = np.sign(white)
                        elif norm_type == 'running_mean':
                            white = noise_module.running_abs_mean(white,int(1 / freqmin / 2))
                        source_white = scipy.fftpack.fft(white, Nfft, axis=axis)
                        del white
                        t1=time.time()
                        if flag:
                            print("temporal normalization takes %f s"%(t1-t0))

                    #-------------save FFTs as HDF5 files-----------------
                    crap=np.zeros(shape=(N,Nfft//2),dtype=np.complex64)
                    fft_h5 = os.path.join(FFTDIR,network+'.'+station+'.h5')

                    if not os.path.isfile(fft_h5):
                        with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as ds:
                            pass # create pyasdf file 
            
                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                        parameters = noise_module.fft_parameters(dt,cc_len,dataS_stats,dataS_t,source_params, \
                            locs.iloc[0],comp,Nfft,N)
                        
                        savedate = '{0:04d}_{1:02d}_{2:02d}'.format(dataS_stats.starttime.year,\
                            dataS_stats.starttime.month,dataS_stats.starttime.day)
                        path = savedate

                        data_type = str(comp)
                        fft_ds.add_stationxml(inv1)
                        crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                        fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                    del fft_ds, crap, parameters, source_slice, source_white, dataS, dataS_stats, dataS_t, source_params            

                del tfiles
        t11=time.time()
        print('it takes '+str(t11-t10)+' s to process one station in step 1')

t01=time.time()
print('step1 takes '+str(t01-t00)+' s')

comm.barrier()
if rank == 0:
    sys.exit()