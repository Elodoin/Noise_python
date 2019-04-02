import os
import sys
import time
import glob
import pyasdf
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI

'''
this script stacks the cross-correlation functions according to the parameter of stack_days, 
which afford exploration of the stability of the stacked ccfs for monitoring purpose

modified to make statistic analysis of the daily CCFs (amplitude) in order to select the ones
with abnormal large amplitude that would dominate the finally stacked waveforms (Apr/01/2019)
'''

t0=time.time()

#-------------absolute path of working directory-------------
rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
CCFDIR = os.path.join(rootpath,'CCF')
FFTDIR = os.path.join(rootpath,'FFT')
STACKDIR = os.path.join(rootpath,'STACK')

#------------make correction due to mis-orientation of instruments---------------
correction = True
if correction:
    corrfile = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/rotation/meso_angles.dat'
    locs     = pd.read_csv(corrfile)
    sta_list = list(locs.iloc[:]['station'])
    angles   = list(locs.iloc[:]['angle'])

#---control variables---
flag = False
do_rotation   = True
one_component = False
stack_days = 1
num_seg = 4

maxlag = 500
downsamp_freq=20
dt=1/downsamp_freq
pi = 3.141593

#----parameters to estimate SNR----
snr_parameters = {
    'freqmin':0.08,
    'freqmax':6,
    'steps': 15,
    'minvel': 0.5,
    'maxvel': 3.5,
    'noisewin':100}

#--------------9-component Green's tensor------------------
if not one_component:
    enz_components = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']

    #---!!!each element is corresponding to each other in 2 systems!!!----
    if do_rotation:
        rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
else:
    enz_components = ['ZZ']


#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    #----check the directory of STACK----
    if os.path.exists(STACKDIR)==False:
        os.mkdir(STACKDIR)

    #------a way to keep same station-pair orders as S2-------
    sfiles = sorted(glob.glob(os.path.join(FFTDIR,'*.h5')))
    sta = []
    for ifile in sfiles:
        temp = ifile.split('/')[-1]
        ista = temp.split('.')[1]
        inet = temp.split('.')[0]

        #--------make directory for storing stacked data------------
        if not os.path.exists(os.path.join(STACKDIR,inet+'.'+ista)):
            os.mkdir(os.path.join(STACKDIR,inet+'.'+ista))
        sta.append(inet+'.'+ista)

    #-------make station pairs based on list--------        
    pairs= noise_module.get_station_pairs(sta)
    ccfs = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))
    splits = len(pairs)
else:
    pairs,ccfs,splits=[None for _ in range(3)]

#---------broadcast-------------
pairs  = comm.bcast(pairs,root=0)
ccfs   = comm.bcast(ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size


#-----loop I: source station------
for ii in range(rank,splits+size-extra,size):
    
    if ii<splits:

        source,receiver = pairs[ii][0],pairs[ii][1]
        ndays  = len(ccfs)
        ncomp  = len(enz_components)
        nstack = ndays//stack_days

        stack_h5 = os.path.join(STACKDIR,source+'/'+source+'_'+receiver+'.h5')
        if os.isfile(stack_h5):
            print('file %s already exists! continue' % stack_h5.split('/')[-1])
            continue

        #------------parameters to store the CCFs--------
        corr   = np.zeros((ndays*ncomp,int(2*maxlag/dt)+1),dtype=np.float32)
        ampmax = np.zeros((ndays,ncomp),dtype=np.float32)
        ngood  = np.zeros((ndays,ncomp),dtype=np.int16)
        nflag  = np.zeros((ndays,ncomp),dtype=np.int16)

        #-----source information-----
        staS = source.split('.')[1]
        netS = source.split('.')[0]

        #-----receiver information------
        staR = receiver.split('.')[1]
        netR = receiver.split('.')[0]

        #-----loop through each day----
        for iday in range(ndays):
            if flag:
                print("source %s receiver %s at day %s" % (source,receiver,ccfs[iday].split('/')[-1]))

            fft_h5 = ccfs[iday]
            with pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r') as ds:

                #-------data types for source A--------
                data_types = ds.auxiliary_data.list()
                slist = np.array([s for s in data_types if staS in s])

                #---in case no such source-----
                if len(slist)==0:
                    print("no source %s at %s! continue" % (staS,fft_h5.split('/')[-1]))
                    continue

                for data_type in slist:
                    paths = ds.auxiliary_data[data_type].list()

                    #-------find the correspoinding receiver--------
                    rlist = np.array([r for r in paths if staR in r])
                    if len(rlist)==0:
                        print("no receiver %s for source %s at %s! continue" % (staR,staS,fft_h5.split('/')[-1]))
                        continue

                    if flag:
                        print('found the station-pair at %dth day' % iday)

                    #----------------copy the parameter information---------------
                    parameters  = ds.auxiliary_data[data_type][rlist[0]].parameters
                    ndt = parameters['dt']
                    nmaxlag = parameters['lag']

                    #------double check dt and maxlag-------
                    if ndt != dt or nmaxlag != maxlag:
                        raise ValueError('dt or maxlag parameters not correctly set in the beginning')

                    for path in rlist:

                        #--------cross component-------
                        if num_seg==0:
                            ccomp = data_type[-1]+path[-1]
                        elif num_seg < 10:
                            ccomp = data_type[-1]+path[-2]
                        else:
                            ccomp = data_type[-1]+path[-3]

                        #------put into 1D and 2D matrix----------
                        tindx  = enz_components.index(ccomp)
                        findx  = iday*ncomp+tindx
                        corr[findx] += ds.auxiliary_data[data_type][path].data[:]
                        ngood[iday,tindx] += ds.auxiliary_data[data_type][path].parameters['ngood']
                        nflag[iday,tindx] += 1

                        #------maximum amplitude of daly CCFs--------
                        if ampmax[iday,tindx] < np.max(corr[findx]):
                            ampmax[iday,tindx] = np.max(corr[findx])


        #--------make statistic analysis of CCFs at each component----------
        for icomp in range(ncomp):
            indx1 = np.where(nflag[:,icomp]>0)[0]
            indx2 = np.where(ampmax[indx1,icomp]<50*np.median(ampmax[indx1,icomp]))[0]     # remove the ones with too big amplitudes
            indx_gooday  = indx1[indx2]
            
            #----normalize daily CCFs by num_segs----
            for gooday in indx_gooday:
                findx = gooday*ncomp+icomp
                corr[findx] /= nflag[gooday,icomp]

        #------stack the CCFs--------
        for isday in range(nstack):

            indx1 = isday*stack_days
            indx2 = (indx1+stack_days)

            #--------start and end day information--------
            date_s = ccfs[indx1].split('/')[-1].split('.')[0]
            date_s = data_s.replace('_','')
            date_e = ccfs[indx2].split('/')[-1].split('.')[0]
            date_e = date_e.replace('_','')

            if flag:
                print('write the stacked data to ASDF between %s and %s' % (date_s,date_e))

            #------------------output path and file name----------------------
            crap   = np.zeros(int(2*maxlag/dt)+1,dtype=np.float32)

            #------in case it already exists------
            if not os.path.isfile(stack_h5):
                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
                    pass 

            with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:

                tcorr = np.zeros((ncomp,int(2*maxlag/dt)+1),dtype=np.float32)
                #-----loop through all E-N-Z components-----
                for ii in range(ncomp):
                    icomp = enz_components[ii]
                    indx  = xxx

                    #------do average-----
                    tcorr[ii] = np.mean(corr[indx],axis=0)

                    if flag:
                        print('estimate the SNR of component %s for %s_%s in E-N-Z system' % (enz_components[ii],source,receiver))
                    
                    #--------evaluate the SNR of the signal at target period range-------
                    #new_parameters = noise_module.get_SNR(corr[ii],snr_parameters,parameters)

                    #------save the time domain cross-correlation functions-----
                    data_type = 'F'+date_s+'T'+date_e
                    path = icomp
                    crap = tcorr[ii]
                    stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                #-------do rotation here if needed---------
                if do_rotation:
                    if flag:
                        print('doing matrix rotation now!')

                    #---read azi, baz info---
                    azi = parameters['azi']
                    baz = parameters['baz']

                    #---angles to be corrected----
                    if correction:
                        ind = sta_list.index(staS)
                        acorr = angles[ind]
                        ind = sta_list.index(staR)
                        bcorr = angles[ind]
                        cosa = np.cos((azi+acorr)*pi/180)
                        sina = np.sin((azi+acorr)*pi/180)
                        cosb = np.cos((baz+bcorr)*pi/180)
                        sinb = np.sin((baz+bcorr)*pi*180)
                    else:
                        cosa = np.cos(azi*pi/180)
                        sina = np.sin(azi*pi/180)
                        cosb = np.cos(baz*pi/180)
                        sinb = np.sin(baz*pi/180)

                    #------9 component tensor rotation 1-by-1------
                    for ii in range(len(rtz_components)):
                        
                        if ii==0:
                            crap = -cosb*tcorr[7]-sinb*tcorr[6]
                        elif ii==1:
                            crap = sinb*tcorr[7]-cosb*tcorr[6]
                        elif ii==2:
                            crap = tcorr[8]
                            continue
                        elif ii==3:
                            crap = -cosa*cosb*tcorr[4]-cosa*sinb*tcorr[3]-sina*cosb*tcorr[1]-sina*sinb*tcorr[0]
                        elif ii==4:
                            crap = cosa*sinb*tcorr[4]-cosa*cosb*tcorr[3]+sina*sinb*tcorr[1]-sina*cosb*tcorr[0]
                        elif ii==5:
                            crap = cosa*tcorr[5]+sina*tcorr[2]
                        elif ii==6:
                            crap = sina*cosb*tcorr[4]+sina*sinb*tcorr[3]-cosa*cosb*tcorr[1]-cosa*sinb*tcorr[0]
                        elif ii==7:
                            crap = -sina*sinb*tcorr[4]+sina*cosb*tcorr[3]+cosa*sinb*tcorr[1]-cosa*cosb*tcorr[0]
                        else:
                            crap = -sina*tcorr[5]+cosa*tcorr[2]

                        if flag:
                            print('estimate the SNR of component %s for %s_%s in R-T-Z system' % (rtz_components[ii],source,receiver))
                        #--------evaluate the SNR of the signal at target period range-------
                        #new_parameters = noise_module.get_SNR(crap,snr_parameters,parameters)

                        #------save the time domain cross-correlation functions-----
                        data_type = 'F'+date_s+'T'+date_e
                        path = rtz_components[ii]
                        stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

        #--------------now stack all of the days---------------
        tcorr = np.zeros((ncomp,int(2*maxlag/dt)+1),dtype=np.float32)
        with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
            for ii in range(ncomp):
                icomp = enz_components[ii]

                #------do average here--------
                if num2[ii]==0:
                    print('station-pair %s_%s no data at all for  %s: filling zero' % (source,receiver,icomp))
                else:
                    ncorr[ii] = ncorr[ii]/num2[ii]
                
                #--------evaluate the SNR of the signal at target period range-------
                #new_parameters = noise_module.get_SNR(ncorr[ii],snr_parameters,parameters)

                #------save the time domain cross-correlation functions-----
                data_type = 'Allstacked'
                path = icomp
                crap = tcorr[ii]
                stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=new_parameters)

            #----do rotation-----
            if do_rotation:

                #------9 component tensor rotation 1-by-1------
                for ii in range(len(rtz_components)):
                    
                    if ii==0:
                        crap = -cosb*tcorr[7]-sinb*tcorr[6]
                    elif ii==1:
                        crap = sinb*tcorr[7]-cosb*tcorr[6]
                    elif ii==2:
                        crap = tcorr[8]
                        continue
                    elif ii==3:
                        crap = -cosa*cosb*tcorr[4]-cosa*sinb*tcorr[3]-sina*cosb*tcorr[1]-sina*sinb*tcorr[0]
                    elif ii==4:
                        crap = cosa*sinb*tcorr[4]-cosa*cosb*tcorr[3]+sina*sinb*tcorr[1]-sina*cosb*tcorr[0]
                    elif ii==5:
                        crap = cosa*tcorr[5]+sina*tcorr[2]
                    elif ii==6:
                        crap = sina*cosb*tcorr[4]+sina*sinb*tcorr[3]-cosa*cosb*tcorr[1]-cosa*sinb*tcorr[0]
                    elif ii==7:
                        crap = -sina*sinb*tcorr[4]+sina*cosb*tcorr[3]+cosa*sinb*tcorr[1]-cosa*cosb*tcorr[0]
                    else:
                        crap = -sina*tcorr[5]+cosa*tcorr[2]

                    #--------evaluate the SNR of the signal at target period range-------
                    #new_parameters = noise_module.get_SNR(crap,snr_parameters,parameters)

                    #------save the time domain cross-correlation functions-----
                    data_type = 'Allstacked'
                    path = rtz_components[ii]
                    stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)


t1=time.time()
print('S3 takes '+str(t1-t0)+' s')

#---ready to exit---
comm.barrier()
if rank == 0:
    sys.exit()
