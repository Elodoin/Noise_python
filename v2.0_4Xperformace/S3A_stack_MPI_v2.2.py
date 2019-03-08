import os
import glob
import sys
import time
import noise_module
import numpy as np
import pyasdf
from mpi4py import MPI

t0=time.time()

#-------------absolute path of working directory-------------
rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
CCFDIR = os.path.join(rootpath,'CCF/test')
FFTDIR = os.path.join(rootpath,'FFT')
STACKDIR = os.path.join(rootpath,'STACK')

#---common variables---
stack_days = 2
flag = True
maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq
all_components = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    #----check the directory of STACK----
    if os.path.exists(STACKDIR)==False:
        os.mkdir(STACKDIR)

    #------keep same order as S2--------
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


#-----loop I: source stations------
for ii in range(rank,splits+size-extra,size):
    
    if ii<splits:

        source,receiver = pairs[ii][0],pairs[ii][1]

        #----corr records every 10 days; ncorr records all days----
        corr  = np.zeros((len(all_components),int(2*maxlag/dt)+1),dtype=np.float32)
        ncorr = np.zeros((len(all_components),int(2*maxlag/dt)+1),dtype=np.float32)
        num1  = np.zeros(len(all_components),dtype=np.int16)
        num2  = np.zeros(len(all_components),dtype=np.int16)

        #-----source information-----
        staS = source.split('.')[1]
        netS = source.split('.')[0]

        #-----receiver information------
        staR = receiver.split('.')[1]
        netR = receiver.split('.')[0]
        data_type = netS+'s'+staS+'s'+netR+'s'+staR

        date_s = ccfs[0].split('/')[-1].split('.')[0]

        #-------loop through each day-------
        for iday in range(len(ccfs)):
            if flag:
                print("work on source %s receiver %s at day %s" % (source, receiver,ccfs[iday]))

            fft_h5 = ccfs[iday]
            with pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r') as ds:

                #-------data types for source A--------
                data_types = ds.auxiliary_data.list()
                if data_type in data_types:
                    if flag:
                        print('found the station-pair at %dth day' % iday)

                    cross_comps = ds.auxiliary_data[data_type].list()
                    parameters  = ds.auxiliary_data[data_type][cross_comps[0]].parameters
                
                    #----loop through all cross components-----
                    for icomp in cross_comps:
                        comp1 = icomp.split('_')[0]
                        comp2 = icomp.split('_')[1]

                        #-----in case Z direction is labeled as U-----
                        if comp1[-1]=='U':
                            if comp2[-1]=='E':
                                ccomp = 'ZE'
                            elif comp2[-1]=='N':
                                ccomp = 'ZN'
                            else:
                                ccomp = 'ZZ'
                        else:
                            if comp2[-1]=='U':
                                ccomp = comp1[-1]+'Z'
                            else:
                                ccomp = comp1[-1]+comp2[-1]
                        
                        if flag:
                            print("cross component of %s" % ccomp)

                        #------put into a 2D matrix----------
                        tindx  = all_components.index(ccomp)
                        corr[tindx,:] += ds.auxiliary_data[data_type][icomp].data[:]
                        num1[tindx]   += 1

            #------stack every n(10) day data-----
            if iday>0 and iday%stack_days==0:
                if flag:
                    print("write the %d days' stacking into ADSF files" % stack_days)
                date_e = ccfs[iday].split('/')[-1].split('.')[0]

                stack_h5 = os.path.join(STACKDIR,source+'/'+source+'_'+receiver+'.h5')
                crap   = np.zeros(int(2*maxlag/dt)+1,dtype=np.float32)

                #-------in case it already exist------
                if not os.path.isfile(stack_h5):
                    with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
                        pass 

                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
                    for ii in range(len(all_components)):
                        icomp = all_components[ii]

                        #------do average here-----
                        if num1[ii]==0:
                            print('station-pair %s_%s no data in %d days for components %s: filling zero' % (source,receiver,stack_days,icomp))
                        else:
                            corr[ii,:] = corr[ii,:]/num1[ii]
                            ncorr[ii,:] += corr[ii,:]
                            num2[ii]    += 1

                        #------save the time domain cross-correlation functions-----
                        data_type = str(date_s.replace('_',''))+'T'+str(date_e.replace('_',''))
                        path = icomp
                        crap = corr[ii,:]
                        stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                        #----reset----
                        corr[ii,:] = 0
                        num1[ii]   = 0
                        
                date_s = ccfs[iday+1].split('/')[-1].split('.')[0]

        #------------now for the stacking of all days------------
        with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
            for ii in range(len(all_components)):
                icomp = all_components[ii]

                #------do average here--------
                if num2[ii]==0:
                    print('station-pair %s_%s no data in at all for components %s: filling zero' % (source,receiver,icomp))
                else:
                    ncorr[ii,:] = ncorr[ii,:]/num2[ii]

                #------save the time domain cross-correlation functions-----
                data_type = 'all_stacked'
                path = icomp
                crap = ncorr[ii,:]
                stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)


t1=time.time()
print('S3 takes '+str(t1-t0)+' s')

#---ready to exit---
comm.barrier()
if rank == 0:
    sys.exit()