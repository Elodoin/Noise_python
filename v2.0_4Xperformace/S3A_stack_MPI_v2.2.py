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
CCFDIR = os.path.join(rootpath,'CCF')
FFTDIR = os.path.join(rootpath,'FFT')
STACKDIR = os.path.join(rootpath,'STACK')

#---common variables---
stack_days = 2
flag = True
one_component = False
maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq

if not one_component:
    all_components = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']
else:
    all_components = ['ZZ']

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

        #------keep a track of the starting date-----
        date_s = ccfs[0].split('/')[-1].split('.')[0]
        date_s = date_s.replace('_','')

        #-----loop through each day----
        for iday in range(len(ccfs)):
            if flag:
                print("source %s receiver %s at day %s" % (source,receiver,ccfs[iday].split('/')[-1]))

            fft_h5 = ccfs[iday]
            with pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r') as ds:

                #-------data types for source A--------
                data_types = ds.auxiliary_data.list()
                slist = np.array([s for s in data_types if staS in s])

                #---in case no such source-----
                if len(slist)==0:
                    print("no source %s at %dth day! continue" % (staS,iday))
                    continue

                for data_type in slist:
                    paths = ds.auxiliary_data[data_type].list()

                    #-------find the correspoinding receiver--------
                    rlist = np.array([r for r in paths if staR in r])
                    if len(rlist)==0:
                        print("no receiver %s for source %s at %dth day! continue" % (staR,staS,iday))
                        continue

                    if flag:
                        print('found the station-pair at %dth day' % iday)

                    #----------------copy the parameter information---------------
                    parameters  = ds.auxiliary_data[data_type][paths[0]].parameters
                    for path in paths:

                        #--------cross component-------
                        ccomp = data_type[-1]+path[-1]

                        #------put into a 2D matrix----------
                        tindx  = all_components.index(ccomp)
                        corr[tindx,:] += ds.auxiliary_data[data_type][path].data[:]
                        num1[tindx]   += 1

            #------stack every n(10) day or what is left-------
            if (iday>0 and iday%stack_days==0) or iday==len(ccfs):

                #------keep a track of ending date for stacking------
                date_e = ccfs[iday-1].split('/')[-1].split('.')[0]
                date_e = date_e.replace('_','')

                if flag:
                    print('write the stacked data to ASDF between %s and %s' % (date_s,date_e))

                #------------------output path and file name----------------------
                stack_h5 = os.path.join(STACKDIR,source+'/'+source+'_'+receiver+'.h5')
                crap   = np.zeros(int(2*maxlag/dt)+1,dtype=np.float32)

                #------in case it already exist------
                if not os.path.isfile(stack_h5):
                    with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
                        pass 

                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
                    for ii in range(len(all_components)):
                        icomp = all_components[ii]

                        #------do average-----
                        if num1[ii]==0:
                            print('station-pair %s_%s no data in %d days for components %s: filling zero' % (source,receiver,stack_days,icomp))
                        else:
                            corr[ii,:] = corr[ii,:]/num1[ii]
                            ncorr[ii,:] += corr[ii,:]
                            num2[ii]    += 1

                        #------save the time domain cross-correlation functions-----
                        data_type = 'F'+date_s+'T'+date_e
                        path = icomp
                        crap = corr[ii,:]
                        stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                        #----reset----
                        corr[ii,:] = 0
                        num1[ii]   = 0
                        
                date_s = ccfs[iday].split('/')[-1].split('.')[0]
                date_s = date_s.replace('_','')

        #--------------now stack all of the days---------------
        with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
            for ii in range(len(all_components)):
                icomp = all_components[ii]

                #------do average here--------
                if num2[ii]==0:
                    print('station-pair %s_%s no data in at all for components %s: filling zero' % (source,receiver,icomp))
                else:
                    ncorr[ii,:] = ncorr[ii,:]/num2[ii]

                #------save the time domain cross-correlation functions-----
                data_type = 'Allstacked'
                path = icomp
                crap = ncorr[ii,:]
                stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)


t1=time.time()
print('S3 takes '+str(t1-t0)+' s')

#---ready to exit---
comm.barrier()
if rank == 0:
    sys.exit()