import os
import glob
import pyasdf
import numpy as np 
from obspy.signal.filter import bandpass

'''
extract the PGV at each receiver for each source

note that the code uses both lags to account for the wave propagation
'''

#-----------absolute path-----------
rootpath = '/Users/chengxin/Documents/Harvard/Seattle/new_processing'
STACKDIR = os.path.join(rootpath,'STACK')

#--------find all station pairs----------
afiles = sorted(glob.glob(os.path.join(STACKDIR,'*/*.h5')))
nsta  = len(afiles)
ccomp = ['ZR','ZT','ZZ']
tags  = 'Allstacked'
freqmin = 0.3
freqmax = 0.5
tfile = open(os.path.join(STACKDIR,'PGV_Z_all.dat'),'w+')

#---------get some basic parameters---------
ds = pyasdf.ASDFDataSet(afiles[0],mode='r')
try:
    delta = ds.auxiliary_data[tags]['ZZ'].parameters['dt']
    lag   = ds.auxiliary_data[tags]['ZZ'].parameters['lag']
except Exception as error:
    print('Abort due to %s! cannot find delta and lag'%error)
del ds

#------index for data------
npts  = int(lag*2/delta)+1
spr   = int(1/delta)
indx  = npts//2
npair = nsta*2

#-----arrays to store data for plotting-----
'''
lonR = np.zeros(npair,dtype=np.float32)
latR = np.zeros(npair,dtype=np.float32)
lonS = np.zeros(npair,dtype=np.float32)
latS = np.zeros(npair,dtype=np.float32)
amp  = np.zeros((npair,len(ccomp)),dtype=np.float32)
'''
amp1 = np.zeros(len(ccomp),dtype=np.float32)
amp2 = np.zeros(len(ccomp),dtype=np.float32)

#-----loop through each station------
for ista in range(nsta):
    with pyasdf.ASDFDataSet(afiles[ista],mode='r') as ds:
        source = afiles[ista].split('/')[-1].split('_')[0]
        receiver = afiles[ista].split('_')[-1][:-3]
        slist = ds.auxiliary_data.list()

        #----check whether stacked data exists----
        if tags in slist:
            rlist = ds.auxiliary_data[tags].list()
            tlonS = ds.auxiliary_data[tags][rlist[0]].parameters['lonS']
            tlonR = ds.auxiliary_data[tags][rlist[0]].parameters['lonR']
            tlatS = ds.auxiliary_data[tags][rlist[0]].parameters['latS']
            tlatR = ds.auxiliary_data[tags][rlist[0]].parameters['latR']
            #lonS[ista] = tlonS;lonR[ista]=tlonR
            #latS[ista] = tlatS;latR[ista]=tlatR
            #lonS[ista+1] = tlonR;lonR[ista+1]=tlonS
            #latS[ista+1] = tlatR;latR[ista+1]=tlatS

            #------loop through all cross components-----
            for icomp in range(len(ccomp)):
                try:
                    tdata = ds.auxiliary_data[tags][ccomp[icomp]].data[:]
                    tdata = bandpass(tdata,freqmin,freqmax,spr,corners=4, zerophase=True)
                    amp1[icomp] = max(np.abs(tdata[indx:]))
                    amp2[icomp] = max(np.abs(tdata[:indx]))
                    #print(source,receiver,amp1[icomp],amp2[icomp])
                except Exception as error:
                    #print('Continue due to %s! cannot find delta and lag'%error)
                    pass
            
            if amp1[0]<1 and amp1[1]<1 and amp1[2]<1 and amp2[0]<1 and amp2[1]<1 and amp2[2]<1:
                tfile.write('%s %s %6.2f %6.2f %6.2f %6.2f %10.8f %10.8f %10.8f\n' % (source,receiver,tlonS,tlatS,\
                    tlonR,tlatR,amp1[0],amp1[1],amp1[2]))
                tfile.write('%s %s %6.2f %6.2f %6.2f %6.2f %10.8f %10.8f %10.8f\n' % (receiver,source,tlonS,tlatS,\
                    tlonR,tlatR,amp2[0],amp2[1],amp2[2]))
            elif amp1[0]>1 and amp1[1]>1 and amp1[2]>1 and amp2[0]>1 and amp2[1]>1 and amp2[2]>1:
                tfile.write('%s %s %6.2f %6.2f %6.2f %6.2f %10.8f %10.8f %10.8f\n' % (source,receiver,tlonS,tlatS,\
                    tlonR,tlatR,amp1[0]/1000000000,amp1[1]/1000000000,amp1[2]/1000000000))
                tfile.write('%s %s %6.2f %6.2f %6.2f %6.2f %10.8f %10.8f %10.8f\n' % (receiver,source,tlonS,tlatS,\
                    tlonR,tlatR,amp2[0]/1000000000,amp2[1]/1000000000,amp2[2]/1000000000))