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
rootpath = '/Users/chengxin/Documents/Harvard/Seattle'
STACKDIR = os.path.join(rootpath,'STACK')

#--------find all station pairs----------
afiles = sorted(glob.glob(os.path.join(STACKDIR,'*/*.h5')))
nsta  = len(afiles)
ccomp = ['TR','TT','TZ']
tags  = 'Allstacked'
freqmin = 0.5
freqmax = 1
tfile = open(os.path.join(STACKDIR,'PGV_T_all.dat'),'w+')

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
lonR = np.zeros(npair,dtype=np.float32)
latR = np.zeros(npair,dtype=np.float32)
lonS = np.zeros(npair,dtype=np.float32)
latS = np.zeros(npair,dtype=np.float32)
amp  = np.zeros((npair,len(ccomp)),dtype=np.float32)

#-----loop through each station------
for ista in range(nsta):
    with pyasdf.ASDFDataSet(afiles[ista],mode='r') as ds:
        slist = ds.auxiliary_data.list()

        #----check whether stacked data exists----
        if tags in slist:
            rlist = ds.auxiliary_data[tags].list()
            tlonS = ds.auxiliary_data[tags][rlist[0]].parameters['lonS']
            tlonR = ds.auxiliary_data[tags][rlist[0]].parameters['lonR']
            tlatS = ds.auxiliary_data[tags][rlist[0]].parameters['latS']
            tlatR = ds.auxiliary_data[tags][rlist[0]].parameters['latR']
            lonS[ista] = tlonS;lonR[ista]=tlonR
            latS[ista] = tlatS;latR[ista]=tlatR
            lonS[ista+1] = tlonR;lonR[ista+1]=tlonS
            latS[ista+1] = tlatR;latR[ista+1]=tlatS

            #------loop through all cross components-----
            for icomp in range(len(ccomp)):
                try:
                    tdata = ds.auxiliary_data[tags][ccomp[icomp]].data[:]
                    tdata = bandpass(tdata,freqmin,freqmax,spr,corners=4, zerophase=True)
                    amp[ista][icomp]   = max(tdata[indx:])
                    amp[ista+1][icomp] = max(tdata[:indx])
                except Exception as error:
                    #print('Continue due to %s! cannot find delta and lag'%error)
                    pass
            
            if amp[ista].all()<1 and amp[ista+1].all()<1 and amp[ista].all()>0 and amp[ista+1].all()>0:
                tfile.write('%6.2f %6.2f %6.2f %6.2f %10.8f %10.8f %10.8f\n' % (lonS[ista],latS[ista],\
                    lonR[ista],latR[ista],amp[ista][0],amp[ista][1],amp[ista][2]))
                tfile.write('%6.2f %6.2f %6.2f %6.2f %10.8f %10.8f %10.8f\n' % (lonS[ista+1],latS[ista+1],\
                    lonR[ista+1],latR[ista+1],amp[ista+1][0],amp[ista+1][1],amp[ista+1][2]))