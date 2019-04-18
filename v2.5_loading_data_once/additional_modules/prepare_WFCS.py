import os
import obspy
import pyasdf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass
from obspy.signal.util import _npts2nfft

'''
get the spectrum of the stacked cross-correaltion function for 
Waveform Fitting of Cross Spectra analysis
'''
SDIR     = '/Users/chengxin/Documents/Harvard/Kanto_basin/Dispersion/test/KANTO'
h5file   = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK1/E.ABHM/E.ABHM_E.KSCM.h5'
pre_filt = [0.3,5]
tag      = 'Allstacked'
ccomp    = ['ZZ','RR','TT']
tmin     = 22
tmax     = 32

#------read the data and parameters------
with pyasdf.ASDFDataSet(h5file,mode='r') as ds:
    slist = ds.auxiliary_data.list()

    #----read the stacked data------
    if slist[0] == tag:
        dist = ds.auxiliary_data[tag][ccomp[0]].parameters['dist']
        dt   = ds.auxiliary_data[tag][ccomp[0]].parameters['dt']
        npts = ds.auxiliary_data[tag][ccomp[0]].parameters['lag']*int(1/dt)*2+1
        indx = npts//2
        indx1 = tmin*int(1/dt)
        indx2 = tmax*int(1/dt)
        npts  = indx2-indx1

        data = np.zeros((npts,len(ccomp)),dtype=np.float32)

        for icomp in range(len(ccomp)):
            try:
                tdata1 = ds.auxiliary_data[tag][ccomp[icomp]].data[indx+indx1:indx+indx2]
                tdata2 = ds.auxiliary_data[tag][ccomp[icomp]].data[indx-indx2+1:indx-indx1+1]
                data[:,icomp] = 0.5*tdata1+0.5*np.flip(tdata2)
                data[:,icomp] = bandpass(data[:,icomp],min(pre_filt),max(pre_filt),int(1/dt),corners=4, zerophase=True)
            except Exception:
                pass

#------get the spectrum now------
nfft = _npts2nfft(npts)
spec = np.zeros((nfft//2+1,len(ccomp)),dtype=np.complex64)
freq = np.fft.rfftfreq(n=nfft,d=dt)
for icomp in range(len(ccomp)):
    spec[:,icomp] = np.fft.rfft(data[:,icomp],n=nfft)

#---plot the spectrum----
plt.plot(freq,np.real(spec[:,0]),'r-')
plt.plot(freq,np.real(spec[:,1]),'g-')
plt.plot(freq,np.real(spec[:,2]),'b-')
plt.xlabel('frequency [HZ]');plt.ylabel('spectrum')
plt.legend(['ZZ','RR','TT'],loc='upper right')
plt.show()

print(dist,dt)
temp = os.path.join(SDIR,'spec_ZZ.dat')
dict = {'freq':freq,'spec_Z':np.real(spec[:,0]),'spec_R':np.real(spec[:,1]),'spec_T':np.real(spec[:,2])}
locs = pd.DataFrame(dict)

#----------write into a csv file---------------            
locs.to_csv(os.path.join(SDIR,'spec_ZZ.dat'),index=False)