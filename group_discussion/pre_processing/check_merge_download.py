import obspy
import os, glob
import noise_module
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client


# download the data 
time1 = '2012-07-13T02:00:00.000000Z'
time2 = '2012-07-15T00:00:00.000000Z'

client = Client('IRIS') 
tr = client.get_waveforms(network='TA', station='F05D', channel='BHZ', location='*', \
        starttime = time1, endtime=time2, attach_response=True)

#source = obspy.read('/Users/chengxin/Documents/Harvard/JAKARTA/JKA20miniSEED/JKA20131010073600.CHZ')
#nst = noise_module.preprocess_raw(source,20,True)

# process the data
ntr = tr.copy()
mtr = noise_module.preprocess_raw(ntr,20,True)
print('original trace:',tr)
print('new trace',mtr)

# plot the data
plt.subplot(211)
plt.plot(mtr[0].data,'r-')
indx = int(len(mtr[0].data)*(0.05/tr[0].stats.delta))
plt.subplot(212)
plt.plot(tr[0].data[:indx],'b-')
plt.show()