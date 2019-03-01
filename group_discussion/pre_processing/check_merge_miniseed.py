import obspy
import os, glob
import noise_module
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client

tr = obspy.read('JKA20131010073600.CHZ')
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