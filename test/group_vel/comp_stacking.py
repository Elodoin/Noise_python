import os
import glob
import scipy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

'''
this function compares the stacked waveforms from PWS and linear
on the 9 cross components respectively

by Chengxin Jiang
'''

# input file info
sfile = '/Volumes/Chengxin/KANTO/STACK_2012/E.ABHM/E.ABHM_E.USCM.h5'
tmp   = sfile.split('/')[-1].split('_')
spair = tmp[0]+'_'+tmp[1][:-3]
plot_wct = True

# data type and cross-component
dtype1 = 'Allstack0linear'
dtype2 = 'Allstack0pws'
ccomp = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']

# freq bands
fmin = 0.08
fmax = 2

# load waveform data
with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
    try:
        maxlag = ds.auxiliary_data[dtype1]['ZZ'].parameters['maxlag']
        dist   = ds.auxiliary_data[dtype1]['ZZ'].parameters['dist']
        dt = ds.auxiliary_data[dtype1]['ZZ'].parameters['dt']
    except Exception as e:
        raise ValueError(e)

# define signal window
npts = int(1/dt)*2*maxlag+1
indx = npts//2
lag  = 100