import os
import glob
import scipy
import pycwt
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack.helper import next_fast_len

'''
this function uses cwt to track group wave energy in order to compare those from FTAN method

by Chengxin Jiang
'''

# input file info
sfile = '/Volumes/Chengxin/KANTO/STACK_2012/E.ABHM/E.ABHM_E.USCM.h5'
tmp   = sfile.split('/')[-1].split('_')
spair = tmp[0]+'_'+tmp[1][:-3]
pi    = np.pi
plot_wct = True

# data type and cross-component
dtype = 'Allstack0pws'
ccomp = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
post1 = [0,0,0,1,1,1,2,2,2]
post2 = [0,1,2,0,1,2,0,1,2]

# freq bands
fmin = 0.08
fmax = 2
nfin = 50
omb  = 2*pi*fmin
ome  = 2*pi*fmax
nper = np.arange(int(1/fmax),int(1/fmin),0.02)

# set time window
vmin = 0.2
vmax = 2.0
vel  = np.arange(vmin,vmax,0.02)

# load data information
with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
    try:
        maxlag = ds.auxiliary_data[dtype]['ZZ'].parameters['maxlag']
        dist   = ds.auxiliary_data[dtype]['ZZ'].parameters['dist']
        dt = ds.auxiliary_data[dtype]['ZZ'].parameters['dt']
    except Exception as e:
        raise ValueError(e)

# should be dist/freq dependent
alpha = 5*20*np.sqrt(dist/1000)

fig,ax = plt.subplots(3,3,figsize=(12,9), sharex=True)
# load cross-correlation functions
################################
for comp in ccomp:
    cindx = ccomp.index(comp)
    pos1  = post1[cindx]
    pos2  = post2[cindx]

    with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
        try:
            tdata = ds.auxiliary_data[dtype][comp].data[:]
        except Exception as e:
            raise ValueError(e)

    # stack positive and negative lags
    npts = int(1/dt)*2*maxlag+1
    indx = npts//2
    data = 0.5*tdata[indx:]+0.5*np.flip(tdata[:indx+1],axis=0)

    # trim the data according to vel window
    pt1 = int(dist/vmax/dt)
    pt2 = int(dist/vmin/dt)
    if pt1 == 0:
        pt1 = 10
    if pt2>(npts//2):
        pt2 = npts//2
    indx = np.arange(pt1,pt2)
    tvec = indx*dt
    npts = len(tvec)
    data = data[indx]

    ############################################
    ###### make freq-time analysis (FTAN) ######
    ############################################
    Nfft = int(next_fast_len(npts))
    dom  = 2*np.pi/Nfft/dt
    step = (np.log(omb)-np.log(ome))/(nfin-1)
    spec = scipy.fftpack.fft(np.array(data),Nfft)

    #-------initialize arrays----------
    om  = np.zeros(nfin,dtype=np.float32)
    per = np.zeros(nfin,dtype=np.float32)
    amp   = np.zeros((npts,nfin),dtype=np.float32)

    # loop through each freq 
    for ii in range(nfin):
        om[ii] = np.exp(np.log(ome)+ii*step)
        per[ii]= 2*pi/om[ii]

        # initialize the array
        fs = np.zeros(Nfft,dtype=np.complex64)
        b  = np.zeros(Nfft,dtype=np.float32)
        fs1 = np.zeros(Nfft,dtype=np.complex64)
        fs2 = np.zeros(Nfft,dtype=np.complex64)

        # filtering in freq domain
        for jj in range(Nfft//2):
            fs[jj] = np.complex(0,0)
            b[jj]  = 0
            ome = jj*dom
            om2 = -(ome-om[ii])*(ome-om[ii])*alpha/om[ii]/om[ii]
            b[jj]  = np.exp(om2)
            fs[jj] = spec[jj]*b[jj] 

        # hilbert transform (in freq domain) to get envelope
        fs[:Nfft//2]  = 2*fs[:Nfft//2]
        fs[Nfft//2+1:] = np.complex(0,0)
        tmp1 = scipy.fftpack.ifft(fs,Nfft)[:npts]
        amp[:,ii]  = np.abs(tmp1)

    # interpolation to grids of freq-vel domain
    ######################################
    fc = scipy.interpolate.interp2d(per,dist/tvec,amp)
    amp_new = fc(nper,vel)

    # do normalization for each frequency
    for ii in range(len(nper)):
        amp_new[:,ii] /= np.max(amp_new[:,ii])

    # plot wavelet spectrum
    ##########################
    if plot_wct:
        #im=ax[pos1,pos2].imshow(rcwt_new,cmap='jet',extent=[tvec[0],tvec[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
        im=ax[pos1,pos2].imshow(amp_new,cmap='jet',extent=[nper[0],nper[-1],vel[0],vel[-1]],aspect='auto',origin='lower')
        ax[pos1,pos2].set_xlabel('Period [s]')
        ax[pos1,pos2].set_ylabel('U [km/s]')
        if cindx==1:
            ax[pos1,pos2].set_title('%s %5.2fkm linear'%(spair,dist))
        #Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),np.ceil(np.log2(period.max())))
        #ax[pos1,pos2].set_yticks(np.log2(Yticks))
        #ax[pos1,pos2].set_yticklabels(Yticks)
        ax[pos1,pos2].xaxis.set_ticks_position('bottom')
        cbar=fig.colorbar(im,ax=ax[pos1,pos2])
        font = {'family': 'serif', 'color':  'red', 'weight': 'bold','size': 16}
        ax[pos1,pos2].text(int(nper[-1]*0.9),vel[-1]-0.2,comp,fontdict=font)

fig.tight_layout()
fig.show()