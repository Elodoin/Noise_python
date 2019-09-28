import os
import glob
import scipy
import pycwt
import pyasdf
import numpy as np
import matplotlib.pyplot as plt

'''
this function uses cwt to track group wave energy. 
to be compared with FTAN method

by Chengxin Jiang @Sep/05/2019
'''

# input file info
sfile = '/Volumes/Chengxin/KANTO/STACK_2012/E.ABHM/E.ABHM_E.USCM.h5'
tmp   = sfile.split('/')[-1].split('_')
spair = tmp[0]+'_'+tmp[1][:-3]
plot_wct = True

# data type and cross-component
dtype = 'Allstack0linear'
ccomp = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
post1 = [0,0,0,1,1,1,2,2,2]
post2 = [0,1,2,0,1,2,0,1,2]

# freq bands
fmin = 0.08
fmax = 2
per  = np.arange(int(1/fmax),int(1/fmin),0.02)

# set time window
vmin = 0.2
vmax = 2.0
vel  = np.arange(vmin,vmax,0.02)

# basic parameters for wavelet transform
dj=1/12
s0=-1
J=-1
wvn='morlet'

# load data information
with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
    try:
        maxlag = ds.auxiliary_data[dtype]['ZZ'].parameters['maxlag']
        dist   = ds.auxiliary_data[dtype]['ZZ'].parameters['dist']
        dt = ds.auxiliary_data[dtype]['ZZ'].parameters['dt']
    except Exception as e:
        raise ValueError(e)

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
    data = data[indx]

    # wavelet transformation
    ################################
    cwt, sj, freq, coi, _, _ = pycwt.cwt(data, dt, dj, s0, J, wvn)

    # do filtering here
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: frequency out of limits!')
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]

    period = 1/freq
    rcwt,pcwt = np.abs(cwt)**2,np.real(cwt)

    # interpolation to grids of freq-vel
    ######################################
    fc = scipy.interpolate.interp2d(dist/tvec,period,rcwt)
    rcwt_new = fc(vel,per)

    # do normalization for each frequency
    for ii in range(len(per)):
        rcwt_new[ii] /= np.max(rcwt_new[ii])

    # plot wavelet spectrum
    ##########################
    if plot_wct:
        #im=ax[pos1,pos2].imshow(rcwt_new,cmap='jet',extent=[tvec[0],tvec[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
        im=ax[pos1,pos2].imshow(np.transpose(rcwt_new),cmap='jet',extent=[per[0],per[-1],vel[0],vel[-1]],aspect='auto',origin='lower')
        ax[pos1,pos2].set_xlabel('Period [s]')
        ax[pos1,pos2].set_ylabel('U [km/s]')
        if cindx==1:
            ax[pos1,pos2].set_title('%s %5.2fkm linear'%(spair,dist))
        #Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),np.ceil(np.log2(period.max())))
        #ax[pos1,pos2].set_yticks(np.log2(Yticks))
        #ax[pos1,pos2].set_yticklabels(Yticks)
        ax[pos1,pos2].xaxis.set_ticks_position('bottom')
        #ax[0,0].fill(np.concatenate([tvec, tvec[-1:]+dt, tvec[-1:]+dt, tvec[:1]-dt, tvec[:1]-dt]), \
        #    np.concatenate([np.log2(coi1), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]), \
        #    'k', alpha=0.3, hatch='x')
        cbar=fig.colorbar(im,ax=ax[pos1,pos2])
        font = {'family': 'serif', 'color':  'red', 'weight': 'bold','size': 16}
        ax[pos1,pos2].text(int(per[-1]*0.9),vel[-1]-0.2,comp,fontdict=font)
fig.tight_layout()
fig.show()