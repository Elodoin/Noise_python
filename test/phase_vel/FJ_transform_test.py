from scipy.fftpack.helper import next_fast_len
from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
from scipy import special
import pandas as pd
import numpy as np 
import pyasdf
import obspy
import scipy
import glob
import os

'''
this script stacks the cross-correlation functions resulted from 1-year daily stacking of Kanto data
and perform integral transformation from move-out plot to U(x,w) domain
'''

####################################
#### LOAD DATA & INITIALIZATION ####
####################################

# loop through each station
sta_file = '/Users/chengxin/Documents/Harvard/Kanto_basin/figures/stations/all_Mesonet/station.lst'
locs = pd.read_csv(sta_file)
sta  = locs['station']
lon  = locs['longitutde']
lat  = locs['latitude']
pi   = np.pi

# absolute path for stacked data
data_path = '/Volumes/Chengxin/KANTO/STACK_2012'

# absolute path for predicted dispersion from JIVSM
pre_file = '/Users/chengxin/Documents/Harvard/Kanto_basin/JIVSM_model/disp_prediction_CPS'

# loop through each source station
for ista in range(1):

    # all station-paris with sta[ista]
    allfiles  = glob.glob(os.path.join(data_path,'*/*AYHM*.h5'))
    nfiles    = len(allfiles)
    if not nfiles:
        continue

    # find closest grid point to load predicted dispersion curves
    tfiles = ['{0:s}/{1:5.1f}_{2:4.1f}.dat'.format(pre_file+'/pred_Rg',lon[ista],lat[ista]),\
        '{0:s}/{1:5.1f}_{2:4.1f}.dat'.format(pre_file+'/pred_Lg',lon[ista],lat[ista])]

    # basic parameters for loading data
    freqmin  = 0.08
    freqmax  = 1
    maxdist  = 12
    dist_inc = 0.5
    lag      = 50

    # different components
    dtype = 'Allstack0pws'
    ccomp  = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    pindx  = [1,0,1,1,0,1,0,2,0]
    onelag = False
    stack_method = dtype.split('0')[-1]

    # extract common variables
    try:
        ds    = pyasdf.ASDFDataSet(allfiles[0],mode='r')
        dt    = ds.auxiliary_data[dtype][ccomp[0]].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][ccomp[0]].parameters['maxlag']
    except Exception:
        print('cannot open %s to read'%allfiles[0])
        continue

    # make sure lag is within range
    if lag>maxlag:
        raise ValueError('required lag exceeds the limit')

    plt.figure(figsize=(12,9))
    for path in ccomp:

        # load prediction information (Rg/Lg)
        tindx = pindx[ccomp.index(path)]
        if tindx:
            disp_file = tfiles[tindx-1]
            disp_pred = pd.read_csv(disp_file)
            mode = disp_pred['mode']
            pper = disp_pred['period']
            vel  = disp_pred['c']
            indx1 = np.where(mode==1)[0]
            indx2 = np.where(mode==2)[0]
            per1  = pper[indx1]
            vel1  = vel[indx1]
            per2  = pper[indx2]
            vel2  = vel[indx2]


        ###################################
        ######## LOAD CCFS DATA ###########
        ###################################

        # initialize array
        anpts = int(maxlag/dt)+1
        tnpts = int(lag/dt)+1
        Nfft = int(next_fast_len(tnpts))
        dist = np.zeros(nfiles,dtype=np.float32)
        data = np.zeros(shape=(nfiles,tnpts),dtype=np.float32)

        # loop through each cc file
        icc=0
        for ii in range(nfiles):
            tmp = allfiles[ii]

            # load data into memory
            with pyasdf.ASDFDataSet(tmp,mode='r') as ds:
                try:
                    tdist = ds.auxiliary_data[dtype][path].parameters['dist']
                    tdata = ds.auxiliary_data[dtype][path].data[:]
                    if tdist > maxdist:continue
                except Exception:
                    print("continue! cannot read %s "%tmp)
                    continue  
            
            # assign data matrix
            dist[icc] = tdist
            data[icc] = tdata[anpts-1:anpts+tnpts-1]*0.5+np.flip(tdata[anpts-tnpts:anpts],axis=0)*0.5
            icc+=1

        # sort according to distance
        ntrace = int(np.round(np.max(dist)+0.51)/dist_inc)
        spec   = np.zeros(shape=(ntrace,Nfft//2),dtype=np.complex64)
        ndist  = np.zeros(ntrace,dtype=np.float32)
        flag   = np.zeros(ntrace,dtype=np.int16)
        for td in range(0,ntrace):
            ndist[td] = td*dist_inc
            tindx = np.where((dist>=ndist[td]-0.5*dist_inc)&(dist<ndist[td]+0.5*dist_inc))[0]
            if len(tindx):
                flag[td] = 1
                tdata = np.mean(data[tindx],axis=0)
                tdata = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                spec[td] = scipy.fftpack.fft(tdata,Nfft)[:Nfft//2]
        flag[0] = 1

        # remove ones without data
        indx = np.where(flag==1)[0]
        ndist = ndist[indx]
        spec  = spec[indx]
        ntrace = len(indx)

        #####################################
        ############ F-J TRANSFORM ##########
        #####################################

        # common variables
        p1 = 0.2
        p2 = 10
        pp = np.arange(p1,p2,0.02)
        c1 = 0.2
        c2 = 4.2
        cc = np.arange(c1,c2,0.05)
        nc = len(cc)

        # 2D dispersion array
        freqVec = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft // 2]
        indx = np.where((freqVec<=freqmax) & (freqVec>=freqmin))[0]
        freqVec = freqVec[indx]
        tNfft = len(freqVec)
        disp_array = np.zeros(shape=(nc,tNfft),dtype=np.complex64)

        # frequency-bessel transfom according to Wang et al., JGR 2019 [doi:10.1029/2018JB016595]
        for ifreq in range(tNfft):
            om = 2*pi*freqVec[ifreq]
            for ic in range(nc):
                tc = cc[ic]
                k = om/tc; vk = tc/om
                for idis in range(1,ntrace):
                    # make components for summation
                    dis1 = ndist[idis];dis0 = ndist[idis-1]
                    z1 = np.linspace(0,dis1,51); z0 = np.linspace(0,dis0,51)
                    B1 = np.sum(special.jv(0,z1)*(z1[1]-z1[0])); B0 = np.sum(special.jv(0,z0)*(z0[1]-z0[0]))

                    M = (k*dis1*special.jv(0,dis1*k)-B1)-(k*dis0*special.jv(0,dis0*k)-B0)
                    N = vk*spec[idis][ifreq]*dis1*special.jv(1,k*dis1)-vk*spec[idis][ifreq]*dis0*special.jv(1,k*dis0)
                    disp_array[ic][ifreq] +=  N+(spec[idis][ifreq]-spec[idis-1][ifreq])/(dis1-dis0)*vk**3*M
                        
        # do interpolation
        fc = scipy.interpolate.interp2d(1/freqVec,cc,np.abs(disp_array))
        disp_new = fc(pp,cc)

        # do normalization for each frequency
        for ii in range(len(pp)):
            disp_new[:,ii] /= np.max(disp_new[:,ii])

        #####################################
        ############ PLOTTING ###############
        #####################################

        #---plot 2D dispersion image-----
        tmpt = '33'+str(ccomp.index(path)+1)
        plt.subplot(tmpt)

        extent = [pp[0],pp[-1],cc[0],cc[-1]]
        plt.imshow(np.abs(disp_new),cmap='jet',interpolation='bicubic',extent=extent,origin='lower',aspect='auto')
        if ccomp.index(path) == 1:
            plt.title('%s with %d pairs in %d km'%(sta[ista],nfiles,maxdist))
        plt.xlabel('period [s]')
        plt.ylabel('phase velocity [km/s]')
        plt.colorbar()
        font = {'family': 'serif', 'color':  'black', 'weight': 'bold','size': 16}
        plt.text(max(pp)*0.8,cc[-1]*0.8,path,fontdict=font)
        plt.tight_layout()

    # output figure to pdf files
    outfname = data_path+'/figures/frequency_bessel/'+str(sta[ista])+'_'+stack_method+'_'+str(maxdist)+'km.pdf'
    print(outfname)
    plt.savefig(outfname, format='pdf', dpi=300)
    plt.close()