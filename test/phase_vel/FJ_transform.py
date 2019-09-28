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
this script finds all station-pairs within a small region and use the cross-correlation functions 
for all these pairs resulted from 1-year daily stacking of Kanto data to perform F-J transformation 
according to Wang et al., JGR 2019 [doi:10.1029/2018JB016595]

by Chengxin Jiang (chengxin_jiang@fas.harvard.edu) @ Aug/2019
'''

#######################################
########## PARAMETER SECTION ##########
#######################################

# absolute path for stacked data
data_path = '/Volumes/Chengxin/KANTO/STACK_2012'

# absolute path for predicted dispersion from JIVSM
pre_file = '/Users/chengxin/Documents/Harvard/Kanto_basin/JIVSM_model/disp_prediction_CPS'

# station list
sta_file = '/Users/chengxin/Documents/Harvard/Kanto_basin/figures/stations/all_Mesonet/station.lst'
locs = pd.read_csv(sta_file)
net  = locs['network']
sta  = locs['station']
lon  = locs['longitutde']
lat  = locs['latitude']
pi   = np.pi

# parameters for data reading
dtype = 'Allstack0pws'
ccomp  = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
pindx  = [1,0,1,1,0,1,0,2,0]
onelag = False
norm   = False
bdpass = False
stack_method = dtype.split('0')[-1]

# basic parameters for loading data
freqmin  = 0.125
freqmax  = 2
maxdist  = 10
dist_inc = 0.2
maxnpair = 20
lag      = 50

######################################
######## loop through station ########
######################################
for ista in range(3,152):
    staS = sta[ista]

    # find all stations within maxdist
    sta_list = []
    for ii in range(len(sta)):
        dist,_,_ = obspy.geodetics.base.gps2dist_azimuth(lat[ista],lon[ista],lat[ii],lon[ii])
        if dist/1000 < maxdist:
            sta_list.append(net[ii]+'.'+sta[ii])
    nsta = len(sta_list)
    
    # construct station pairs from the found stations
    allfiles = []
    for ii in range(nsta-1):
        for jj in range(ii+1,nsta):
            tfile1 = data_path+'/'+sta_list[ii]+'/'+sta_list[ii]+'_'+sta_list[jj]+'.h5'
            tfile2 = data_path+'/'+sta_list[jj]+'/'+sta_list[jj]+'_'+sta_list[ii]+'.h5'
            if os.path.isfile(tfile1):
                allfiles.append(tfile1)
            elif os.path.isfile(tfile2):
                allfiles.append(tfile2)
    nfiles = len(allfiles)

    # give it another chance for larger region
    if nfiles<maxnpair:
        print('station %s has no enough pairs'%sta[ista])
        continue

    # find closest grid point to load predicted dispersion curves
    tfiles = ['{0:s}/{1:5.1f}_{2:4.1f}.dat'.format(pre_file+'/pred_Rg',lon[ista],lat[ista]),\
        '{0:s}/{1:5.1f}_{2:4.1f}.dat'.format(pre_file+'/pred_Lg',lon[ista],lat[ista])]

    # extract common variables from ASDF data
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
            indx1 = np.where((mode==1) & (pper<1/freqmin))[0]
            indx2 = np.where((mode==2) & (pper<1/freqmin))[0]
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
                except Exception:
                    print("continue! cannot read %s "%tmp)
                    continue  
            
            # assign data matrix
            dist[icc] = tdist
            data[icc] = tdata[anpts-1:anpts+tnpts-1]*0.5+np.flip(tdata[anpts-tnpts:anpts],axis=0)*0.5
            if norm:
                data[icc] = data[icc]/np.max(data[icc])
            icc+=1

        # remove bad ones
        data = data[:icc]
        dist = dist[:icc]

        # sort according to distance
        ntrace = int(np.round(np.max(dist)+0.5)/dist_inc)
        spec   = np.zeros(shape=(ntrace,Nfft//2),dtype=np.complex64)
        ndist  = np.zeros(ntrace,dtype=np.float32)
        flag   = np.zeros(ntrace,dtype=np.int16)
        for td in range(0,ntrace):
            ndist[td] = td*dist_inc
            tindx = np.where((dist>=ndist[td]-0.5*dist_inc)&(dist<ndist[td]+0.5*dist_inc))[0]
            if len(tindx):
                flag[td] = 1
                tdata = np.mean(data[tindx],axis=0)
                if bdpass:
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
        p1 = 0.5
        p2 = 8
        pp = np.arange(p1,p2,0.02)
        c1 = 0.2
        c2 = 4.2
        cc = np.arange(c1,c2,0.05)
        nc = len(cc)

        # initalize 2D dispersion array
        freqVec = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft // 2]
        indx = np.where((freqVec<=freqmax) & (freqVec>=freqmin))[0]
        spec = spec[:,indx]
        freqVec = freqVec[indx]
        tNfft = len(freqVec)
        disp_array = np.zeros(shape=(nc,tNfft),dtype=np.complex64)

        # frequency-bessel transfom according to formulat 13 of Wang et al., JGR 2019
        for ifreq in range(tNfft):
            om = 2*pi*freqVec[ifreq]
            for ic in range(nc):
                tc = cc[ic]
                k = om/tc; vk = tc/om
                for idis in range(1,ntrace):
                    # make components for summation
                    r1 = ndist[idis];r0 = ndist[idis-1]
                    z1 = np.linspace(0,k*r1,101); z0 = np.linspace(0,k*r0,101)
                    B1 = np.sum(special.jv(0,z1)*(z1[1]-z1[0])); B0 = np.sum(special.jv(0,z0)*(z0[1]-z0[0]))

                    M = (k*r1*special.jv(0,r1*k)-B1)-(k*r0*special.jv(0,r0*k)-B0)
                    N = vk*spec[idis][ifreq]*r1*special.jv(1,k*r1)-vk*spec[idis-1][ifreq]*r0*special.jv(1,k*r0)
                    disp_array[ic][ifreq] +=  N+(spec[idis][ifreq]-spec[idis-1][ifreq])/(r1-r0)*vk**3*M
                        
        # convert from freq-c domain to period-c domain by interpolation
        fc = scipy.interpolate.interp2d(1/freqVec,cc,np.abs(disp_array))
        disp_new = fc(pp,cc)

        # do normalization for each frequency
        for ii in range(len(pp)):
            disp_new[:,ii] /= np.max(disp_new[:,ii])

        ########################################
        ############### PLOTTING ###############
        ########################################

        #---plot 2D dispersion image-----
        tmpt = '33'+str(ccomp.index(path)+1)
        plt.subplot(tmpt)

        extent = [pp[0],pp[-1],cc[0],cc[-1]]
        plt.imshow(np.abs(disp_new),cmap='jet',interpolation='bicubic',extent=extent,origin='lower',aspect='auto')
        if ccomp.index(path) == 1:
            plt.title('%s with %d pairs in %d km'%(sta[ista],nfiles,maxdist))
        if not len(tindx):
            plt.plot(per1,vel1,'w--');plt.plot(per2,vel2,'m--')
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