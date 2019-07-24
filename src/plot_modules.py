import os
import sys
import glob
import obspy
import pyasdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

'''
Ensembles of plotting functions to display intermediate/final waveforms from the NoisePy package.
by Chengxin Jiang (Jul.04.2019)

Specifically, this plotting module includes functions of:
    1) plot_substack_cc -> plot 2D matrix of the CC functions for one time-chunck (e.g., 2 days)
    2) plot_substack_moveout -> plot moveout of the CC functions between all station-pair averaged in one time-chunck (e.g. 2d)
    3) plot_substack_all -> plot 2D matrix of the CC functions for all time-chunck (e.g., every 1 day in 1 year)
    4) plot_all_moveout -> plot the moveout of the stacked CC functions for all time-chunk
'''

#############################################################################
###############PLOTTING FUNCTIONS FOR FILES FROM S0##########################
#############################################################################
def plot_waveform(sfile,net,sta,comp,freqmin,freqmax):
    '''
    display the downloaded waveform for station A

    Input parameters:
    sfile: containing all wavefrom data for a time-chunck in ASDF format
    net,sta,comp: network, station name and component 
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered

    USAGE: plot_waveform('temp.h5','CC','A05','BHZ',0.1,1)
    '''
    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    

#############################################################################
###############PLOTTING FUNCTIONS FOR FILES FROM S1##########################
#############################################################################

def plot_substack_cc(sfile,freqmin,freqmax,disp_lag=None,savefig=False,sdir=None):
    '''
    display the 2D matrix of the cross-correlation functions for a time-chunck.

    INPUT parameters:
    sfile: cross-correlation functions outputed by S1
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display

    USAGE: plot_substack_cc('temp.h5',0.1,1)
    '''
    # open data for read
    if savefig:
        if sdir==None:print('no path selected! save figures in the default path')

    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    # extract common variables
    dtype = 'CCF'
    path_lists = ds.auxiliary_data[dtype].list()
    dt     = ds.auxiliary_data[dtype][path_lists[0]].parameters['dt']
    maxlag = ds.auxiliary_data[dtype][path_lists[0]].parameters['maxlag']

    # lags for display   
    if not disp_lag:disp_lag=maxlag
    t = np.arange(-int(disp_lag),int(disp_lag),step=int(2*int(disp_lag)/5)) 
    indx1 = int((maxlag-disp_lag)/dt)
    indx2 = indx1+2*int(disp_lag/dt)+1

    for ipath in path_lists:
        net1,sta1,chan1,loc1,net2,sta2,chan2,loc2 = ipath.split('s')
        dist = ds.auxiliary_data[dtype][ipath].parameters['dist']
        ngood= ds.auxiliary_data[dtype][ipath].parameters['ngood']
        ttime= ds.auxiliary_data[dtype][ipath].parameters['time']
        timestamp = np.empty(ttime.size,dtype='datetime64[s]')
        
        # cc matrix
        data = ds.auxiliary_data[dtype][ipath].data[indx1:indx2]
        nwin = data.shape[0]
        amax = np.zeros(nwin,dtype=np.float32)

        # load cc for each station-pair
        for ii in range(nwin):
            data[ii] = bandpass(data[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            amax[ii] = max(data[ii])
            data[ii] /= amax[ii]
            timestamp[ii] = obspy.UTCDateTime(ttime[ii])
        
        # plotting
        fig,ax = plt.subplots(2,sharex=False)
        ax[0].matshow(data,cmap='seismic',extent=[-disp_lag,disp_lag,nwin,1],aspect='auto')
        ax[0].set_title('%s.%s.%s  %s.%s.%s  dist:%5.2f km' % (net1,sta1,chan1,net2,sta2,chan2,dist))
        ax[0].set_xlabel('time [s]')
        ax[0].set_ylabel('wavefroms')
        ax[0].set_xticks(t)
        ax[0].set_yticks(np.arange(0,nwin,step=2))
        ax[0].set_yticklabels(timestamp[0:-1:2])
        ax[0].xaxis.set_ticks_position('bottom')
        ax[1].plot(amax/min(amax),'r-')
        ax[1].plot(ngood,'b-')
        ax[1].set_xlabel('waveform number')
        ax[1].set_xticks(np.arange(0,nwin,2))
        ax[1].legend(['relative amp','ngood'],loc='upper right')

        # save figure or just show
        if savefig:
            if sdir==None:sdir = sfile.split('.')[0]
            if not os.path.isdir(sdir):os.mkdir(sdir)
            outfname = sdir+'/{0:s}{1:s}_{2:s}_{3:s}{4:s}_{5:s}.pdf'.format(net1,sta1,chan1,net2,sta2,chan2)
            fig.savefig(outfname, format='pdf', dpi=400)
            fig.close()
        else:
            plt.show()
            fig.close()

def plot_substack_moveout(sfile,freqmin,freqmax,disp_lag=None,savefig=False,sdir=None):
    '''
    display the moveout of the cross-correlation functions stacked for a time-chunck.

    INPUT parameters:
    sfile: cross-correlation functions outputed by S1
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE: plot_substack_moveout('temp.h5',0.1,1)
    '''
    # open data for read
    if savefig:
        if sdir==None:raise ValueError('need path info to savefig!')

    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    # extract common variables
    dtype = 'CCF'
    path_lists = ds.auxiliary_data[dtype].list()
    dt     = ds.auxiliary_data[dtype][path_lists[0]].parameters['dt']
    maxlag = ds.auxiliary_data[dtype][path_lists[0]].parameters['maxlag']

    # lags for display   
    if not disp_lag:disp_lag=maxlag
    t = np.arange(-int(disp_lag),int(disp_lag),step=(int(2*int(disp_lag)/5)))
    indx1 = int((maxlag-disp_lag)/dt)
    indx2 = indx1+2*int(disp_lag/dt)+1

    # cc matrix
    nwin = len(path_lists)
    data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    dist = np.zeros(nwin,dtype=np.float32)
    ngood= np.zeros(nwin,dtype=np.int16)

    for ii,ipath in enumerate(path_lists):
        dist[ii] = ds.auxiliary_data[dtype][ipath].parameters['dist']
        ngood[ii]= np.sum(ds.auxiliary_data[dtype][ipath].parameters['ngood'])
        tdata    = ds.auxiliary_data[dtype][ipath].data[:,indx1:indx2]
        if tdata.ndim==2:
            tdata= np.mean(tdata,axis=0)

        data[ii] = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
        data[ii] /= max(data[ii])

    ndist = np.argsort(dist)
    data = data[ndist]

    # plotting
    fig,ax = plt.subplots()
    ax.matshow(data,cmap='seismic',extent=[-disp_lag,disp_lag,dist[ndist[-1]],dist[ndist[0]]],aspect='auto')
    ax.set_title('%s' % (sfile.split('.')[0].split('/')[-1]))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('distance [km]')
    ax.set_xticks(t)
    #ax.grid(True)
    ax.xaxis.set_ticks_position('bottom')
    #ax.text(np.ones(len(ndist))*(disp_lag-5),dist[ndist],ngood[ndist],fontsize=8)

    # save figure or just show
    if savefig:
        outfname = sdir+'/moveout_'+sfile.split('.')[0].split('/')[-1]+'.pdf'
        fig.savefig(outfname, format='pdf', dpi=400)
        fig.close()
    else:
        plt.show()
        fig.close()

#############################################################################
###############PLOTTING FUNCTIONS FOR FILES FROM S2##########################
#############################################################################

def plot_substack_all(sfile,freqmin,freqmax,ccomp,disp_lag=None,savefig=False,sdir=None):
    '''
    display the 2D matrix of the cross-correlation functions stacked for all time windows.

    INPUT parameters:
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display
    ccomp: cross component of the targeted cc functions

    USAGE: plot_substack_all('temp.h5',0.1,1,'ZZ',50,True,'./')
    '''
    # open data for read
    if savefig:
        if sdir==None:print('no path selected! save figures in the default path')

    paths = ccomp
    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        # extract common variables
        dtype_lists = ds.auxiliary_data.list()[1:]
        dt     = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dt']
        dist   = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dist']
        maxlag = ds.auxiliary_data[dtype_lists[0]][paths].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    # lags for display   
    if not disp_lag:disp_lag=maxlag
    if disp_lag>maxlag:raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(disp_lag),int(disp_lag),step=int(2*int(disp_lag)/4)) 
    indx1 = int((maxlag-disp_lag)/dt)
    indx2 = indx1+2*int(disp_lag/dt)+1

    # other parameters to keep
    nwin = len(dtype_lists)
    data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    ngood= np.zeros(nwin,dtype=np.int16)
    ttime= np.zeros(nwin,dtype=np.int)
    amax = np.zeros(nwin,dtype=np.float32)

    for ii,itype in enumerate(dtype_lists):
        ngood[ii] = ds.auxiliary_data[itype][paths].parameters['ngood']
        ttime[ii] = ds.auxiliary_data[itype][paths].parameters['time']
        
        # cc matrix
        data[ii] = ds.auxiliary_data[itype][paths].data[indx1:indx2]
        data[ii] = bandpass(data[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
        amax[ii] = np.max(data[ii])
        data[ii] /= amax[ii]
        
    # plotting
    fig,ax = plt.subplots(2,sharex=False)
    ax[0].matshow(data,cmap='seismic',extent=[-disp_lag,disp_lag,nwin,1],aspect='auto')
    ax[0].set_title('%s dist:%5.2f km' % (sfile.split('/')[-1],dist))
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('wavefroms')
    ax[0].set_xticks(t)
    ax[0].xaxis.set_ticks_position('bottom')
    ax[1].plot(amax/min(amax),'r-')
    ax[1].plot(ngood,'b-')
    ax[1].set_xlabel('waveform number')
    ax[1].set_xticks(np.arange(0,nwin,nwin//15))
    ax[1].legend(['relative amp','ngood'],loc='upper right')
    # save figure or just show
    if savefig:
        if sdir==None:sdir = sfile.split('.')[0]
        if not os.path.isdir(sdir):os.mkdir(sdir)
        outfname = sdir+'/{0:s}.pdf'.format(sfile.split('/')[-1])
        fig.savefig(outfname, format='pdf', dpi=400)
        plt.close()
    else:
        plt.show()


def plot_all_moveout(sfiles,freqmin,freqmax,ccomp,dist_inc,disp_lag=None,savefig=False,sdir=None):
    '''
    display the moveout of the cross-correlation functions stacked for all time chuncks.

    INPUT parameters:
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    ccomp:   cross component
    dist_inc: distance bins to stack over
    disp_lag: lag times for displaying
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE: plot_substack_moveout('temp.h5',0.1,0.2,1,'ZZ',200,True,'./temp')
    '''
    # open data for read
    if savefig:
        if sdir==None:print('no path selected! save figures in the default path')
    
    dtype = 'Allstack'
    path  = ccomp

    # extract common variables
    try:
        ds    = pyasdf.ASDFDataSet(sfiles[0],mode='r')
        dt    = ds.auxiliary_data[dtype][path].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][path].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfiles[0]);sys.exit()
    
    # lags for display   
    if not disp_lag:disp_lag=maxlag
    t = np.arange(-int(disp_lag),int(disp_lag)+dt,step=(int(2*int(disp_lag)/4)))
    indx1 = int((maxlag-disp_lag)/dt)
    indx2 = indx1+2*int(disp_lag/dt)+1

    # cc matrix
    nwin = len(sfiles)
    data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    dist = np.zeros(nwin,dtype=np.float32)
    ngood= np.zeros(nwin,dtype=np.int16)    

    # load cc and parameter matrix
    for ii in range(len(sfiles)):
        sfile = sfiles[ii]

        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        try:
            # load data to variables
            dist[ii] = ds.auxiliary_data[dtype][path].parameters['dist']
            ngood[ii]= ds.auxiliary_data[dtype][path].parameters['ngood']
            tdata    = ds.auxiliary_data[dtype][path].data[indx1:indx2]
        except Exception:
            print("continue! cannot read %s "%sfile);continue

        data[ii] = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)

    # average cc
    ntrace = int(np.round(np.max(dist)+0.51)/dist_inc)
    ndata  = np.zeros(shape=(ntrace,indx2-indx1),dtype=np.float32)
    ndist  = np.zeros(ntrace,dtype=np.float32)
    for td in range(0,ntrace):
        tindx = np.where((dist<=td+dist_inc)&(dist>td))[0]
        if len(tindx):
            ndata[td] = np.mean(data[tindx],axis=0)
            ndist[td] = (td+0.5)*dist_inc

    # normalize waveforms 
    indx  = np.where(ndist>0)[0]
    ndata = ndata[indx]
    ndist = ndist[indx]
    for ii in range(ndata.shape[0]):
        ndata[ii] /= np.max(np.abs(ndata[ii]))

    # plotting figures
    fig,ax = plt.subplots()
    ax.matshow(ndata,cmap='seismic',extent=[-disp_lag,disp_lag,ndist[-1],ndist[0]],aspect='auto')
    ax.set_title('moveout allstack')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('distance [km]')
    ax.set_xticks(t)
    ax.xaxis.set_ticks_position('bottom')
    #ax.text(np.ones(len(ndist))*(disp_lag-5),dist[ndist],ngood[ndist],fontsize=8)
    
    # save figure or show
    if savefig:
        outfname = sdir+'/moveout_allstack_'+str(dist_inc)+'kmbin.pdf'
        fig.savefig(outfname, format='pdf', dpi=400)
        plt.close()
    else:
        fig.show()

