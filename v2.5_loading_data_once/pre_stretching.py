from obspy.signal.regression import linear_regression
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
import noise_module
import numpy as np 
import pyasdf
import scipy
import glob
import os


def Stretching_current(ref, cur, t, dvmin, dvmax, nbtrial, window, fmin, fmax, tmin, tmax):
    """
    modified the script from L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    https://github.com/lviens/2018_JGR

    INPUTS:
        - ref = Reference waveform (np.ndarray, size N)
        - cur = Current waveform (np.ndarray, size N)
        - t = time vector, common to both ref and cur (np.ndarray, size N)
        - dvmin = minimum bound for the velocity variation; example: dvmin=-0.03 for -3% of relative velocity change ('float')
        - dvmax = maximum bound for the velocity variation; example: dvmax=0.03 for 3% of relative velocity change ('float')
        - nbtrial = number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
        - window = vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
        For error computation:
            - fmin = minimum frequency of the data
            - fmax = maximum frequency of the data
            - tmin = minimum time window where the dv/v is computed 
            - tmax = maximum time window where the dv/v is computed 

    OUTPUTS:
        - dv = Relative velocity change dv/v (in %)
        - cc = correlation coefficient between the reference waveform and the best stretched/compressed current waveform
        - cdp = correlation coefficient between the reference waveform and the initial current waveform
        - error = Errors in the dv/v measurements based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384?1392

    The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .
    """ 
    Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
    cof = np.zeros(Eps.shape,dtype=np.float32)

    # Set of stretched/compressed current waveforms
    for ii in range(len(Eps)):
        nt = t*Eps[ii]
        s = np.interp(x=t, xp=nt, fp=cur[window])
        waveform_ref = ref[window]
        waveform_cur = s
        cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cdp = np.corrcoef(cur[window], ref[window])[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(cof)
    if imax >= len(Eps)-2:
        imax = imax - 2
    if imax <= 2:
        imax = imax + 2

    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], 100)
    ncof    = np.zeros(dtfiner.shape,dtype=np.float32)
    for ii in range(len(dtfiner)):
        nt = t*dtfiner[ii]
        s = np.interp(x=t, xp=nt, fp=cur[window])
        waveform_ref = ref[window]
        waveform_cur = s
        ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtfiner[np.argmax(ncof)]-100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384?1392
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, cc, cdp, error


def smooth(x, window='boxcar', half_win=3):
    """ some window smoothing """
    window_len = 2 * half_win + 1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.hanning(window_len).astype('complex')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]


def getCoherence(dcs, ds1, ds2):
    '''
    a sub-function from MSNoise to estimate the weight for getting delta t
    '''
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh

def nextpow2(x):
    """
    Returns the next power of 2 of `x`.
    :type x: int
    :param x: any value
    :rtype: int
    :returns: the next power of 2 of `x`
    """
    return int(np.ceil(np.log2(np.abs(x))))

def mwcs_dvv(ref, cur, moving_window_length, slide_step, delta, window, fmin, fmax, tmin, smoothing_half_win=5):
    #mwcs_dvv(ref, cur, t, dvmin, dvmax, nbtrial, window, fmin, fmax, tmin, tmax):
    """
    modified sub-function from MSNoise package by Thomas Lecocq. download from
    https://github.com/ROBelgium/MSNoise/tree/master/msnoise

    combine the mwcs and dv/v functionality of MSNoise into a single function

    ref: The "Reference" timeseries
    cur: The "Current" timeseries
    moving_window_length: The moving window length (in seconds)
    slide_step: The step to jump for the moving window (in seconds)
    delta: The sampling rate of the input timeseries (in Hz)
    window: The target window for measuring dt/t
    fmin: The lower frequency bound to compute the dephasing (in Hz)
    fmax: The higher frequency bound to compute the dephasing (in Hz)
    tmin: The leftmost time lag (used to compute the "time lags array")
    smoothing_half_win: If different from 0, defines the half length of
        the smoothing hanning window.
    :returns: [time_axis,delta_t,delta_err,delta_mcoh]. time_axis contains the
        central times of the windows. The three other columns contain dt, error and
        mean coherence for each window.
    """
    
    ##########################
    #-----part I: mwcs-------
    ##########################
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    window_length_samples = np.int(moving_window_length * delta)
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.85)

    #----does minind really start from 0??-----
    minind = 0
    maxind = window_length_samples

    #-------loop through all sub-windows-------
    while maxind <= len(window):
        cci = cur[window[minind:maxind]]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[window[minind:maxind]]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step*delta)
        maxind += int(slide_step*delta)

        #-------------get the spectrum-------------
        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # Calculate the cross-spectrum
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(smooth(fcur2, window='hanning',half_win=smoothing_half_win))
            dref = np.sqrt(smooth(fref2, window='hanning',half_win=smoothing_half_win))
            X = smooth(X, window='hanning',half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / delta)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= fmin,freq_vec <= fmax))

        # Get Coherence and its mean value
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin
        # weights for the WLS must be the variance !
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())

        delta_t.append(m)

        # print phi.shape, v.shape, w.shape
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(cur) + slide_step*delta:
        print("The last window was too small, but was computed")

    delta_t = np.array(delta_t)
    delta_err = np.array(delta_err)
    delta_mcoh = np.array(delta_mcoh)
    time_axis  = np.array(time_axis)

    #####################################
    #-----------part II: dv/v------------
    #####################################
    delta_mincho = 0.65
    delta_maxerr = 0.1
    delta_maxdt  = 0.1
    indx1 = np.where(delta_mcoh>delta_mincho)
    indx2 = np.where(delta_err<delta_maxerr)
    indx3 = np.where(delta_t<delta_maxdt)

    #-----find good dt measurements-----
    indx = np.intersect1d(indx1,indx2)
    indx = np.intersect1d(indx,indx3)
    if len(indx) > 2:

        #----estimate weight for regression----
        w = 1/delta_err[indx]
        w[~np.isfinite(w)] = 1.0

        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis[indx], delta_t[indx], w,intercept_origin=True)
    else:
        m0=0;em0=0

    return np.array([-m0*100,em0*100]).T


def wavg_wstd(data, errors):
    '''
    estimate the weights for doing linear regression in order to get dt/t
    '''

    d = data
    errors[errors == 0] = 1e-6
    w = 1. / errors
    wavg = (d * w).sum() / w.sum()
    N = len(np.nonzero(w)[0])
    wstd = np.sqrt(np.sum(w * (d - wavg) ** 2) / ((N - 1) * np.sum(w) / N))
    return wavg, wstd


#----the path for the data---
'''
this is not important at this moment

rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
STACKDIR = os.path.join(rootpath,'STACK')
sta = glob.glob(os.path.join(STACKDIR,'E.*'))
nsta = len(sta)
'''

#----some common variables-----
epsilon = 0.01
nbtrial = 50
tmin = -20
tmax = -5
fmin = 1
fmax = 3
comp = 'ZZ'
maxlag = 100

#----for plotting-----
Mdate = 12
NSV   = 2

h5file = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.AYHM/E.AYHM_E.ENZM.h5'

#-------open ASDF file to read data-----------
with pyasdf.ASDFDataSet(h5file,mode='r') as ds:
    slist = ds.auxiliary_data.list()

    #------loop through the reference waveforms------
    if slist[0]== 'Allstacked':

        #------useful parameters from ASDF file------
        rlist = ds.auxiliary_data[slist[0]].list()
        indxc  = rlist.index(comp)
        delta = ds.auxiliary_data[slist[0]][rlist[indxc]].parameters['dt']
        lag   = ds.auxiliary_data[slist[0]][rlist[indxc]].parameters['lag']

        #--------index for the data---------
        indx1 = int((lag-maxlag)/delta)
        indx2 = int((lag+maxlag)/delta)
        t     = np.arange(0,indx2-indx1+1,400)*delta-maxlag
        Ntau  = int(np.ceil(np.min([1/fmin,1/fmax])/delta)) + 15

        #----------plot waveforms-----------
        ndays = len(slist)
        data  = np.zeros((ndays,indx2-indx1+1),dtype=np.float32)
        for ii in range(ndays):
            tdata = ds.auxiliary_data[slist[ii]][rlist[indxc]].data[indx1:indx2+1]
            data[ii,:] = bandpass(tdata,fmin,fmax,int(1/delta),corners=4, zerophase=True)

        fig,ax = plt.subplots(2,sharex=True)
        ax[0].matshow(data/data.max(),cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
        new = noise_module.NCF_denoising(data,np.min([Mdate,data.shape[0]]),Ntau,NSV)
        ax[1].matshow(new/new.max(),cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
        ax[0].set_title('Filterd Cross-Correlations')
        ax[1].set_title('Denoised Cross-Correlations')
        ax[0].xaxis.set_visible(False)
        ax[1].set_xticks(t)
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].set_xlabel('Time [s]')
        plt.show()
        #outfname = directory + '/' + 'Fig_dv_' + virt + '.pdf'
        #fig.savefig(outfname, format='pdf', dpi=400)
        #plt.close(fig)

        #-------parameters for doing stretching-------
        tvec = np.arange(tmin,tmax,delta)
        ref  = data[0,:]
        window = np.arange(int(tmin/delta),int(tmax/delta))+int(maxlag/delta)

        #-------parameters for MWCS-----------
        moving_window_length = 2.5*int(1/fmin)
        slide_step = 0.5*moving_window_length

        #--------parameters to store dv/v and cc--------
        dv1 = np.zeros(ndays,dtype=np.float32)
        cc = np.zeros(ndays,dtype=np.float32)
        cdp = np.zeros(ndays,dtype=np.float32)
        error1 = np.zeros(ndays,dtype=np.float32)
        dv2 = np.zeros(ndays,dtype=np.float32)
        error2 = np.zeros(ndays,dtype=np.float32)

        #------loop through the reference waveforms------
        for ii in range(1,ndays):
            cur = data[ii,:]

            #----plug in the stretching function-------
            [dv1[ii], cc[ii], cdp[ii], error1[ii]] = Stretching_current(ref, cur, tvec, -epsilon, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)
            [dv2[ii], error2[ii]] = mwcs_dvv(ref, cur, moving_window_length, slide_step, int(1/delta), window, fmin, fmax, tmin)

        #----plot the results------
        plt.subplot(311)
        plt.title(h5file.split('/')[-1])
        plt.plot(dv1,'r-');plt.plot(dv2,'b-')
        plt.ylabel('dv/v [%]')
        plt.subplot(312)
        plt.plot(cc,'r-');plt.plot(cdp,'b-')
        plt.ylabel('cc')
        plt.subplot(313)
        plt.plot(error1,'r-');plt.plot(error2,'b-')
        plt.xlabel('days');plt.ylabel('errors [%]')
        plt.show()
