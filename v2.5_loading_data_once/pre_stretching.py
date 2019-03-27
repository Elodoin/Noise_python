from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
import noise_module
import numpy as np 
import pyasdf
import scipy
import glob
import os

#----the path for the data---
'''
this is not important at this moment

rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
STACKDIR = os.path.join(rootpath,'STACK')
sta = glob.glob(os.path.join(STACKDIR,'E.*'))
nsta = len(sta)
'''

#----some common variables-----
epsilon = 0.1
nbtrial = 50
tmin = -30
tmax = -15
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

        #--------parameters to store dv/v and cc--------
        dv = np.zeros(ndays,dtype=np.float32)
        cc = np.zeros(ndays,dtype=np.float32)
        cdp = np.zeros(ndays,dtype=np.float32)
        error = np.zeros(ndays,dtype=np.float32)

        #------loop through the reference waveforms------
        for ii in range(1,ndays):
            cur = data[ii,:]

            #----plug in the stretching function-------
            [dv[ii], cc[ii], cdp[ii], error[ii]] = Stretching_current(ref, cur, tvec, -epsilon, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)

        #----plot the results------
        plt.subplot(311)
        plt.title(h5file.split('/')[-1])
        plt.plot(dv,)
        plt.ylabel('dv/v [%]')
        plt.subplot(312)
        plt.plot(cc,'r-');plt.plot(cdp,'b-')
        plt.ylabel('cc')
        plt.subplot(313)
        plt.plot(error)
        plt.xlabel('days');plt.ylabel('errors [%]')
        plt.show()


def Stretching_current(ref, cur, t, dvmin, dvmax, nbtrial, window, fmin, fmax, tmin, tmax):
    """
    Stretching function: 
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.

    modified based on the script from L. Viens 04/26/2018 (Viens et al., 2018 JGR)

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
    Cof = np.zeros(Eps.shape,dtype=np.float32)

    # Set of stretched/compressed current waveforms
    for ii in range(len(Eps)):
        nt = t*Eps[ii]
        s = np.interp(x=t, xp=nt, fp=cur[window])
        waveform_ref = ref[window]
        waveform_cur = s
        Cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cdp = np.corrcoef(cur[window], ref[window])[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(Cof)
    if imax >= len(Eps)-1:
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


def mwcs(current, reference, freqmin, freqmax, df, tmin, window_length, step,
         smoothing_half_win=5):
    """The `current` time series is compared to the `reference`.
    Both time series are sliced in several overlapping windows.
    Each slice is mean-adjusted and cosine-tapered (85% taper) before being Fourier-
    transformed to the frequency domain.
    :math:`F_{cur}(\\nu)` and :math:`F_{ref}(\\nu)` are the first halves of the
    Hermitian symmetric Fourier-transformed segments. The cross-spectrum
    :math:`X(\\nu)` is defined as
    :math:`X(\\nu) = F_{ref}(\\nu) F_{cur}^*(\\nu)`
    in which :math:`{}^*` denotes the complex conjugation.
    :math:`X(\\nu)` is then smoothed by convolution with a Hanning window.
    The similarity of the two time-series is assessed using the cross-coherency
    between energy densities in the frequency domain:
    :math:`C(\\nu) = \\frac{|\overline{X(\\nu))}|}{\sqrt{|\overline{F_{ref}(\\nu)|^2} |\overline{F_{cur}(\\nu)|^2}}}`
    in which the over-line here represents the smoothing of the energy spectra for
    :math:`F_{ref}` and :math:`F_{cur}` and of the spectrum of :math:`X`. The mean
    coherence for the segment is defined as the mean of :math:`C(\\nu)` in the
    frequency range of interest. The time-delay between the two cross correlations
    is found in the unwrapped phase, :math:`\phi(\nu)`, of the cross spectrum and is
    linearly proportional to frequency:
    :math:`\phi_j = m. \nu_j, m = 2 \pi \delta t`
    The time shift for each window between two signals is the slope :math:`m` of a
    weighted linear regression of the samples within the frequency band of interest.
    The weights are those introduced by [Clarke2011]_,
    which incorporate both the cross-spectral amplitude and cross-coherence, unlike
    [Poupinet1984]_. The errors are estimated using the weights (thus the
    coherence) and the squared misfit to the modelled slope:
    :math:`e_m = \sqrt{\sum_j{(\\frac{w_j \\nu_j}{\sum_i{w_i \\nu_i^2}})^2}\sigma_{\phi}^2}`
    where :math:`w` are weights, :math:`\\nu` are cross-coherences and
    :math:`\sigma_{\phi}^2` is the squared misfit of the data to the modelled slope
    and is calculated as :math:`\sigma_{\phi}^2 = \\frac{\sum_j(\phi_j - m \\nu_j)^2}{N-1}`
    The output of this process is a table containing, for each moving window: the
    central time lag, the measured delay, its error and the mean coherence of the
    segment.
    .. warning::
        The time series will not be filtered before computing the cross-spectrum!
        They should be band-pass filtered around the `freqmin`-`freqmax` band of
        interest beforehand.
    :type current: :class:`numpy.ndarray`
    :param current: The "Current" timeseries
    :type reference: :class:`numpy.ndarray`
    :param reference: The "Reference" timeseries
    :type freqmin: float
    :param freqmin: The lower frequency bound to compute the dephasing (in Hz)
    :type freqmax: float
    :param freqmax: The higher frequency bound to compute the dephasing (in Hz)
    :type df: float
    :param df: The sampling rate of the input timeseries (in Hz)
    :type tmin: float
    :param tmin: The leftmost time lag (used to compute the "time lags array")
    :type window_length: float
    :param window_length: The moving window length (in seconds)
    :type step: float
    :param step: The step to jump for the moving window (in seconds)
    :type smoothing_half_win: int
    :param smoothing_half_win: If different from 0, defines the half length of
        the smoothing hanning window.
    :rtype: :class:`numpy.ndarray`
    :returns: [time_axis,delta_t,delta_err,delta_mcoh]. time_axis contains the
        central times of the windows. The three other columns contain dt, error and
        mean coherence for each window.
    """
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    window_length_samples = np.int(window_length * df)
    # try:
    #     from scipy.fftpack.helper import next_fast_len
    # except ImportError:
    #     from obspy.signal.util import next_pow_2 as next_fast_len
    from msnoise.api import nextpow2
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    # padd = next_fast_len(window_length_samples)
    count = 0
    tp = cosine_taper(window_length_samples, 0.85)
    minind = 0
    maxind = window_length_samples
    while maxind <= len(current):
        cci = current[minind:(minind + window_length_samples)]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = reference[minind:(minind + window_length_samples)]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(step*df)
        maxind += int(step*df)

        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # Calculate the cross-spectrum
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(smooth(fcur2, window='hanning',
                                  half_win=smoothing_half_win))
            dref = np.sqrt(smooth(fref2, window='hanning',
                                  half_win=smoothing_half_win))
            X = smooth(X, window='hanning',
                       half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / df)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,
                                                 freq_vec <= freqmax))

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
        time_axis.append(tmin+window_length/2.+count*step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(current) + step*df:
        logging.warning("The last window was too small, but was computed")

    return np.array([time_axis, delta_t, delta_err, delta_mcoh]).T