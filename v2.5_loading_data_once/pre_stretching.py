import numpy as np 
import pyasdf
import scipy
import glob
import os

#----the path for the data---
rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
STACKDIR = os.path.join(rootpath,'STACK')
sta = glob.glob(os.path.join(STACKDIR,'E.*'))
nsta = len(sta)

#----some common variables-----
epsilon = 0.1
nbtrial = 50
tmin = -30
tmax = -15
fmin = 1
fmax = 3
comp = 'ZZ'


h5file = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.AYHM/E.AYHM_E.ENZM.h5'

#-------open ASDF file to read data-----------
with pyasdf.ASDFDataSet(h5file,mode='r') as ds:
    slist = ds.auxiliary_data.list()

    #------loop through the reference waveforms------
    if slist[0]== 'AllStacked':
        rlist = ds.auxiliary_data[slist[0]].list()
        indx  = rlist.index(comp)
        delta = ds.auxiliary_data[slist[0]][rlist[indx]].parameters['dt']
        lag   = ds.auxiliary_data[slist[0]][rlist[indx]].parameters['lag']

        #-------final preparation of the parameters-------
        tvec = np.arange(tmin,tmax,delta)
        window = np.arange(tmin/delta,tmax/delta)+int(lag/delta)
        ref  = ds.auxiliary_data[slist[0]][rlist[indx]].data[:]

        #------loop through the reference waveforms------
        for ii in range(1,len(slist)):
            cur = ds.auxiliary_data[slist[ii]][rlist[indx]].data[:]

            #----plug in the stretching function-------
            [dv, cc, cdp, dt, error, C] = Stretching_current(ref, cur, tvec, -epsilon, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)


def Stretching_current(ref, cur, t, dvmin, dvmax, nbtrial, window, fmin, fmax, tmin, tmax):
    """
    Stretching function: 
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.

    L. Viens 04/26/2018 (Viens et al., 2018 JGR)

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
        - Eps = Vector of Epsilon values (Epsilon =-dt/t = dv/v)
        - error = Errors in the dv/v measurements based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384?1392
        - C = vector of the correlation coefficient between the reference waveform and every stretched/compressed current waveforms

    The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .
    """ 
    Eps = np.asmatrix(np.linspace(dvmin, dvmax, nbtrial))
    L = 1 + Eps
    tt = np.matrix.transpose(np.asmatrix(t))
    tau = tt.dot(L)  # stretched/compressed time axis
    C = np.zeros((1, np.shape(Eps)[1]))

    # Set of stretched/compressed current waveforms
    for j in np.arange(np.shape(Eps)[1]):
        s = np.interp(x=np.ravel(tt), xp=np.ravel(tau[:, j]), fp=cur)
        waveform_ref = ref[window]
        waveform_cur = s[window]
        C[0, j] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cdp = np.corrcoef(cur[window], ref[window])[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(C)
    if imax >= np.shape(Eps)[1]-1:
        imax = imax - 2
    if imax <= 2:
        imax = imax + 2
    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[0, imax-2], Eps[0, imax+1], 500)
    func = scipy.interpolate.interp1d(np.ravel(Eps[0, np.arange(imax-3, imax+2)]), np.ravel(C[0,np.arange(imax-3, imax+2)]), kind='cubic')
    CCfiner = func(dtfiner)

    cc = np.max(CCfiner) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtfiner[np.argmax(CCfiner)] # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384?1392
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, cc, cdp, Eps, error, C