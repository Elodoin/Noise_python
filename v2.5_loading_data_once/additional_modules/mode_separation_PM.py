import obspy
import pyasdf
import numpy as np 
import matplotlib.pyplot as plt 
from obspy.signal.filter import bandpass

'''
seperate fundamental and higher mode by particle motion, based on the unwrapped phase
from the Z-Z and Z-R components of the Green's tensor
'''

h5file = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK1/E.ABHM/E.ABHM_E.KSCM.h5'
ccomp1 = 'ZZ'
ccomp2 = 'ZR'
twin   = 50
fqmin  = 0.3
fqmax  = 5

with pyasdf.ASDFDataSet(h5file,mode='r') as ds:
    slist = ds.auxiliary_data.list()

    #---check whether stacked data exists----
    if slist[0] == 'Allstacked':
        rlist = ds.auxiliary_data[slist[0]].list()

        #------some common parameters--------
        dt = ds.auxiliary_data[slist[0]][rlist[0]].parameters['dt']
        tlag = ds.auxiliary_data[slist[0]][rlist[0]].parameters['lag']
        npts = tlag*int(1/dt)*2+1
        indx1 = npts//2
        tindx = twin*int(1/dt)+1
        tt    = np.linspace(0,twin,tindx)

        #----check if ZZ and ZR both exists----
        if (ccomp1 in rlist) and (ccomp2 in rlist):
            data1 = ds.auxiliary_data[slist[0]][ccomp1].data[indx1:indx1+tindx]
            data2 = ds.auxiliary_data[slist[0]][ccomp1].data[indx1-tindx+1:indx1+1]
            dataZ = 0.5*data1+0.5*np.flip(data2)
            dataZ = bandpass(dataZ,fqmin,fqmax,df=int(1/dt),corners=4,zerophase=True)

            data1 = ds.auxiliary_data[slist[0]][ccomp2].data[indx1:indx1+tindx]
            data2 = ds.auxiliary_data[slist[0]][ccomp2].data[indx1-tindx+1:indx1+1]
            dataR = 0.5*data1+0.5*np.flip(data2)
            dataR = bandpass(dataR,fqmin,fqmax,df=int(1/dt),corners=4,zerophase=True)

            #-------unwrapping phase--------
            tdata = dataR+1j*dataZ
            phase = np.angle(tdata)
            plt.plot(tt,np.unwrap(2 * phase)/2)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Unwrapped Phase')
            plt.show()