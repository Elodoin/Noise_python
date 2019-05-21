import os
import sys
import glob
import obspy
import pyasdf
import numpy as np
from obspy.io.sac.sactrace import SACTrace

'''
this script outputs the stacked cross-correlation functions into SAC traces

add an option to output the CCFs into txt files for image transform analysis
'''

#------absolute path to output data-------
STACKDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/pre_processing/STACK_decon/E.AYHM'
ALLFILES = glob.glob(os.path.join(STACKDIR,'*.h5'))
COMP_OUT = ['ZZ','TT','RR']
#COMP_OUT = ['ZR','ZT','ZZ','TR','TT','TZ','RR','RT','RZ']
dtype    = 'Allstacked'

#---output file format-----
out_SAC = True
out_TXT = False

if (not out_SAC) and (not out_TXT):
    raise ValueError('out_SAC and out_TXT cannot be False at the same time')

nfiles = len(ALLFILES)
if not os.path.isdir(os.path.join(STACKDIR,'STACK_SAC')):
    os.mkdir(os.path.join(STACKDIR,'STACK_SAC'))

#----loop through station pairs----
for ii in range(nfiles):

    with pyasdf.ASDFDataSet(ALLFILES[ii],mode='r') as ds:
        
        #-----get station info from file name-----
        fname = ALLFILES[ii].split('/')[-1]
        staS = fname.split('_')[0].split('.')[1]
        netS = fname.split('_')[0].split('.')[0]
        staR = fname.split('_')[1].split('.')[1]
        netR = fname.split('_')[1].split('.')[0]

        #-----read data information-------
        slist = ds.auxiliary_data.list()
        rlist = ds.auxiliary_data[slist[0]].list()
        maxlag= ds.auxiliary_data[slist[0]][rlist[0]].parameters['lag']
        dt    = ds.auxiliary_data[slist[0]][rlist[0]].parameters['dt']
        slat  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['latS']
        slon  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['lonS']
        rlat  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['latR']
        rlon  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['lonR']

        #----make sure data exists------
        if dtype in slist:
            for icomp in range(len(COMP_OUT)):
                comp = COMP_OUT[icomp]

                if comp in rlist:

                    if out_SAC:

                        #--------read the correlations---------
                        corr = ds.auxiliary_data[dtype][comp].data[:]
                        temp = netS+'.'+staS+'_'+netR+'.'+staR+'_'+comp+'.SAC'

                        #-------check whether folder exists-------
                        if not os.path.isdir(os.path.join(STACKDIR,'STACK_SAC')):
                            os.mkdir(os.path.join(STACKDIR,'STACK_SAC'))
                        filename = os.path.join(STACKDIR,'STACK_SAC',temp)

                        #--------write into SAF format----------
                        sac = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,\
                            delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=corr)
                        sac.write(filename,byteorder='big')

                    if out_TXT:
                        
                        #-----------output name and read data-------------
                        temp = netS+'.'+staS+'_'+netR+'.'+staR+'_'+comp+'.dat'
                        if not os.path.isdir(os.path.join(STACKDIR,'STACK_DAT')):
                            os.mkdir(os.path.join(STACKDIR,'STACK_DAT'))
                        filename = os.path.join(STACKDIR,'STACK_DAT',temp)
                        corr = ds.auxiliary_data[dtype][comp].data[:]
                        
                        #-------make an array for output-------
                        npts = len(corr)
                        indx = npts//2
                        data = np.zeros((3,indx+2),dtype=np.float32)
                        data[0,0] = slon; data[1,0] = slat; data[2,0] = 0
                        data[0,1] = rlon; data[1,1] = rlat; data[2,1] = 0
                        tt   = 0
                        for jj in range(indx):
                            data[0,2+jj] = tt
                            data[1,2+jj] = corr[indx+jj]
                            data[2,2+jj] = corr[indx-jj]
                            tt = tt+dt
                        
                        np.savetxt(filename,np.transpose(data))
