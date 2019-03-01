import h5py
import numpy as np
from matplotlib import pyplot as plt

CCFDIR = '/Users/chengxin/Documents/Harvard/Julia/julia_noise_shared/data_test/CCF/python'
py_allnames=[]
py_fid=h5py.File(CCFDIR+"/2010_01_10.h5",'r')
py_allnames=sorted(list(py_fid.keys()))
print("python version #=",len(py_allnames))

CCFDIR = '/Users/chengxin/Documents/Harvard/Julia/julia_noise_shared/data_test/CCF/julia'
jul_fid=h5py.File(CCFDIR+"/test.h5",'r')
jul_allnames=sorted(list(jul_fid.keys()))
print("julia version #=",len(jul_allnames))

id=210
py_dat = py_fid[py_allnames[id]][:]
jul_dat=jul_fid[jul_allnames[id]][:]
print("python and julia files:",py_allnames[id],jul_allnames[id])
print("lengths:",len(py_dat),len(jul_dat))

maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq

t=np.linspace(-maxlag,maxlag,len(jul_dat))
plt.plot(t,jul_dat);plt.plot(t,py_dat)
plt.legend(['julia','python'],loc='upper right')
plt.show()