import numpy as np
import matplotlib.pyplot as plt
import noise_module
import obspy
import timeit

'''
This script compares several different ways for smoothing a signal,
including convolution(one-side average), running average mean with 
numba compiled and a function from obspy
'''

N = 40
a = np.random.rand(500,).astype(np.float32)
b = noise_module.running_abs_mean(a,N)
c = noise_module.running_ave(a,N)
#d = noise_module.running_mean(a,N)
e = obspy.signal.util.smooth(a,N)

plt.plot(a,'r')
plt.plot(b,'g')
plt.plot(c,'b')
plt.plot(e,'y')

modes = ['original','running_abs_mean','running_ave','util smooth']
plt.legend(modes, loc='upper right')
plt.show()
