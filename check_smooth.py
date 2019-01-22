import numpy as np
import matplotlib.pyplot as plt
import noise_module
import timeit

N = 10
a = np.random.rand(500,).astype(np.float32)
b = noise_module.running_abs_mean(a,N)
c = noise_module.running_ave(a,N)
d = noise_module.running_mean(a,N)

plt.plot(a,'r')
plt.plot(b,'g')
plt.plot(c,'b')
plt.plot(d,'y')

modes = ['original','running_abs_mean','running_mean']
plt.legend(modes, loc='upper right')
#plt.show()
