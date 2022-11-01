import numpy as np
import matplotlib.pyplot as plt


t = np.arange(0, 1.01, 0.01)
freq_range = 1/t
f = 10

a = np.sin(2*np.pi*f*t)
ffta = np.fft.fft(a/(len(t)))


fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(t, a, '-o', lw=1, color='r')
ax.set_xlim(0,1)
ax.set_ylim(-2,2)
ax.grid()


ax1 = fig.add_subplot(122)
ax1.plot(abs(ffta)**2, '-o', lw=1, color='b')
ax1.plot([1,0],[1,-1])
ax1.grid()

plt.show()