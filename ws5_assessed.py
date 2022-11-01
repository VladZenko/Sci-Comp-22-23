import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import scipy.fft


def sinf(t, f):
    return np.sin(2*np.pi*f*t)

def fft(t, f):
    return scipy.fft.fft(sinf(t, f)/(len(t)))

def line(x1, x2, y1, y2):
    return x1, x2,y1, y2

def pts(N):
    t = np.linspace(0, 1.0, int(N))
    return t



def sliderCallback(val):
    
    axesHandle.set_ydata(sinf(t, val))
    axesHandle1.set_ydata(abs(fft(t, val)**2))
    axesHandle2.set_xdata([val, val])
    axesHandle3.set_x(val+(Nqst_f/100))
    axesHandle3.set_text(str("{:.0f} Hz".format(val)))
    ax1.set_ylim(-0.1,
                 np.max(abs(fft(t, val)**2))+np.max((abs(fft(t, val)**2))/2))
    plt.draw()




t = pts(1000)

Nqst_f = len(t)/2*(1/np.max(t))





fig = plt.figure(figsize=(7, 5))
ax = plt.axes([0.1, 0.2, 0.3, 0.7])
ax.set_xlim(0,np.max(t))
ax.set_ylim(-2,2)
ax.set_xlabel('t, s')
ax.set_ylabel('Amplitude')
ax.set_title('signal')
ax.grid()

axesHandle, = ax.plot(t, sinf(t, 10), '-', lw=1, color='r')



ax1 = plt.axes([0.5, 0.2, 0.3, 0.7])
ax1.set_xlabel('Freq., Hz')
ax1.set_ylabel('Amplitude^2')
ax1.set_title('power spectrum')
ax1.set_xlim(0,Nqst_f)
ax1.set_ylim(-0.1,
             np.max(abs(fft(t, 10)**2))+np.max((abs(fft(t, 10)**2))/2))
ax1.grid()

axesHandle1, = ax1.plot(abs(fft(t, 10))**2, '-o', lw=1, color='b')
axesHandle2, = ax1.plot([10, 10], [0,-0.1], "--", lw=2, color='k')
axesHandle3 = ax1.text(10+(Nqst_f/100),-0.09, "10 Hz")


sax = plt.axes([0.1, 0.1, 0.3, 0.03])
sliderHandle = widgets.Slider(sax, "Frequency, Hz", 1, 100, valinit=10.0)
sliderHandle.on_changed(sliderCallback)

sax1 = plt.axes([0.1, 0.05, 0.3, 0.03])
sliderHandle1 = widgets.Slider(sax1, "$N_{points}$", 10, 2000, valinit=1000)
#sliderHandle1.on_changed()


plt.tight_layout()
plt.show()