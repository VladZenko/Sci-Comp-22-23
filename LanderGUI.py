import numpy as np
from matplotlib import pyplot as plt
from matplotlib import widgets


g = -1.6
y0 = 10
y_upd = y0
v0 = 0
dt = 0.05


fig = plt.figure(1)
ax = plt.axes()
ax.set_position([0.075, 0.1, 0.4, 0.8])


while y_upd>=0:

    ax.cla()
    ax.set_xlim(-y0*0.45,y0*0.45)
    ax.set_ylim(-2, y0+5)
    ax.set_aspect('equal')
    ax.plot([-y0*0.45,y0*0.45], [0,0], lw=4)
    ax.plot(0, y_upd, marker='s')
    plt.pause(0.01)

    y_upd = y_upd-0.1




def closeCallback(event):
    plt.close()

bax = plt.axes()
bax.set_position([0.8, 0.75, 0.1, 0.1])
buttonHandle = widgets.Button(bax, 'Close')
buttonHandle.on_clicked(closeCallback)
plt.show()