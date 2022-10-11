import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import widgets



y0 = 100

v0 = 0
dt = 0


def y_update(g):
    
    dv = v0 - g*dt
    dy = dv*dt
    
    return dv, dy



fig = plt.figure(1)
ax = plt.axes()
ax.set_position([0.075, 0.1, 0.4, 0.8])
ax.set_xlim(-y0*0.45,y0*0.45)
ax.set_ylim(-2, y0+5)
ax.set_aspect('equal')
ax.plot([-y0*0.45,y0*0.45], [0,0], lw=4)
init = ax.plot(0, y0, marker='s')

def Animate():

    dg = 0   
    g = -1.6 + dg
    y_upd = y0
    dt = 0
    while y_upd>=0:
    

        dt += 0.01
        
        dv, dy = y_update(g)
        
        ax.cla()
        ax.set_xlim(-y0*0.45,y0*0.45)
        ax.set_ylim(-2, y0+5)
        ax.set_aspect('equal')
        ax.plot([-y0*0.45,y0*0.45], [0,0], lw=4)
        ax.plot(0, y_upd, marker='s')
        plt.pause(0.01)
    
        y_upd = y_upd-dy
        
        if y_upd<=0:
            y_upd=0
            ax.cla()
            ax.set_xlim(-y0*0.45,y0*0.45)
            ax.set_ylim(-2, y0+5)
            ax.set_aspect('equal')
            ax.plot([-y0*0.45,y0*0.45], [0,0], lw=4)
            ax.plot(0, y_upd, marker='s')
            plt.pause(0.01)
            break
        

    


def StartCallback(event):
    Animate()

Startax = plt.axes()
Startax.set_position([0.6, 0.75, 0.1, 0.1])
StartButtonHandle = widgets.Button(Startax, 'Start')
StartButtonHandle.on_clicked(StartCallback)


def ThrustSliderCallback(val):
    dg = val/25
    return dg


thrustax = plt.axes()
thrustax.set_position([0.6, 0.5, 0.3, 0.03])
ThrustSliderHandle = widgets.Slider(thrustax, 'Thrust', 0, 100.0, valinit=0)
ThrustSliderHandle.on_changed(ThrustSliderCallback)

def closeCallback(event):
    plt.close()

    

bax = plt.axes()
bax.set_position([0.8, 0.75, 0.1, 0.1])
buttonHandle = widgets.Button(bax, 'Close')
buttonHandle.on_clicked(closeCallback)
plt.show()
