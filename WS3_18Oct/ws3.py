import numpy as np
import matplotlib.pyplot as plt
import matplotlib

dir_number = np.random.randint(0,4,100)
print(max(dir_number))
N_walks = 10

fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_aspect('equal')

ax1 = fig.add_subplot(122)
ax1.set_xlim(-1,N_walks)
ax1.set_ylim(-1,10)
ax1.set_xlabel('t')
ax1.set_ylabel('RMS')
plt.tight_layout()



x0 = 0
y0 = 0

x0prime = 0
y0prime = 0



dir = ['UP','DOWN','LEFT','RIGHT']

n=1

clr  = np.random.randint(0,6,1)

r = np.random.rand(1)
g = np.random.rand(1)
b = np.random.rand(1)
rgb = [r[0],g[0],b[0]]


undr_root = 0


while n<=N_walks:
    
    
    
    for i in range(len(dir_number)):
        
    
        if dir_number[i]==0:
            y0prime += 0.5
        if dir_number[i]==1:
            y0prime -= 0.5
        if dir_number[i]==2:
            x0prime -= 0.5
        if dir_number[i]==3:
            x0prime += 0.5
    
        
        ax.plot([x0, x0prime],[y0 ,y0prime], color=rgb)
        plt.pause(0.01)
        
        
        
        x0 = x0prime
        y0 = y0prime
    
       
        

        
    dist = x0prime**2 + y0prime**2
    undr_root+=dist
    undr_root = undr_root/n
    rms = np.sqrt(undr_root)
    
    ax1.plot(n, rms, marker='o', color=rgb)
    plt.pause(0.1)   
    
    dir_number = np.random.randint(0,4,100)
    clr  = np.random.randint(0,6,1)
    
    n+=1
    x0 = 0
    y0 = 0

    x0prime = 0
    y0prime = 0
    
    r = np.random.rand(1)
    g = np.random.rand(1)
    b = np.random.rand(1)
    rgb = [r[0],g[0],b[0]]
    


plt.show()