import numpy as np
import torchfile
import matplotlib.pyplot as plt

y = torchfile.load('save/spiral.t7')

# set up
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

plotting = True
count = 0
while plotting:
    
    y_recon = torchfile.load('save/recon.t7')

    ax.cla()
    ax.scatter(y[:,0], y[:, 1], marker='.' , s = 5 )
    ax.scatter(y_recon[:,0], y_recon[:, 1], marker='.' , s = 5, color= 'red' )
    plt.draw()
    plt.pause(0.05)
    count += 1
    if count > 10000:
        plotting = False
        print('Finish Simulation')

plt.ioff()
plt.show()