import numpy as np
import torchfile
import matplotlib.pyplot as plt


# set up
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

plotting = True
count = 0
while plotting:

    xs = torchfile.load('save/xs.t7')

    ax.cla()
    ax.scatter(xs[:,0], xs[:, 1], marker='.' , s = 5 )

    plt.draw()
    plt.pause(0.05)
    count += 1
    if count > 10000:
        plotting = False
        print('Finish Simulation')

plt.ioff()
plt.show()