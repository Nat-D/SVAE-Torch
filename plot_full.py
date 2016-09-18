import numpy as np 
import torchfile
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse



# set up
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)



mean = torchfile.load('save/m.t7')
colors = {}
for j in xrange(len(mean)):
    colors[j] = np.random.rand(3,1)


plotting = True
count = 0
while plotting:
    
    ax.cla()
    x = torchfile.load('save/xs.t7')
    ax.scatter(x[:,0], x[:, 1], marker='.' , s = 5 )
    mean = torchfile.load('save/m.t7')
    ax.scatter(mean[:,0], mean[:,1], marker='o', s = 10, color='red')

    ax2.cla()
    y = torchfile.load('save/spiral.t7')
    ax2.scatter(y[:,0], y[:, 1], marker='.' , s = 5 )
    y_recon = torchfile.load('save/recon.t7')
    label   = torchfile.load('save/label.t7')

    # draw ellipse
    cov = torchfile.load('save/cov.t7')
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    for i in xrange(len(cov)):
        vals, vecs = eigsorted(cov[i])
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        # Width and height are "full" widths, not radius
        nstd = 2.0
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy= mean[i], width=width, height=height, angle=theta, alpha=0.1, color=colors[i])

        ax.add_artist(ellip)

        # plots latents samples
        key = label - 1 == i
        # plot reconstruction 
        ax2.scatter(y_recon[key,0], y_recon[key, 1], marker='.' , s = 5 , color=colors[i])



    plt.draw()
    plt.pause(0.05)
    count += 1
    if count > 100000:
        plotting = False
        print('Finish Simulation')

plt.ioff()
plt.show()
