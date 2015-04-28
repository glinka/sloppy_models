import util_fns
import pca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

def shit_plot():
    optvals = np.genfromtxt('./data/optvals.csv', delimiter=',')
    errs = np.genfromtxt('./data/errs.csv')
    npts = optvals.shape[0]
    npts_toplot = 0
    optvals_toplot = np.empty((npts, 3))
    for i in range(npts):
        if errs[i] < 8e-9:
            optvals_toplot[npts_toplot] = optvals[i]
            npts_toplot = npts_toplot + 1
    print npts_toplot
    optvals_toplot = optvals_toplot[:npts_toplot,:]

    optvals_toplot[:,2] = optvals_toplot[:,2]*1e8
    optvals_toplot[:,2] = optvals_toplot[:,2] - np.amin(optvals_toplot[:,2])
    print optvals_toplot[:,2]
    sing_vals, right_sing_vect = pca.pca(optvals_toplot)
    print sing_vals
    print right_sing_vect
    fig = plt.figure()
    ax = fig.add_subplot(111)
    proj = np.dot(optvals_toplot, right_sing_vect[:,:2])
    ax.scatter(proj[:,0], proj[:,1], c=optvals_toplot[:,2])
    ax.set_ylim((np.amin(proj[:,1]), np.amax(proj[:,1])))
    plt.show(fig)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(optvals_toplot[:,0], optvals_toplot[:,1], optvals_toplot[:npts_toplot,2])
    # ax.set_xlabel('x0')
    # ax.set_ylabel('y0')
    # ax.set_zlabel(r'$\epsilon$')
    # plt.show(fig)

def committee_meeting_sloppiness():
    p = 5
    n = 10
    k1s = np.logspace(-2, 3, n)
    k2s = p/k1s
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k1s, k2s)
    ax.set_xlabel(r'$k_1$')
    ax.set_ylabel(r'$k_2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('./figs/committee/ks.png')
    # add noise as if finding from optimization
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = 100
    k = 5
    ps = np.linspace(4, 5, k)
    for p in enumerate(ps):
        k1s = np.logspace(-1, 1, n) + np.random.uniform(size=n)
        k2s = p[1]/k1s
        ax.scatter(k1s, k2s)
    ax.set_xlabel(r'$k_1$')
    ax.set_ylabel(r'$k_2$')
    plt.tight_layout()
    plt.savefig('./figs/committee/noisey_ks.png')
    

if __name__=="__main__":
    committee_meeting_sloppiness()
    # shit_plot()
