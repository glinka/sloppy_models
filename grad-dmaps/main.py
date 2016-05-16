# numpy
import numpy as np
# scipy
from scipy.optimize import minimize
import scipy.sparse.linalg as spla
# matplotlib
from matplotlib import colors, colorbar, cm, pyplot as plt, gridspec as gs, tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# mpi
from mpi4py import MPI
# utilities
import os
import sys
from time import time

import dmaps
import plot_dmaps
from dmaps_kernels import data_kernel
from util_fns import progress_bar
from rawlings_model.main import dmaps_param_set
from zagaris_model import Z_Model as ZM
from algorithms import Integration
import algorithms.CustomErrors as CustomErrors
from algorithms.PseudoArclengthContinuation import PSA
from algorithms.Derivatives import gradient
import util_fns as uf
from pca import pca

# a = 1 # parabola scaling, global for the sake of below fns

# # Gamma(x) = a*x^2
# def dGamma(x):
#     return 2*a*x

# def ddGama(x):
#     return 2*a

# # Gamma(x) = a*x^3
# def dGamma(x):
#     return 3*a*np.power(x, 2)

# # A = -0.95; B = 0.95; loglam_min = -2; loglam_max = 0
# def ddGama(x):
#     return 6*a*x

# Gamma(x) = 0.25*(np.log(x+1) - np.log(1-x) - 2*x)
# def Gamma(x):
#     return 0.25*(np.log(x+1) - np.log(1-x) - 2*x)

# def dGamma(x):
#     return 0.5*np.power(x,2)/(1 - np.power(x, 2))

# def ddGama(x):
#     return x/np.power(1 - np.power(x, 2), 2)


# A = -3.0; B = 3.0; loglam_min = -2; loglam_max = 1
# BSQRCONST = 9
# def Gamma(x):
#     return np.sqrt((1 + BSQRCONST)/(1 + BSQRCONST*np.power(np.cos(x), 2)))*np.cos(x)

# def dGamma(x):
#     return -np.power((1+BSQRCONST)/(1+BSQRCONST*np.power(np.cos(x), 2)), 1.5)*np.sin(x)/(1+BSQRCONST)

# def ddGama(x):
#     return np.cos(x)*np.sqrt((1+BSQRCONST)/(1+BSQRCONST*np.power(np.cos(x), 2)))*(-1 - 2*BSQRCONST + BSQRCONST*np.cos(2*x))/np.power(1+BSQRCONST*np.power(np.cos(x), 2), 2)

A = -1.0; B = 1.0;
loglam_min = -4; loglam_max = 2

def Gamma(x):
    return np.sin(np.pi*(x - 1)) + x

def dGamma(x):
    return np.pi*np.cos(np.pi*(x-1)) + 1

def ddGama(x):
    return -np.pi*np.pi*np.sin(np.pi*(x - 1))

def kernel_lam_fig():
    """Reveals effect of different values of epsilon on the resulting eigenfunctions"""
    n = 1000 # number of grid points
    # # visualize dataset
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xdata, Gamma(xdata))
    # ax.set_xlabel('x')
    # ax.set_ylabel('y(x)')
    # ax.set_title('Curve on which to apply new kernel')
    # plt.show()

    # # loop over different values of lambda, storing top 'k' eigenvectors for each
    k = 50
    nlams = 1
    eps = 0.01
    lam = 1e-3
    # lams = np.logspace(loglam_min,loglam_max,nlams)
    # eigvects = np.empty((nlams, k-1, n))
    # eigvals = np.empty((nlams, k))

    nas = 5
    As = np.linspace(-1, -0.9, nas)
    eigvects = np.empty((nas, k-1, n))
    eigvals = np.empty((nas, k))
    for m, A in enumerate(As):
        print A
        B = -A
        h = (B - A)/(n-1) # corresponding gridsize
        h_squared = np.power(h, 2)
        xdata = np.linspace(A, B, n)
        M = np.empty((n,n))
        for i in range(n):
            x1 = xdata[i]
            for j in range(n):
                x2 = xdata[j]
                M[i,j] = np.exp(-(np.power(x1-x2, 2) + np.power(Gamma(x1) - Gamma(x2), 2)/lam)/eps)

        # normalize by degree
        D_half_inv = np.identity(n)/np.sqrt(np.sum(M,1))
    
        # # find eigendecomp
        Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k) # eigsh (eigs hermitian)
        Meigvects = np.dot(D_half_inv, Meigvects)
        sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
        Meigvals = Meigvals[sorted_indices]
        Meigvects = Meigvects[:,sorted_indices]
        Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)

        eigvects[m] = Meigvects[:,1:k].T
        eigvals[m] = Meigvals

    eigvects.dump('./data/eigenfns-vs-lambda.pkl')    
    eigvals.dump('./data/eigenvals-vs-lambda.pkl')

def temppfig():
    eigvals = np.load('./data/eigenvals-vs-lambda.pkl')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(eigvals.shape[0]):
        ax.plot(np.arange(eigvals.shape[1]), eigvals[i], label=str(i))
    ax.legend()

def tempfig2():
    eigvects = np.load('./data/eigenfns-vs-lambda.pkl')
    n = eigvects.shape[2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nas = 5
    As = np.linspace(-1, -0.9, nas)
    for i, A in enumerate(As):
        B = -A
        xdata = np.linspace(A, B, n)
        if eigvects[i,0,0] > 0:
            eigvects[i,0] = -eigvects[i,0]
        ax.plot(xdata, eigvects[i,0], label='A=' + str(A))
    ax.legend()
    plt.show()
    

def nonuniform_sampling_lambda_fig():
    """varying lambda (allowing density along curve to change)"""
    
    # a0 = 1  # slope of 1st and 3rd segment
    # b0 = -1  # slope of 2nd segment
    # for b0 in [0, -0.1, -0.25, -0.5, -0.75, -1]:#[-1, -5, -10, -100]:#[0.0, 0.25, 0.50, 0.75, 1, 1.5, 2, 5, 10, 100]:
    # b0 = np.pi/2 - np.arctan(b0)

    a = 1
    b = -0.5
    ra = np.sqrt(1+a*a)
    rb = np.sqrt(1+b*b)  # arc-length factors on segments
    L = 2*ra/3 + rb/3  # total graph length
    N = 200  # no. of grid intervals of length ds
    ds = L/N  # arc length of grid interval

    xy = np.zeros((N+1,2))  # initialization of (x,y)-values
    i = 0  # counter initialization

    # 1st segment
    while xy[i,0] < 1./3:
        xy[i+1] = xy[i] + [ds/ra,ds*np.sqrt(1-1/(ra*ra))*np.sign(a)] 
        i = i+1 

    # 1st point of 2nd segment
    xy[i,0] = 1./3 + (ds - (1./3-xy[i-1,0])*ra)/rb
    xy[i,1] = xy[i-1,1] + a*(1./3 - xy[i-1,0]) + b*(xy[i,0] - 1./3) 

    # 2nd segment
    while xy[i,0] < 2./3:
        xy[i+1] = xy[i] + [ds/rb,ds*np.sqrt(1-1/(rb*rb))*np.sign(b)] 
        i = i+1 

    # 1st point of 3rd segment
    xy[i,0] = 2./3 + (ds - (2./3-xy[i-1,0])*rb)/ra 
    xy[i,1] = xy[i-1,1] + b*(2./3 - xy[i-1,0]) + a*(xy[i,0] - 2./3) 

    # 3rd segment
    while xy[i,0] < (1-1e-12):
        xy[i+1] = xy[i] + [ds/ra,ds*np.sqrt(1-1/(ra*ra))*np.sign(a)]
        i = i+1

    # print '%1.16f, %1.16f' % (xy[-2,0], xy[-1,0])

    # actual arc lengths (measured in ds-units) two entries should be slightly
    # under one, due because we measure along chord instead of through corner.

    ds_ratio = np.empty(N)
    for i in range(N):
        ds_ratio[i] = np.sqrt(np.power(xy[i+1,0]-xy[i,0], 2) + np.power(xy[i+1,1]-xy[i,1], 2))/ds 

    # # print ds_ratio
    # plt.plot(xy[:,0], xy[:,1], lw=5)
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$\gamma(x)$')
    # plt.xlim((0,1))
    # plt.show()


    # calculate cos(\pi x)
    phi1 = np.cos(np.pi*xy[:,0])
    # i = 0
    # while xy[i,0] < 1./3:
    #     phi1[i] = np.cos(np.pi*ra*xy[i,0]/L)
    #     i = i + 1
    # while xy[i,0] < 2./3:
    #     phi1[i] = np.cos(np.pi*rb*(xy[i,0] - (1 - ra/rb)/3)/L)
    #     i = i + 1
    # for x in xy[i:]:
    # # while xy[i,0] <= 1:
    #     phi1[i] = np.cos(np.pi*ra*(xy[i,0] - (1 - rb/ra)/3)/L)
    #     i = i + 1
    



    # do some dmaps
    npts = N + 1
    xvals = xy[:,0]
    yvals = xy[:,1]

    gsize = 30
    gspec = gs.GridSpec(gsize,gsize)
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(gspec[gsize/2:,:gsize-2])

    ax_dataset = fig.add_subplot(gspec[:gsize/2-2,:gsize-2])
    ax_dataset.scatter(xy[:,0], xy[:,1], s=50)
    ax_dataset.set_xticklabels([''])
    ax_dataset.set_xlim((0,1))
    ax_dataset.set_ylim((0, 0.5))
    ax_dataset.set_ylabel(r'$y$')
    ax_dataset.set_title('Data')

    lams = [10, 1, 0.75, 0.5, 0.1, 0.01]
    colornorm = colors.Normalize(vmin=np.min(np.log10(lams)), vmax=np.max(np.log10(lams))) # based on lam range
    colormap = cm.ScalarMappable(norm=colornorm, cmap='viridis')

    M = np.empty((npts,npts))
    for lam in lams:# = 0.01
        # lam = 0.01
        eps = ds/np.sqrt(lam)
        # for eps in [0.5*ds, ds, 3*ds, 5*ds]:# = 2*ds

        # fig_eigvals = plt.figure(figsize=(36, 20))
        # ax_eigvals = fig_eigvals.add_subplot(111)
        # neigvals = 10

        for i in range(npts):
            x1 = xvals[i]
            y1 = yvals[i]
            for j in range(npts):
                x2 = xvals[j]
                y2 = yvals[j]
                M[i,j] = np.exp(-(np.power(x1-x2, 2) + np.power(y1 - y2, 2)/lam)/eps)
        # normalize by degree
        D_half_inv = np.identity(npts)/np.sqrt(np.sum(M,1))

        # # find eigendecomp
        k = 50
        Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k) # eigsh (eigs hermitian)
        Meigvects = np.dot(D_half_inv, Meigvects)
        sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
        Meigvals = Meigvals[sorted_indices]
        Meigvects = Meigvects[:,sorted_indices]
        Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)

        ax.plot(xy[:,0], 0.1*Meigvects[:,1]/Meigvects[0,1], c=colormap.to_rgba(np.log10(lam)), lw=3)

        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # gsize = 2
        # gspec = gs.GridSpec(gsize,gsize)
        # fig = plt.figure(figsize=(36, 20))
        # ax = fig.add_subplot(gspec[0,:])
        # ax.scatter(xy[:,0], xy[:,1], c=np.sign(Meigvects[0,1])*Meigvects[:,1], s=100)
        # ax.set_ylabel(r'$\gamma(x)$')

        # ax2 = fig.add_subplot(gspec[1,:], sharex=ax)
        # # ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[0,1]*phi1, c='r', label=r'$\cos(x)$')
        # # ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[:,1], c='b', label='DMAPS')

        # ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[:,1], c='r', label=r'$\Phi_1$')
        # # ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[:,2], c='b', label=r'$\Phi_2$')

        # ax2.legend(fontsize=36)
        # ax2.set_xlabel(r'$x$')
        # ax2.set_ylabel(r'$\Phi_1$')
        # # ax2.set_ylim((-0.1, 0.1))

        # # add some vertical lines

        # # yrange = ax.get_ylim()
        # # yrange = np.linspace(yrange[0], yrange[1], 100)
        # # ax.plot(np.ones(100)/3., yrange, color='k')
        # # ax.plot(2*np.ones(100)/3., yrange, color='k')
        # # ax2.plot(np.ones(100)/3., yrange, color='k')
        # # ax2.plot(2*np.ones(100)/3., yrange, color='k')

        # ax.set_xlim((0, np.max(xy[:,0])))

        # ax.set_ylim(bottom=0)

        # lam_str = '%1.0e' % lam
        # bstr = '%1.2f' % b
        # eps_str = '%1.1e' % eps
        # title = r' $b=$' + bstr + r' $\lambda=$' + lam_str + r' $\epsilon=$' + eps_str
        # ax.set_title(title)

        # fig.subplots_adjust(bottom=0.15)

        # # plt.show()
        # # plt.savefig('./figs/slope-test/b' + str(b) + 'eps' + eps_str + '.png')
        # # plt.savefig('./figs/slope-test/b' + str(b) + 'lam' + lam_str + '.png')
        # # plt.savefig('./figs/slope-test/eps' + eps_str + 'lam' + lam_str + '.png')
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # ax_eigvals.plot(np.arange(neigvals), Meigvals[:neigvals], label=r'$\lambda=$' + lam_str)

    ax.set_ylim((-0.1, 0.1))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\Phi_1$')
    ax.set_title('DMAPS')
    ax_cb = fig.add_subplot(gspec[gsize/2:,gsize-1])
    colorbar.ColorbarBase(ax_cb, cmap='viridis', norm=colornorm, label=r'$\log(\lambda)$', ticks=(-2, -1, 0, 1))

    plt.show()

    # ax_eigvals.set_xlabel('Eigenvalue index')
    # ax_eigvals.set_ylabel('Eigenvalue')
    # ax_eigvals.legend(loc=3)
    # ax_eigvals.set_xlim(right=neigvals-1)
    # plt.figure(fig_eigvals.number)
    # plt.savefig('./figs/slope-test/eigvals.png')

    xy.dump('./data/tempxy' + str(b) + '.pkl')    
    Meigvects.dump('./data/tempeigvects' + str(b) + '.pkl')
    Meigvals.dump('./data/tempeigvals' + str(b) + '.pkl')


def disconnected_parallel_lines():
    """Data dmaps on disconnected parallel lines"""
    # make data on line
    a = 1
    npts = 200
    xdata = np.linspace(0,2,npts)
    ydata = a*xdata
    # shift second half down and to the right
    npts_half = npts/2
    dx = 0.5*(1 + a*a) # x1*(1+a*a)
    # dx = 0.0
    ydata[npts_half:] = ydata[npts_half:] - a
    xdata[npts_half:] = xdata[npts_half:] + dx - 1
    data = np.array((xdata, ydata)).T

    # # angle between line and x-axis
    # theta = np.arccos(1./np.sqrt(1 + a*a))
    # start of 'overlap' should be at x = 1/2, set dx accordingly
    
    print np.min(np.linalg.norm(data[:90] - data[101], axis=1))
    print 'distance between segments:', a*dx/np.sqrt(1 + a*a)

    # # plot dataset
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(data[:,0], data[:,1])
    # plt.show()

    # loop over range of epsilons
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    # eps = 0.2
    k = 150
    epss = [0.1, 0.2, 0.5, 0.6, 0.8, 1, 2, 3, 5]

    colornorm = colors.Normalize(vmin=np.log10(np.min(epss)), vmax=np.log10(np.max(epss))) # colors.Normalize(vmin=np.min(embeddings), vmax=np.max(embeddings))
    colormap = cm.ScalarMappable(norm=colornorm, cmap='viridis')

    # cs = ['r', 'b', 'g', 'c', 'y']
    eigvects_index = 1
    for i, eps in enumerate(epss):
        eigvals, eigvects = dmaps.embed_data(data, k, epsilon=eps)
        print eigvals[:4]
        # ax.scatter(data[:,0], np.sign(eigvects[0,eigvects_index])*eigvects[:,eigvects_index], c=cs[i%5])
        eps_str = '%1.1f' % eps
        ax.scatter(eigvals[1]*np.sign(eigvects[0,1])*eigvects[:,1], eigvals[2]*np.sign(eigvects[0,2])*eigvects[:,2], eigvals[3]*np.sign(eigvects[0,3])*eigvects[:,3], c=colormap.to_rgba(np.log10(eps)), label=r'$\epsilon=$' + eps_str)
        # ax.scatter(eigvals[1]*np.sign(eigvects[0,1])*eigvects[:,1], eigvals[2]*np.sign(eigvects[0,2])*eigvects[:,2], c=colormap.to_rgba(np.log10(eps)))
    ax.set_xlabel(r'$\Phi_1$')
    ax.set_ylabel(r'$\Phi_2$')

    ax.set_zlabel(r'$\Phi_3$')
    ax.set_zticks([])

    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(fontsize=48, framealpha=0.8)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # do some heatmaps
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # fig = plt.figure(figsize=(36, 20))
    # ax = fig.add_subplot(111)
    # eps = 0.5
    # k = 150
    # eigvals, eigvects = dmaps.embed_data(data, k, epsilon=eps)
    # embeddings = eigvects*eigvals
    # for i in range(npts):
    #     progress_bar(i, npts-1)
    #     ax.scatter(data[:,0], data[:,1], c=np.linalg.norm(embeddings - embeddings[i], axis=1), s=1000)

    #     ax.set_xlim((0,2))
    #     ax.set_ylim((0,1))
    #     ax.set_xlabel(r'$x$')
    #     ax.set_ylabel(r'$\gamma(x)$')
        

    #     plt.savefig('./figs/temp/pt' + str(i) + '.png')
    # print 'dun'

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # stop doin some heatmaps
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    plt.show()
    # eigvals.dump('./data/tempeigvals.pkl')
    # eigvects.dump('./data/tempeigvects.pkl')
    


def uniform_sampling_demo():
    """uniform sampling of non-monotonic surface"""
    
    a0 = 1.  # slope of 1st and 3rd segment
    b0 = -0.5  # slope of 2nd segment
    # for b0 in [0, -0.1, -0.25, -0.5, -0.75, -1]:#[-1, -5, -10, -100]:#[0.0, 0.25, 0.50, 0.75, 1, 1.5, 2, 5, 10, 100]:
    # b0 = np.pi/2 - np.arctan(b0)
    lam_true = 1

    a = a0/np.sqrt(lam_true)
    b = b0/np.sqrt(lam_true)
    ra = np.sqrt(1+a*a)
    rb = np.sqrt(1+b*b)  # arc-length factors on segments
    L = 2*ra/3 + rb/3  # total graph length
    N = 200  # no. of grid intervals of length ds
    ds = L/N  # arc length of grid interval

    xy = np.zeros((N+1,2))  # initialization of (x,y)-values
    i = 0  # counter initialization

    # 1st segment
    while xy[i,0] < 1./3:
        xy[i+1] = xy[i] + [ds/ra,ds*np.sqrt(1-1/(ra*ra))*np.sign(a)] 
        i = i+1 

    # 1st point of 2nd segment
    xy[i,0] = 1./3 + (ds - (1./3-xy[i-1,0])*ra)/rb
    xy[i,1] = xy[i-1,1] + a*(1./3 - xy[i-1,0]) + b*(xy[i,0] - 1./3) 

    # 2nd segment
    while xy[i,0] < 2./3:
        xy[i+1] = xy[i] + [ds/rb,ds*np.sqrt(1-1/(rb*rb))*np.sign(b)] 
        i = i+1 

    # 1st point of 3rd segment
    xy[i,0] = 2./3 + (ds - (2./3-xy[i-1,0])*rb)/ra 
    xy[i,1] = xy[i-1,1] + b*(2./3 - xy[i-1,0]) + a*(xy[i,0] - 2./3) 

    # 3rd segment
    while xy[i,0] < (1-1e-12):
        xy[i+1] = xy[i] + [ds/ra,ds*np.sqrt(1-1/(ra*ra))*np.sign(a)]
        i = i+1

    # print '%1.16f, %1.16f' % (xy[-2,0], xy[-1,0])

    # actual arc lengths (measured in ds-units) two entries should be slightly
    # under one, due because we measure along chord instead of through corner.

    ds_ratio = np.empty(N)
    for i in range(N):
        ds_ratio[i] = np.sqrt(np.power(xy[i+1,0]-xy[i,0], 2) + np.power(xy[i+1,1]-xy[i,1], 2))/ds 

    # # print ds_ratio
    plt.plot(xy[:,0], xy[:,1], lw=5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\gamma(x)$')
    plt.xlim((0,1))
    plt.show()


    phi1 = np.zeros(N+1)
    i = 0
    while xy[i,0] < 1./3:
        phi1[i] = np.cos(np.pi*ra*xy[i,0]/L)
        i = i + 1
    while xy[i,0] < 2./3:
        phi1[i] = np.cos(np.pi*rb*(xy[i,0] - (1 - ra/rb)/3)/L)
        i = i + 1
    for x in xy[i:]:
    # while xy[i,0] <= 1:
        phi1[i] = np.cos(np.pi*ra*(xy[i,0] - (1 - rb/ra)/3)/L)
        i = i + 1



    # do some dmaps
    npts = N + 1
    xvals = xy[:,0]
    yvals = xy[:,1]


    M = np.empty((npts,npts))
    lam = 1.0
    # for eps in [0.5*ds, ds, 3*ds, 5*ds]:# = 2*ds
    eps = 2*ds
    for i in range(npts):
        x1 = xvals[i]
        y1 = yvals[i]
        for j in range(npts):
            x2 = xvals[j]
            y2 = yvals[j]
            M[i,j] = np.exp(-(np.power(x1-x2, 2) + np.power(y1 - y2, 2)/lam)/eps)
    # normalize by degree
    D_half_inv = np.identity(npts)/np.sqrt(np.sum(M,1))

    # # find eigendecomp
    k = 50
    Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k) # eigsh (eigs hermitian)
    Meigvects = np.dot(D_half_inv, Meigvects)
    sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
    Meigvals = Meigvals[sorted_indices]
    Meigvects = Meigvects[:,sorted_indices]
    Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)

    gsize = 2
    gspec = gs.GridSpec(gsize,gsize)
    fig = plt.figure(figsize=(36, 20))
    ax = fig.add_subplot(gspec[0,:])
    ax.scatter(xy[:,0], xy[:,1], c=np.sign(Meigvects[0,1])*Meigvects[:,1], s=100)
    ax.set_ylabel(r'$\gamma(x)$')

    ax2 = fig.add_subplot(gspec[1,:], sharex=ax)
    ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[0,1]*phi1, c='r', label='Scaled cos(x)')
    ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[:,1], c='b', label='DMAPS')
    ax2.legend(fontsize=36)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\Phi_1$')
    ax2.set_ylim((-0.1, 0.1))

    # add some vertical lines

    # yrange = ax.get_ylim()
    # yrange = np.linspace(yrange[0], yrange[1], 100)
    # ax.plot(np.ones(100)/3., yrange, color='k')
    # ax.plot(2*np.ones(100)/3., yrange, color='k')
    # ax2.plot(np.ones(100)/3., yrange, color='k')
    # ax2.plot(2*np.ones(100)/3., yrange, color='k')

    ax.set_xlim((0, np.max(xy[:,0])))

    ax.set_ylim(bottom=0)

    lamstr = '%1.1f' % lam_true
    bstr = '%1.2f' % b0
    title = r' $b=$' + bstr
    ax.set_title(title)

    fig.subplots_adjust(bottom=0.15)

    plt.show()
    eps_str = '%1.2f' % eps
    # plt.savefig('./figs/slope-test/b' + str(b) + 'eps' + eps_str + '.png')


    xy.dump('./data/tempxy' + str(b) + '.pkl')    
    Meigvects.dump('./data/tempeigvects' + str(b) + '.pkl')
    Meigvals.dump('./data/tempeigvals' + str(b) + '.pkl')

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # # # method the second
    # # ds = 0.005
    # alpha = 1
    # beta = -3
    # ndsalpha = np.ceil(np.sqrt(1 + alpha*alpha)/(3*ds))
    # nptsalpha = ndsalpha
    # ndsbeta = np.ceil(np.sqrt(1 + beta*beta)/(3*ds))
    # nptsbeta = ndsbeta

    # dxalpha = ds/np.sqrt(1 + alpha*alpha)
    # dxbeta = ds/np.sqrt(1 + beta*beta)

    # xalpha1 = dxalpha*np.arange(nptsalpha + 1)
    # xbeta = xalpha1[-1] + dxbeta*np.arange(1,nptsbeta+1)
    # xalpha2 = xbeta[-1] + dxalpha*np.arange(1,nptsalpha+1)

    # x = np.empty(2*nptsalpha + nptsbeta + 1)
    # y = np.empty(2*nptsalpha + nptsbeta + 1)
    # x[:nptsalpha+1] = xalpha1
    # x[1+nptsalpha:1+nptsalpha + nptsbeta] = xbeta
    # x[1+nptsalpha + nptsbeta:1+2*nptsalpha + nptsbeta] = xalpha2

    # y[:nptsalpha+1] = alpha*x[:nptsalpha+1]
    # c = (alpha - beta)*nptsalpha*dxalpha # y - beta*x
    # y[1+nptsalpha:1+nptsalpha + nptsbeta] = beta*x[1+nptsalpha:1+nptsalpha + nptsbeta] + c
    # c = (beta - alpha)*xbeta[-1] + c # y - alpha x
    # y[1+nptsalpha + nptsbeta:1+2*nptsalpha + nptsbeta] = alpha*x[1+nptsalpha + nptsbeta:1+2*nptsalpha + nptsbeta] + c

    # # check segment endpoints (should be 1/3, 2/3, 1)
    # e1 = xalpha1[-1]
    # e2 = xbeta[-1]
    # e3 = xalpha2[-1]
    # print e1, e2, e3
    # print dxalpha, dxbeta
    # print np.abs(e1 - 1./3) < dxalpha, np.abs(e2 - 2./3) < (dxbeta + dxalpha), np.abs(e3 - 1.) < (2*dxalpha + dxbeta)
    # npts = x.shape[0]
    # ds_ratio = np.empty(npts-1)
    # for i in range(npts-1):
    #     ds_ratio[i] = np.sqrt(np.power(x[i+1]-x[i], 2) + np.power(y[i+1]-y[i], 2))/ds 

    # print ds_ratio

    # plt.scatter(x, y)
    # plt.scatter(xy[:,0], xy[:,1], c='r')
    # plt.show()

    # # # # end method the second

    # npts = 6+1
    # npts_per_section = (npts-1)/3
    # L = 2*np.sqrt(1 + alpha*alpha)/3. + np.sqrt(1 + beta*beta)/3.
    # ds = L/(npts - 1)
    # xvals = np.zeros(npts)
    # print dxalpha
    # print dxbeta
    # xvals[:npts_per_section+1] = dxalpha*np.arange(npts_per_section+1)
    # xvals[1+npts_per_section:1+2*npts_per_section] = xvals[npts_per_section] + dxbeta*np.arange(1,npts_per_section+1)
    # xvals[1+2*npts_per_section:1+3*npts_per_section] = xvals[2*npts_per_section] + dxalpha*np.arange(1,npts_per_section+1)
    
    # for i in range(1,npts_per_section):
    #     xvals[i] = xvals[i-1] + dxalpha
    # for i in range(1+npts_per_section, 1+2*npts_per_section):
    #     xvals[i] = xvals[i-1] + dxbeta
    # for i in range(1+2*npts_per_section, 1+3*npts_per_section):
    #     xvals[i] = xvals[i-1] + dxalpha
    # yvals = np.empty(npts)
    # yvals[:npts_per_section] = alpha*xvals[:npts_per_section]
    # c = yvals[npts_per_section-1] - beta*xvals[npts_per_section-1]
    # yvals[npts_per_section:2*npts_per_section] = beta*xvals[npts_per_section:2*npts_per_section] + c
    # c = yvals[2*npts_per_section-1] - alpha*xvals[2*npts_per_section-1]
    # yvals[2*npts_per_section:] = alpha*xvals[2*npts_per_section:] + c
    # print xvals

    # # sanity check
    # xalphadensity = np.sqrt(np.power(xvals[0] - xvals[npts_per_section-1], 2) + np.power(yvals[0] - yvals[npts_per_section-1], 2))/30.
    # xbetadensity = np.sqrt(np.power(xvals[npts_per_section] - xvals[2*npts_per_section-1], 2) + np.power(yvals[npts_per_section] - yvals[2*npts_per_section-1], 2))/30.
    # print xalphadensity, xbetadensity

    

    # M = np.empty((npts,npts))
    # lam = 0.5
    # eps = 2*ds
    # for i in range(npts):
    #     x1 = xvals[i]
    #     y1 = yvals[i]
    #     for j in range(npts):
    #         x2 = xvals[j]
    #         y2 = yvals[j]
    #         M[i,j] = np.exp(-(np.power(x1-x2, 2) + np.power(y1 - y2, 2)/lam)/eps)
    # # normalize by degree
    # D_half_inv = np.identity(npts)/np.sqrt(np.sum(M,1))
    
    # # # find eigendecomp
    # k = 50
    # Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k) # eigsh (eigs hermitian)
    # Meigvects = np.dot(D_half_inv, Meigvects)
    # sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
    # Meigvals = Meigvals[sorted_indices]
    # Meigvects = Meigvects[:,sorted_indices]
    # Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)

    # Meigvects.dump('./data/tempeigvects.pkl')    
    # Meigvals.dump('./data/tempeigvals.pkl')


def tempfig3():
    n = 100
    y = np.linspace(0,1,n)
    # data = y # 2*np.arcsin(y)/np.pi
    beta = 1
    # data = ((beta + 1)/beta - np.sqrt(np.power((beta + 1)/beta, 2) - 4*y/beta))/2
    data = ((beta + 1)/beta - np.sqrt(np.power((beta + 1)/beta, 2) - 4*y/beta))/8
    xdata = np.empty((4*n))
    xdata[:n] = data
    xdata[n:2*n] = 0.5 - data
    xdata[2*n:3*n] = 0.5 + data
    xdata[3*n:4*n] = 1 - data

    ydata = np.empty((4*n))
    ydata[:n] = data
    ydata[n:2*n] = 0.25 + data[::-1]
    ydata[2*n:3*n] = 0.5 + data
    ydata[3*n:4*n] = 0.75 + data[::-1]

    
    plt.scatter(ydata, np.ones(4*n))
    plt.scatter(xdata, np.ones(4*n), c='r')
    plt.show()
    
    eps = 2*np.max(data[1:] - data[:-1])
    eigvals, eigvects = dmaps.embed_data(xdata, k=5, epsilon=eps)
    eigvects.dump('./data/tempeigvects.pkl')    
    eigvals.dump('./data/tempeigvals.pkl')
    plt.scatter(xdata, eigvects[:,1])
    plt.scatter(xdata, eigvects[0,1]*np.cos(np.pi*xdata), c='r')
    plt.show()



def lambda_fig():
    """Reveals effect of different values of lambda on the resulting eigenfunctions"""

    n = 1000 # number of grid points

    M = np.empty((n,n))
    eps = 0.01
    for i in range(n):
        x1 = xdata[i]
        for j in range(n):
            x2 = xdata[j]
            M[i,j] = np.exp(-(np.power(x1-x2, 2) + np.power(Gamma(x1) - Gamma(x2), 2)/lam)/eps)
    # normalize by degree
    D_half_inv = np.identity(n)/np.sqrt(np.sum(M,1))
    
    # # find eigendecomp
    k = 50
    
    Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k) # eigsh (eigs hermitian)
    Meigvects = np.dot(D_half_inv, Meigvects)
    sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
    Meigvals = Meigvals[sorted_indices]
    Meigvects = Meigvects[:,sorted_indices]
    Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)


    L = np.zeros((n,n)) # matrix approx to operator
    # A = -1.0 # left boundary
    # B = 1.0 # right boundary
    h = (B - A)/(n-1) # corresponding gridsize
    h_squared = np.power(h, 2)
    xdata = np.linspace(A, B, n)

    # visualize dataset
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xdata, Gamma(xdata))
    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.set_title('Curve on which to apply new kernel')
    plt.show()

    # # loop over different values of lambda, storing top 'k' eigenvectors for each
    k = 4
    nlams = 50
    lams = np.logspace(loglam_min,loglam_max,nlams)
    eigvects = np.empty((nlams, k, n))
    for j, lam in enumerate(lams):
        # # set up the matrix approximation to the operator
        # use left BCs
        L[0,0] = -2*lam/((lam + np.power(dGamma(A),2))*h_squared)
        L[0,1] = 2*lam/((lam + np.power(dGamma(A),2))*h_squared)
        # use right BCs
        L[n-1,n-2] = 2*lam/((lam + np.power(dGamma(B),2))*h_squared)
        L[n-1,n-1] = -2*lam/((lam + np.power(dGamma(B),2))*h_squared)
        # set approximation at interior points
        for i in range(1,n-1):
            x = xdata[i]
            denom = lam + np.power(dGamma(x), 2)
            L[i,i-1] = lam/(denom*h_squared) + 3*lam*dGamma(x)*ddGama(x)/(np.power(denom, 2)*2*h)
            L[i,i] = -2*lam/(denom*h_squared)
            L[i,i+1] = lam/(denom*h_squared) - 3*lam*dGamma(x)*ddGama(x)/(np.power(denom, 2)*2*h)
    
        # # find eigendecomp
        Leigvals, Leigvects = np.linalg.eig(L) # eigs (plain eigs)
        sorted_indices = np.argsort(np.abs(Leigvals))
        Leigvals = Leigvals[sorted_indices]
        Leigvects = Leigvects[:,sorted_indices]
        Leigvects = Leigvects/np.linalg.norm(Leigvects, axis=0)

        eigvects[j] = Leigvects[:,1:k+1].T

    eigvects.dump('./data/eigenfns-vs-lambda.pkl')

def lambda_fig_plot():
    """Plots data form 'lambda_fig'"""
    
    eigvects = np.load('./data/eigenfns-vs-lambda.pkl')
    nlams = eigvects.shape[0]
    lams = np.logspace(loglam_min,loglam_max,nlams)
    k = eigvects.shape[1]
    n = 100
    slice = eigvects.shape[2]/n

    # # set up plotting stuff
    gsize = 10
    gspec = gs.GridSpec(gsize,gsize)
    fig = plt.figure()
    # ax = fig.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax = fig.add_subplot(gspec[:,:gsize-1])

    # set up colornorm
    colornorm = colors.Normalize(vmin=loglam_min, vmax=loglam_max) # colors.Normalize(vmin=np.min(embeddings), vmax=np.max(embeddings))
    colormap = cm.ScalarMappable(norm=colornorm, cmap='viridis')

    for i in range(nlams):
        if eigvects[i,0,0] > 0:
            eigvects[i,0] = -eigvects[i,0]
        # ax.scatter(np.log10(lams[i])*np.ones(n), np.linspace(-1, 1, n), eigvects[i,0,::slice], c=colormap.to_rgba(np.log10(lams[i])))
        ax.plot(np.linspace(A, B, n), eigvects[i,0,::slice], c=colormap.to_rgba(np.log10(lams[i])))

    # axes labels
    # ax.set_xlabel('\n\n' + r'$\log(\lambda)$')
    # ax.set_ylabel('\n\n' + r'$x$')
    # ax.set_zlabel('\n\n' + r'$\Phi_2$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\Phi_2$')

    # colorbar
    ax_cb = fig.add_subplot(gspec[:,gsize-1])
    colorbar.ColorbarBase(ax_cb, cmap='viridis', norm=colornorm, label=r'$\log(\lambda)$')#, ticks=cb_ticks)
    # plt.tight_layout()
    plt.show()
    

def approx_eigenfn_test():
    """Compares the approximate eignfns obtained by 1.) discretizing the underlying linear operator and 2.) finding an eigendecomposition of the modified DMAPS kernel. The kernel under investigation is given by k(x,y) = exp(-1/eps ((x-y)^2 + 1/lam (f(x) - f(y))^2)) where f(x) is the model prediction at x. Here f(x) = a x^2 is hard-coded (see William Leeb's notes from 2016/03/14"""
    n = 1000 # number of grid points
    lam = 1e-3 # scaling for second term in kernel
    L = np.zeros((n,n)) # matrix approx to operator
    # A = -0.95 # left boundary
    # B = 0.95 # right boundary
    h = (B - A)/(n-1) # corresponding gridsize
    h_squared = np.power(h, 2)
    xdata = np.linspace(A, B, n)

    # # visualize dataset
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xdata, Gamma(xdata))
    # ax.set_title('Curve on which to apply new kernel')
    # plt.show()

    t1 = time()
    # # set up the matrix approximation to the operator
    # use left BCs
    L[0,0] = -2*lam/((lam + np.power(dGamma(A),2))*h_squared)
    L[0,1] = 2*lam/((lam + np.power(dGamma(A),2))*h_squared)
    # use right BCs
    L[n-1,n-2] = 2*lam/((lam + np.power(dGamma(B),2))*h_squared)
    L[n-1,n-1] = -2*lam/((lam + np.power(dGamma(B),2))*h_squared)
    # set approximation at interior points
    for i in range(1,n-1):
        x = xdata[i]
        denom = lam + np.power(dGamma(x), 2)
        L[i,i-1] = lam/(denom*h_squared) + 3*lam*dGamma(x)*ddGama(x)/(np.power(denom, 2)*2*h)
        L[i,i] = -2*lam/(denom*h_squared)
        L[i,i+1] = lam/(denom*h_squared) - 3*lam*dGamma(x)*ddGama(x)/(np.power(denom, 2)*2*h)
    
    # # find eigendecomp
    Leigvals, Leigvects = np.linalg.eig(L) # eigs (plain eigs)
    sorted_indices = np.argsort(np.abs(Leigvals))
    Leigvals = Leigvals[sorted_indices]
    Leigvects = Leigvects[:,sorted_indices]
    Leigvects = Leigvects/np.linalg.norm(Leigvects, axis=0)

    print 'operator decomp took:', time() - t1, 's'
    t1 = time()
    # # find eigendecomp of kernel
    # remember, f(x) = a x^2
    M = np.empty((n,n))
    eps = 0.01
    for i in range(n):
        x1 = xdata[i]
        for j in range(n):
            x2 = xdata[j]
            M[i,j] = np.exp(-(np.power(x1-x2, 2) + np.power(Gamma(x1) - Gamma(x2), 2)/lam)/eps)
    # normalize by degree
    D_half_inv = np.identity(n)/np.sqrt(np.sum(M,1))
    
    # # find eigendecomp
    k = 50
    
    print lam, eps
    Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k) # eigsh (eigs hermitian)
    Meigvects = np.dot(D_half_inv, Meigvects)
    sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
    Meigvals = Meigvals[sorted_indices]
    Meigvects = Meigvects[:,sorted_indices]
    Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)

    print 'kernel matrix decomp took:', time() - t1, 's'

    # # plot eigenvectors from different techniques on same fig.
    for i in range(1,k):
        # flip eigevect if necessary
        if Leigvects[0,i] > 0:
            Leigvects[:,i] = -Leigvects[:,i]
        if Meigvects[0,i] > 0:
            Meigvects[:,i] = -Meigvects[:,i]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata, Leigvects[:,i], c='b', label='operator discretization')
        ax.plot(xdata, Meigvects[:,i], c='r', label='kernel eigendecomposition')
        # ax.legend(loc=8)
        ax.set_title(r'$\Phi_{' + str(i) + '}$')
        plt.savefig('./figs/eig' + str(i+1) + '.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(Meigvals.shape[0], Meigvals))
    plt.savefig('./figs/eigvals.png')
    # plt.show()



def main():
    # lambda_fig()
    # kernel_lam_fig()
    # lambda_fig_plot()
    # uniform_sampling_demo()
    # nonuniform_sampling_lambda_fig()
    disconnected_parallel_lines()
    # tempfig3()
    # approx_eigenfn_test()

if __name__=='__main__':
    main()
