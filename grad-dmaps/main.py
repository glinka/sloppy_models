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

# def ddGama(x):
#     return 6*a*x

# Gamma(x) = 0.25*(np.log(x+1) - np.log(1-x) - 2*x)
def Gamma(x):
    return 0.25*(np.log(x+1) - np.log(1-x) - 2*x)

def dGamma(x):
    return 0.5*np.power(x,2)/(1 - np.power(x, 2))

def ddGama(x):
    return x/np.power(1 - np.power(x, 2), 2)


def approx_eigenfn_test():
    """Compares the approximate eignfns obtained by 1.) discretizing the underlying linear operator and 2.) finding an eigendecomposition of the modified DMAPS kernel. The kernel under investigation is given by k(x,y) = exp(-1/eps ((x-y)^2 + 1/lam (f(x) - f(y))^2)) where f(x) is the model prediction at x. Here f(x) = a x^2 is hard-coded (see William Leeb's notes from 2016/03/14"""
    n = 1000 # number of grid points
    lam = 0.05 # scaling for second term in kernel
    L = np.zeros((n,n)) # matrix approx to operator
    A = -0.95 # left boundary
    B = 0.95 # right boundary
    h = (B - A)/(n-1) # corresponding gridsize
    h_squared = np.power(h, 2)
    xdata = np.linspace(A, B, n)

    # # visualize dataset
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xdata, Gamma(xdata))
    ax.set_title('Curve on which to apply new kernel')
    plt.show()

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
    Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k) # eigsh (eigs hermitian)
    Meigvects = np.dot(D_half_inv, Meigvects)
    sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
    Meigvals = Meigvals[sorted_indices]
    Meigvects = Meigvects[:,sorted_indices]
    Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)

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
        ax.legend(loc=8)
        ax.set_title(r'$\Phi_{' + str(i) + '}$')
        plt.savefig('./figs/eig' + str(i+1) + '.png')
    # plt.show()



def main():
    approx_eigenfn_test()

if __name__=='__main__':
    main()
