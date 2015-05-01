"""Uses modules :py:mod:`MM` and :py:mod:`Hessian` to investigate the sloppiness Michaelis Menten parameters"""

import MM
from Hessian import hessian
import numpy as np
from sympy import Function, dsolve, Eq, Derivative, symbols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_sloppiness():
    """Checks for sloppiness in the model by printing the Hessian's eigenvalues when evaluated at the minimum of least-squares objective fn."""
    # set parameters as per suggestions in paper
    K = 1.0; V = 1.0; sigma = 1.0; epsilon = 1e-2; kappa = 10.0
    S0 = K*sigma; C0 = 0.0; P0 = 0.0
    tscale = (sigma + 1)*K/V
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    params = np.array((K, V, sigma, epsilon, kappa))
    Cs0 = np.array((S0, C0, P0))
    # generate data to be fit
    data = MM.gen_profile(Cs0, times, params)
    # # visualize data
    # plt.hold(True)
    # for i in range(3):
    #     plt.plot(times, data[:,i])
    # plt.show()
    # set data in objective function
    enzyme_of = MM.EnzymeOF(data, times)
    # test stepsize's effect on hessian approx as the calculation seems prone to numerical errors
    nhvals = 20
    hvals = np.logspace(-5,-3, nhvals) # numerical delta_h values that will be used in the finite-difference approximations
    m = 5 # the number of parameters of interest, i.e. '3' if only looking at the effects of K, V and sigma, otherwise '5' to look at effects of all parameters on the obj. fn.
    eigs = np.empty((nhvals, m))
    for i in range(nhvals):
        hessian_eval = hessian(enzyme_of.of_t, params, h=hvals[i])
        eigs[i] = np.sort(np.linalg.eigvalsh(hessian_eval[:m,:m]))

    # plot output
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(m):
        ax.cla()
        ax.plot(hvals, eigs[:,i])
        ax.yaxis.get_children()[1].set_size(16)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
        ax.set_xscale('log')
        ax.set_ylabel('Eigenvalue ' + str(i + 1), fontsize=18)
        ax.set_xlabel('Finite difference approximation stepsize', fontsize=16)
        plt.tight_layout()
        plt.savefig('../figs/hessian/all_eigs' + str(i) + '.png')

    # # use transformed ob. fn. here, of_t, which is given in terms of the original parameters
    # hessian_eval = hessian(enzyme_of.of_t, params, h=0.007)
    # print 'Results for transformed problem:\n'
    # print 'eigs of 3x3 hessian, hopefully not sloppy:\n', np.linalg.eigvals(hessian_eval[:3,:3])
    # print 'eigs of full 5x5 hessian, hopefully has two sloppy directions:\n', np.linalg.eigvals(hessian_eval)

    # # now use original variables
    # hessian_eval = hessian(enzyme_of.of, transform_params(params), h=0.007)
    # print '\nResults for original problem:\n'
    # print 'eigs of 3x3 hessian, hopefully not sloppy:\n', np.linalg.eigvals(hessian_eval[:3,:3])
    # print 'eigs of full 5x5 hessian, hopefully has two sloppy directions:\n', np.linalg.eigvals(hessian_eval)

    # # plot the 3d eigenvectors to examine whether they lie along specific axes
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # eigvals, eigvects = np.linalg.eigh(hessian_eval[:m,:m])
    # for i in range(m):
    #     xyz = np.vstack((np.zeros(3), eigvects[:,i]/np.linalg.norm(eigvects[:,i])))
    #     ax.plot(xyz[:,0], xyz[:,1], xyz[:,2])
    # ax.set_xlabel('K')
    # ax.set_ylabel('V')
    # ax.set_zlabel(r'$\sigma$')
    # plt.show()

if __name__=='__main__':
    check_sloppiness()
