"""Uses modules :py:mod:`MM` and :py:mod:`Hessian` to investigate the sloppiness Michaelis Menten parameters"""

import MM
import dmaps
import dmaps_kernel
import plot_dmaps
from solarized import solarize
from Hessian import hessian
import numpy as np
from sympy import Function, dsolve, Eq, Derivative, symbols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def dmap_sloppy_params():
    """Perform DMAPs on a set of sloppy parameters, attempt to capture sloppy directions"""
    # set up true system
    K = 2.0; V = 1.0; St = 2.0; epsilon = 1e-3; kappa = 10.0 # from Antonios' writeup
    params = np.array((K, V, St, epsilon, kappa))
    transform_id = 't2'
    sigma = St/K 
    # set init concentrations
    S0 = K*sigma; C0 = 0.0; P0 = 0.0 # init concentrations
    Cs0 = np.array((S0, C0, P0))
    # set times at which to collect data
    tscale = (sigma + 1)*K/V # timescale of slow evolution
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    # use these params, concentrations and times to define the MM system
    MM_system = MM.MM_System(Cs0, times, params, transform_id)
    
    # data generation, if saved data exists, use it. otherwise generate in this script
    if os.path.isfile('./sloppy_params.csv'):
        print '******************************\nLoading data from: ./sloppy_params.csv\n******************************'
        kept_params = np.genfromtxt('./sloppy_params.csv', delimiter=',')
    else:
        # sample params noisily in 5d space, 10 points per axis for a total of 10e5 points (too many?)
        # each param at K = 2.0; V = 1.0; St = 2.0; epsilon = 1e-3; kappa = 10.0

        npts_per_axis = 100

        Ks = 2*np.logspace(-4, 1, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis)) # K*np.ones(npts_per_axis)
        Vs = np.logspace(-4, 1, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis))
        # Sts = St*np.ones(npts_per_axis) # np.logspace(-4, 4, npts_per_axis)*(1 + np.random.normal(size=npts_per_axis))
        # epsilons = np.logspace(-7, 1, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis))
        # kappas = np.logspace(-3, 5, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis))
        # param_sets = [Ks, Vs, Sts, epsilons, kappas]
        param_sets = [Ks, Vs]
        nparams = len(param_sets)
        npts = np.power(npts_per_axis, nparams)
        index = np.empty(nparams)
        powers = np.array([np.power(npts_per_axis, i) for i in range(nparams)]) # powers of nparams, e.g. 1, 5, 25, ...
        tol = 5
        kept_params = np.empty((npts, nparams+1)) # storage for all possible params and their respective ob. fn. evaluations
        kept_npts = 0 # number of parameter sets that fall within tolerated ob. fn. range
        of_evals = np.empty(npts)
        np.seterr(all='ignore')
        for i in range(npts):
            # probably a more efficient method of calculating the current index instead of performing 'nparams' calculations every time
            index = i/powers%npts_per_axis
            params = np.array([param_sets[j][index[j]] for j in range(nparams)])

            # params = np.array((params[0], params[1], 2.0, 1e-3, 10.0))

            # record param set and ob. fn. value if below tolerance
            ob_fn_eval = MM_system.of(np.array((params[0], params[1], 2.0, 1e-3, 10.0)))#params)
            of_evals[i] = ob_fn_eval
            if ob_fn_eval < tol:
                kept_params[kept_npts,:-1] = np.log(params)
                kept_params[kept_npts,-1] = ob_fn_eval
                kept_npts += 1

        kept_params = kept_params[:kept_npts]
        np.savetxt('./sloppy_params.csv', kept_params, delimiter=',')

        print '************************************************************'
        print 'generated', kept_npts, 'new points with min obj. fn. value of', np.min(kept_params[-1,:])
        print '************************************************************'

    # from analysis below, epsilon values around 80 - 100 should work
    # nepsilons = 10
    # epsilons = np.logspace(0, 4, nepsilons)
    # kernels = [dmaps_kernel.custom_kernel(epsilon) for epsilon in epsilons]
    # plot_dmaps.kernel_plot(kernels, epsilons, kept_params)
    epsilon = 2.0
    k = 20
    eigvals, eigvects = dmaps.embed_data_customkernel(kept_params, k, dmaps_kernel.custom_kernel(epsilon))
    plot_dmaps.plot_embeddings(eigvects, eigvals, k, plot_3d=True, color=kept_params[:,-1])
    
    
        
        

def check_sloppiness():
    """Checks for sloppiness in the model by printing the Hessian's eigenvalues when evaluated at the minimum of least-squares objective fn."""
    # set parameters as per suggestions in paper
    # K = 1.0; V = 1.0; sigma = 1.0; epsilon = 1e-2; kappa = 10.0 # used in first param. transform/ation, now use St instead of sigma
    # params = np.array((K, V, sigma, epsilon, kappa)) # again, from old transformation
    # transform_id = 't1'

    K = 2.0; V = 1.0; St = 2.0; epsilon = 1e-3; kappa = 10.0 # from Antonios' writeup
    params = np.array((K, V, St, epsilon, kappa))
    transform_id = 't2'
    sigma = St/K 

    # set init concentrations
    S0 = K*sigma; C0 = 0.0; P0 = 0.0 # init concentrations
    Cs0 = np.array((S0, C0, P0))
    # set times at which to collect data
    tscale = (sigma + 1)*K/V # timescale of slow evolution
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    # use these params, concentrations and times to define the MM system
    MM_system = MM.MM_System(Cs0, times, params, transform_id)

    # test stepsize's effect on hessian approx as the calculation seems prone to numerical errors
    nhvals = 20
    hvals = np.logspace(-7,-4, nhvals) # numerical delta_h values that will be used in the finite-difference approximations
    m = 3 # the number of parameters of interest, i.e. '3' if only looking at the effects of K, V and sigma, otherwise '5' to look at effects of all parameters on the obj. fn.
    eigs = np.empty((nhvals, m))
    for i in range(nhvals):
        hessian_eval = hessian(MM_system.of, params, h=hvals[i])
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
        # save in special directory if exists
        if os.path.isdir('./figs/hessian'):
            plt.savefig('./figs/hessian/all_eigs' + str(i) + '.png')
        else:
            plt.savefig('./all_eigs' + str(i) + '.png')

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
    # test_combinator()
    dmap_sloppy_params()
    # check_sloppiness()
