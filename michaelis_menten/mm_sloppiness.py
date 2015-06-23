"""Uses modules :py:mod:`MM` and :py:mod:`Hessian` to investigate the sloppiness Michaelis Menten parameters"""

import MM
import dmaps
import dmaps_kernels
import plot_dmaps
from Hessian import hessian
from solarized import solarize
import numpy as np
from sympy import Function, dsolve, Eq, Derivative, symbols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# switch to nicer color scheme
solarize()

def dmap_sloppy_params():
    """Perform DMAPs on a set of sloppy parameters, attempt to capture sloppy directions"""
    # set up true system
    K = 2.0; V = 1.0; St = 2.0; epsilon = 1e-3; kappa = 10.0 # from Antonios' writeup
    true_params = np.array((K, V, St, epsilon, kappa))
    nparams = true_params.shape[0]
    transform_id = 't2'
    sigma = St/K 
    # set init concentrations
    S0 = St; C0 = 0.0; P0 = 0.0 # init concentrations
    Cs0 = np.array((S0, C0, P0))
    # set times at which to collect data
    tscale = (sigma + 1)*K/V # timescale of slow evolution
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    # use these params, concentrations and times to define the MM system
    MM_system = MM.MM_System(Cs0, times, true_params, transform_id)

    # visualize concentration profiles
    conc_profiles = MM_system.gen_profile(Cs0, times, true_params)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times, conc_profiles[:,0], label='S')
    ax.plot(times, conc_profiles[:,1], label='C')
    ax.plot(times, conc_profiles[:,2], label='P')
    ax.set_xlabel('times')
    ax.set_ylabel('concentration (potentially dimensionless)')
    ax.legend(loc=2)
    plt.show(fig)
    
    
    # data generation, if saved data exists, use it. otherwise generate in this script
    if os.path.isfile('./data/input/sloppy_params.csv'):
        kept_params = np.genfromtxt('./data/input/sloppy_params.csv', delimiter=',')
        print '************************************************************\nLoaded data from: ./data/input/sloppy_params.csv\n************************************************************'
    else:
        # sample params noisily in 5d space, 10 points per axis for a total of 10e5 points (too many?)
        # center each param at K = 2.0; V = 1.0; St = 2.0; epsilon = 1e-3; kappa = 10.0
        npts_per_axis = 10
        Ks = 2*np.logspace(-4, 1, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis)) # K*np.ones(npts_per_axis)
        Vs = np.logspace(-4, 1, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis))
        # Sts = St*np.ones(npts_per_axis) # np.logspace(-4, 4, npts_per_axis)*(1 + np.random.normal(size=npts_per_axis))
        epsilons = np.logspace(-7, 1, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis))
        # kappas = np.logspace(-3, 5, npts_per_axis)#*(1 + np.random.normal(size=npts_per_axis))
        # param_sets = [Ks, Vs, Sts, epsilons, kappas]
        param_sets = [Ks, Vs, epsilons]
        ntest_params = len(param_sets)
        npts = np.power(npts_per_axis, ntest_params)
        index = np.empty(ntest_params)
        powers = np.array([np.power(npts_per_axis, i) for i in range(ntest_params)]) # powers of ntest_params, e.g. 1, 5, 25, ...
        
        tol = 5
        kept_params = np.empty((npts, ntest_params+1)) # storage for all possible params and their respective ob. fn. evaluations
        kept_npts = 0 # number of parameter sets that fall within tolerated ob. fn. range
        np.seterr(all='ignore')
        params = np.empty(nparams)

        for i in range(npts):
            # probably a more efficient method of calculating the current index instead of performing 'ntest_params' calculations every time
            index = i/powers%npts_per_axis
            new_params = np.array([param_sets[j][index[j]] for j in range(ntest_params)])
            params[:ntest_params] = new_params
            params[ntest_params:] = true_params[ntest_params:]
            # record param set and ob. fn. value if below tolerance
            ob_fn_eval = MM_system.of(params)
            if ob_fn_eval < tol and ob_fn_eval is not False:
                kept_params[kept_npts,:-1] = np.log(new_params)
                kept_params[kept_npts,-1] = ob_fn_eval
                kept_npts += 1

        kept_params = kept_params[:kept_npts]
        np.savetxt('./data/input/sloppy_params.csv', kept_params, delimiter=',')

        print '************************************************************'
        print 'generated', kept_npts, 'new points with min obj. fn. value of', np.min(kept_params[:,-1])
        print 'saved in ./data/input/sloppy_params.csv'
        print '************************************************************'

    # from analysis below, epsilon values around 1.0 should work
    nepsilons = 10
    epsilons = np.logspace(-2, 2, nepsilons)
    kernels = [dmaps_kernels.objective_function_kernel(epsilon) for epsilon in epsilons]
    plot_dmaps.kernel_plot(kernels, epsilons, kept_params, filename='./figs/dmaps/kernel_plot.png')
    # now actually perform dmaps
    epsilon = 1.0
    k = 20
    save_dir = './figs/dmaps/embeddings/'
    eigvals, eigvects = dmaps.embed_data_customkernel(kept_params, k, dmaps_kernels.objective_function_kernel(epsilon))
    plot_dmaps.plot_embeddings(eigvects, eigvals, k, plot_3d=True, color=kept_params[:,-1], folder=save_dir)
        

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
