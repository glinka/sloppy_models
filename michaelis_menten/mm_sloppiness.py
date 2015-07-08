"""Uses modules :py:mod:`MM` and :py:mod:`Hessian` to investigate the sloppiness Michaelis Menten parameters"""

import algorithms.CustomErrors as CustomErrors
import MM
import dmaps
import dmaps_kernels
import plot_dmaps
from algorithms.Derivates import hessian
import algorithms.PseudoArclengthContinuation as PSA
import MM_Specialization as MMS
from solarized import solarize
import numpy as np
from mpi4py import MPI
from sympy import Function, dsolve, Eq, Derivative, symbols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from collections import OrderedDict

# switch to nicer color scheme
solarize()

def mm_contours():
    """Finds contours of the MM objective function: either one-dimensional curves or two-dimensional surfaces. Distributes computation across processors with mpi4py"""
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # set up base system
    params = OrderedDict((('K',2.0), ('V',1.0), ('St',2.0), ('epsilon',1e-3), ('kappa',10.0))) # from Antonios' writeup
    true_params = np.array(params.values())
    nparams = true_params.shape[0]
    transform_id = 't2'
    state_params = ['K']
    continuation_param = 'V'
    # set init concentrations
    S0 = params['St']; C0 = 0.0; P0 = 0.0 # init concentrations
    Cs0 = np.array((S0, C0, P0))
    # set times at which to collect data
    tscale = (params['St'] + params['K'])/params['V'] # timescale of slow evolution
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    # use these params, concentrations and times to define the MM system
    # contour_val = 1e-7
    # mm_specialization = MMS.MM_Specialization(Cs0, times, true_params, transform_id, state_params, continuation_param, contour_val)

    # # visualize data
    # conc_profiles = mm_specialization.gen_profile(Cs0, times, true_params)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(times, conc_profiles[:,0], label='S')
    # ax.plot(times, conc_profiles[:,1], label='C')
    # ax.plot(times, conc_profiles[:,2], label='P')
    # ax.set_xlabel('times')
    # ax.set_ylabel('concentration (potentially dimensionless)')
    # ax.legend(loc=2)
    # plt.show(fig)

    # define range of V values over which to find level sets
    npts_per_proc = 2
    contour_vals = np.logspace(-7,-1,npts_per_proc*nprocs) # countour values of interest
    ds = 1e-5
    ncontinuation_steps = 20
    # branch = np.empty((ncontinuation_steps*npts_per_proc, 2))
    # current_index = 0
    for i, contour_val in enumerate(contour_vals[rank*npts_per_proc:(rank+1)*npts_per_proc]):
        mm_specialization = MMS.MM_Specialization(Cs0, times, true_params, transform_id, state_params, continuation_param, contour_val)
        psa_solver = PSA.PSA(mm_specialization.f, mm_specialization.f_gradient)
        try:
            branch = psa_solver.find_branch(np.array((params['K'],)), params['V']+1e-3, ds, ncontinuation_steps)
        except CustomErrors.PSAError:
            continue
        else:
            np.savetxt('./data/output/contour_' + str(contour_val) + '.csv', branch, delimiter=',')
        # # 'find_branch' may not actually find 'ncontinuation_steps' branch points, so only add those points that were successfully found
        # partial_branch = psa_solver.find_branch(np.array((params['K'],)), params['V'], ds, ncontinuation_steps)
        # nadditional_branch_pts = partial_branch.shape[0]
        # branch[current_index:current_index+nadditional_branch_pts] = partial_branch
        # current_index = current_index + nadditional_branch_pts
    # branch = branch[:current_index]
    # full_branch = comm.gather(branch, root=0)
    # if rank is 0:
    #     full_branch = np.concatenate(full_branch)
    #     full_npts = full_branch.shape[0]
    #     # create fileheader specifying which params were investigated, e.g. K=True,V=False,St=True,eps=True,kappa=False
    #     file_header = state_params[0] + '=True,' + continuation_param + '=True,transform_id=' + transform_id
    #     # remove trailing comma
    #     np.savetxt('./data/input/sloppy_contours' + str(full_npts) + '.csv', full_branch, delimiter=',', header=file_header, comments='')
    #     print '************************************************************'
    #     print 'generated', full_npts, 'new points with min obj. fn. value of', np.min(full_branch[:,-1])
    #     print 'saved in ./data/input/sloppy_params' + str(full_npts) + '.csv'
    #     print '************************************************************'
    
    # plt.plot(branch[:,0], branch[:,1])
    # plt.show()
    # err = 0
    # for pt in branch:
    #     err = err + np.abs(mm_specialization.f(np.array((pt[0],)), pt[1]))
    # print 'total error along branch:', err

def sample_sloppy_params():
    """Uses mpi4py to parallelize the collection of sloppy parameter sets. The current, naive method is to sample over a noisy grid of points, discarding those whose objective function evaluation exceeds the set tolerance. The resulting sloppy parameter combinations are saved in './data/input'"""

    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # set up true system, basis for future objective function evaluations
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

    # # visualize concentration profiles
    # conc_profiles = MM_system.gen_profile(Cs0, times, true_params)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(times, conc_profiles[:,0], label='S')
    # ax.plot(times, conc_profiles[:,1], label='C')
    # ax.plot(times, conc_profiles[:,2], label='P')
    # ax.set_xlabel('times')
    # ax.set_ylabel('concentration (potentially dimensionless)')
    # ax.legend(loc=2)
    # plt.show(fig)
    
    # always gen new data, can run dmaps far faster with c++
    # sample params noisily in 5d space, 10 points per axis for a total of 10e5 points (too many?)
    # center each param at K = 2.0; V = 1.0; St = 2.0; epsilon = 1e-3; kappa = 10.0
    npts_per_axis = 8
    # # use these ranges as a guide for sampling (with true vals of K = 2.0; V = 1.0; St = 2.0; epsilon = 1e-3; kappa = 10.0):
    # # epsilons = np.linspace(-4, -1, 5) # little effect
    # # kappas = np.linspace(-2, 2, 5) # little effect
    # # Ks = np.linspace(-1, 3, 5) # very significant effect
    # # Vs = np.linspace(-1, 3, 5) # very significant effect
    Ks = np.power(10, 2*np.random.uniform(-3, 3, npts_per_axis))
    Vs = np.power(10, np.random.uniform(-3, 3, npts_per_axis))
    Sts = np.power(10, np.random.uniform(0, 1, npts_per_axis))
    test_params = OrderedDict((('K',Ks), ('V',Vs), ('St',Sts))) # parameters being varied
    param_sets = test_params.values()
    ntest_params = len(param_sets)
    const_params = {'eps':epsilon, 'kappa':kappa} # parameters that will be held constant throughout
    npts = np.power(npts_per_axis, ntest_params)
    index = np.empty(ntest_params) # used in future calculation to get index of desired parameter combination
    powers = np.array([np.power(npts_per_axis, i) for i in range(ntest_params)]) # powers of ntest_params, e.g. 1, 5, 25, ... also used in index calc
    npts_per_proc = npts/nprocs # number of points that will be sent to each process
    tol = 1.0 # ob. fn. tolerance, i.e. any points for which the ob. fn. exceeds this value will be discarded
    kept_params = np.empty((npts_per_proc, ntest_params+1)) # storage for all possible params and their respective ob. fn. evaluations
    kept_npts = 0 # number of parameter sets that fall within tolerated ob. fn. range
    # unset, ordered param dict
    params = OrderedDict((('K',False), ('V',False), ('St',False), ('eps',False), ('kappa',False)))

    for i in range(rank*npts_per_proc, (rank+1)*npts_per_proc):
        # probably a more efficient method of calculating the current index instead of performing 'ntest_params' calculations every time
        index = i/powers%npts_per_axis
        new_params = np.array([param_sets[j][index[j]] for j in range(ntest_params)])
        for j, key in enumerate(test_params.keys()):
            params[key] = new_params[j]
        for key, val in const_params.items():
            params[key] = val
        # record param set and ob. fn. value if below tolerance
        ob_fn_eval = MM_system.of(params.values())
        if ob_fn_eval < tol and ob_fn_eval is not False:
            kept_params[kept_npts,:-1] = np.log10(new_params)
            kept_params[kept_npts,-1] = ob_fn_eval
            kept_npts += 1

    kept_params = kept_params[:kept_npts]
    # possible to use Gather
    kept_params = comm.gather(kept_params, root=0)
    if rank is 0:
        kept_params = np.concatenate(kept_params)
        kept_npts = kept_params.shape[0]
        # create fileheader specifying which params were investigated, e.g. K=True,V=False,St=True,eps=True,kappa=False
        file_header = ''.join([test_param_key + '=True,' for test_param_key in test_params.keys()])
        file_header = file_header + ''.join([const_param_key + '=False,' for const_param_key in const_params.keys()])
        # remove trailing comma
        file_header = file_header[:-1]
        np.savetxt('./data/input/sloppy_params' + str(kept_npts) + '.csv', kept_params, delimiter=',', header=file_header, comments='')
        print '************************************************************'
        print 'generated', kept_npts, 'new points with min obj. fn. value of', np.min(kept_params[:,-1])
        print 'saved in ./data/input/sloppy_params' + str(kept_npts) + '.csv'
        print '************************************************************'

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
    # sample_sloppy_params()
    # check_sloppiness()
    mm_contours()
