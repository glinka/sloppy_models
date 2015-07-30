"""Uses modules :py:mod:`MM` and :py:mod:`Hessian` to investigate the sloppiness Michaelis Menten parameters"""

import util_fns as uf
import algorithms.CustomErrors as CustomErrors
import MM
import dmaps
import dmaps_kernels
import plot_dmaps
from algorithms.Derivatives import hessian
import algorithms.PseudoArclengthContinuation as PSA
import MM_Specialization as MMS
from solarized import solarize
import numpy as np
from mpi4py import MPI
from sympy import Function, dsolve, Eq, Derivative, symbols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import contextlib
from collections import OrderedDict
import tempfile

# switch to nicer color scheme
solarize('light')

class Warning_Catcher():
    def __init__(self):
        self._temp_out = open('tmp.txt', 'w')#tempfile.TemporaryFile(bufsize=1000)
    def write(self, string):
        if 'lsoda' in string:
            raise CustomErrors.LSODA_Warning
    def fileno(self):
        return self._temp_out.fileno()
    def close(self):
        return self._temp_out.close()

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected():
    original_stdout_fd, original_stderr_fd = os.dup(sys.stdout.fileno()), os.dup(sys.stderr.fileno())
    temp_out = Warning_Catcher()
    temp_err = Warning_Catcher()
    try:
        # sys.stdout = Warning_Catcher()
        # sys.stdout = temp_out
        # sys.stderr = temp_err
        os.dup2(temp_out.fileno(), sys.stdout.fileno())
        # os.dup2(temp_err.fileno(), sys.stderr.fileno())
        # sys.stdout = Warning_Catcher()
        # sys.stderr = Warning_Catcher()
        yield sys.stdout, sys.stderr
    except CustomErrors.LSODA_Warning:
        raise
    finally:
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.dup2(original_stderr_fd, sys.stdout.fileno())
        temp_out.close()
        temp_err.close()
        # sys.stdout = original_stdout
        # sys.stderr = original_stderr
    
def test():
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    with stdout_redirected():
        libc.printf('lsoda C\n')
        print 'lsoda P'
        # sys.stdout.write('lsoda')
            
        # try:
        #     os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        # except ValueError:  # filename
        #     with open(to, 'wb') as to_file:
        #         os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        # try:
        #     yield stdout # allow code to be run with the redirected stdout
        # finally:
        #     # restore stdout to its previous value
        #     #NOTE: dup2 makes stdout_fd inheritable unconditionally
        #     stdout.flush()
        #     os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

def mm_sloppy_plot():
    """Plots trajectories of 'P' at different parameter values: a visual confirmation of parameter sloppiness"""
    # set up base system
    params = OrderedDict((('K',2.0), ('V',1.0), ('St',2.0), ('epsilon',1e-3), ('kappa',10.0))) # from Antonios' writeup
    true_params = np.array(params.values())
    nparams = true_params.shape[0]
    transform_id = 't2'
    # set init concentrations
    S0 = params['St']; C0 = 0.0; P0 = 0.0 # init concentrations
    Cs0 = np.array((S0, C0, P0))
    # set times at which to collect data
    tscale = (params['St'] + params['K'])/params['V'] # timescale of slow evolution
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    # use these params, concentrations and times to define the MM system
    MM_system = MM.MM_System(Cs0, times, true_params, transform_id)

    # define variables to test
    param1_name = 'epsilon'
    # nparam1s = 2
    param1s = [0.01, 0.000001]#np.logspace(-1, -0.5, nparam1s)
    # # nparam2s = 1
    param2_name = 'kappa'
    param2s = [10.0]#np.logspace(-2, 2, nparam2s)
    # param1_name = 'K'
    # # nparam1s = 3
    # param1s = np.array((100,))#2*np.logspace(0, 3, nparam1s)
    # param2_name = 'V'
    # param2s = 3.0*param1s/10.0 + np.array((10, 0))

    # set up figure, add true trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    ax.plot(times, MM_system.gen_profile(Cs0, times, true_params)[:,2], label='true params')

    markers = ['.', 'o', 'v', '^', 's', 'p', '*', 'x', '+', '_', 'D']
    count = 0
    # true_traj_squared_norm = np.power(np.linalg.norm(MM_system.gen_profile(Cs0, times, true_params)[:,2]), 2) # for recording relative as error, scale of_eval by this norm
    for param1 in param1s:
        for param2 in param2s:
            # update values
            params[param1_name] = param1
            params[param2_name] = param2
            # params['V'] = params['K']*3.0/10.0
            S0 = params['St']; C0 = 0.0; P0 = 0.0 # init concentrations
            Cs0 = np.array((S0, C0, P0))
            ax.plot(times, MM_system.gen_profile(Cs0, times, np.array(params.values()))[:,2], label=param1_name + '=' + str(param1) + ', ' + param2_name + '=' + str(param2) + ',error=' + str("%1.2e" % (np.power(MM_system.of(params.values()), 0.5)/npts)), marker=markers[count])
            count = count + 1
    ax.set_xlabel('time')
    ax.set_ylabel('P')
    # ax.set_title(r'$\frac{K}{V} = \frac{10}{3}$')
    ax.legend(fontsize=24, loc='lower right')
    plt.show()
            

def mm_contour_grid_mpi():
    """Calculates three-dimensional contours in K/V/S_t space in parallel through mpi4py, distributing S_t values over different processes and saving the output in './data/of_evals.csv'"""

    # # attempt to turn off error messages
    # np.seterr(all='ignore')

    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    # nks = 1000 # number of K vals to sample
    # nvs = 1000 # number of S vals to sample
    # nsts = 6 # number of S_t vals to sample
    # npts_per_proc = nsts/nprocs # number of St values to distribute to each processor
    # Ks = 2*np.logspace(-1, 3, nks) # k vals
    # Vs = np.logspace(-1, 3, nvs) # v vals
    # Sts = 2*np.logspace(-0.5, 0.5, nsts) # st vals
    # npts_per_proc = 1
    # nepsilons = 400
    # nkappas = 400
    # epsilons = np.logspace(-6, 0, nepsilons)
    # kappas = np.logspace(-5, 4, nkappas)
    # nks = 500 # number of K vals to sample
    # nvs = 500 # number of S vals to sample
    # npts_per_proc = 1 #nsts/nprocs # number of St values to distribute to each processor
    # Ks = 2*np.logspace(-1, 3, nks) # k vals
    # Vs = np.logspace(-1, 3, nvs) # v vals

    # set up base system
    params = OrderedDict((('K',2.0), ('V',1.0), ('St',2.0), ('epsilon',1e-3), ('kappa',10.0))) # from Antonios' writeup
    true_params = np.array(params.values())
    nparams = true_params.shape[0]
    transform_id = 't2'
    nstate_params = 100
    ncontinuation_params = 100
    nthird_params = 1*nprocs
    # state_params = {'id':'St', 'data':np.linspace(1.9, 2.1, nstate_params)}#2*np.logspace(-1, 3, nstate_params)}
    state_params = {'id':'V', 'data':np.linspace(0.5, 3.5, nstate_params)}#np.logspace(-1, 3, ncontinuation_params)}
    continuation_params = {'id':'K', 'data':np.linspace(0.5, 4.5, ncontinuation_params)}#np.logspace(-1, 3, ncontinuation_params)}
    third_params = {'id':'St', 'data':np.linspace(1, 3, nthird_params)}
    # set init concentrations
    S0 = params['St']; C0 = 0.0; P0 = 0.0 # init concentrations
    Cs0 = np.array((S0, C0, P0))
    # set times at which to collect data
    tscale = (params['St'] + params['K'])/params['V'] # timescale of slow evolution
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    # use these params, concentrations and times to define the MM system
    contour = 0.001
    MM_system = MMS.MM_Specialization(Cs0, times, true_params, transform_id, [state_params['id']], continuation_params['id'], contour)
    # true_traj_squared_norm = np.power(np.linalg.norm(MM_system.gen_profile(Cs0, times, true_params)[:,2]), 2) # for recording relative as error, scale of_eval by this norm
    #  loop over all parameter combinations
    tol  = 0.001 # f_avg_error below tol will be saved
    st_slices = []
    # suppress output to stdout in the inner loop, as it's always (hopefully) about lsoda's performance
    # with stdout_redirected():

    for third_param in uf.parallelize_iterable(third_params['data'], rank, nprocs):
        MM_system.adjust_const_param(third_params['id'], third_param)
        count = 0 # counter of number of parameter combinations that pass tolerance
        kept_pts = np.empty((nstate_params*ncontinuation_params,4)) # storage for parameter combinations that pass obj. fn. tol.
        for state_param in state_params['data']:
            for continuation_param in continuation_params['data']:
                try:
                    # record relative error
                    f_eval = MM_system.f_avg_error(np.array((state_param,)), continuation_param)
                except CustomErrors.EvalError:
                    continue
                else:
                    if f_eval < tol:
                        kept_pts[count] = (state_param, continuation_param, third_param, f_eval)
                        count = count + 1

        kept_pts = kept_pts[:count]
        if count > 0:
            st_slices.append(kept_pts)

    if len(st_slices) > 0:
        st_slices = np.concatenate(st_slices)
        print st_slices.shape
    else:
        st_slices = 'None'
    # gather all the points to root and save
    # kept_pts = kept_pts[:count]
    all_pts = comm.gather(st_slices, root=0)
    if rank is 0:
        while 'None' in all_pts:
            all_pts.remove('None')
        full_pts = np.concatenate(all_pts)
        header = ','.join([key + "=" + str(val) for key, val in params.items()]) + ',Tested=' + state_params['id'] + continuation_params['id'] + third_params['id']
        np.savetxt('./data/contours_' + state_params['id'] + '_' + continuation_params['id'] + '_' + third_params['id'] +  '.csv', full_pts, delimiter=',', header=header, comments='')


def mm_contour_grid():
    nks = 1000
    nvs = 1000
    Ks = 2*np.logspace(-1, 3, nks)
    Vs = np.logspace(-1, 3, nvs)
    if os.path.isfile('./of_evals.csv'):
        kept_pts = np.genfromtxt('./of_evals.csv', delimiter=',')
        count = kept_pts.shape[0]
    else:
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
        MM_system = MM.MM_System(Cs0, times, true_params, transform_id)
        print 'ofeval', MM_system.of(params.values())
        of_evals = np.empty((nks, nvs))
        test_params = true_params
        ndiscarded = 0
        kept_pts = np.empty((nks*nvs,3))
        tol = 0.1
        count = 0
        for i, K in enumerate(Ks):
            uf.progress_bar(i+1, nks)
            for j, V in enumerate(Vs):
                test_params[0] = K
                test_params[1] = V
                try:
                    # of_evals[i,j] = MM_system.of(test_params)
                    of_eval = MM_system.of(test_params)
                    if of_eval < tol:
                        kept_pts[count,0] = K
                        kept_pts[count,1] = V
                        kept_pts[count,2] = of_eval
                        count = count + 1
                except CustomErrors.EvalError:
                    ndiscarded = ndiscarded + 1
                    continue
        np.savetxt('./of_evals.csv', kept_pts, delimiter=',')
        print 'threw away', ndiscarded, 'pts'

    vgrid, kgrid = np.meshgrid(Vs, Ks)
    solarize('light')
    # plt.imshow(of_evals)
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plot = ax.scatter(kept_pts[:count,0], kept_pts[:count,1], c=kept_pts[:count,2], s=5, lw=0)
    ax.set_xlabel('K')
    ax.set_ylabel('V')
    plt.colorbar(plot)
    plt.show(fig)

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
    state_params = ['V']
    continuation_param = 'K'
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
    nsts = 500
    # Sts = np.linspace(2.00, 2.02, nsts)
    third_param_name = 'St'
    third_param_vals = np.linspace(2.006, 2.008, nsts)
    dthird_param = third_param_vals[1] - third_param_vals[0] # spacing of third param
    contour_val = 0.001 # from discussions, use error=1e-3 # np.logspace(-7,-1,npts_per_proc*nprocs) # countour values of interest
    ds = 1e-4
    ncontinuation_steps = 100

    branches = []
    geodesic_pts = np.empty((nsts, 3))
    init_guesses = np.empty((nsts, 3))
    maxiters=5000

    # current_index = 0 
    # for i, St in enumerate(uf.parallelize_iterable(Sts, rank, nprocs)):
    ncurves = 0
    for third_param_val in third_param_vals:
        mm_specialization = MMS.MM_Specialization(Cs0, times, true_params, transform_id, state_params, continuation_param, contour_val)
        mm_specialization.adjust_const_param(third_param_name, third_param_val)
        psa_solver = PSA.PSA(mm_specialization.f_avg_error, mm_specialization.f_gradient)
        try:
            # branch = psa_solver.find_branch(np.array((params[state_params[0]]+0.01,)), params[continuation_param]+0.01, ds, ncontinuation_steps, maxiters=maxiters)
            
            if ncurves is 0:
                branch = psa_solver.find_branch(np.array((params[state_params[0]] + 0.01,)), params[continuation_param] + 0.01, ds, ncontinuation_steps, maxiters=maxiters)
                init_guesses[0] = (params[state_params[0]] + 0.01, params[continuation_param] + 0.01, third_param_val)
            elif ncurves is 1:
                branch = psa_solver.find_branch(np.array((branches[0][0,0],)), branches[0][0,1], ds, ncontinuation_steps, maxiters=maxiters)
                init_guesses[1] = (branches[0][0,0], branches[0][0,1], third_param_val)
            # use some linear interpolation to get new starting point if we already have two branches
            elif ncurves is 2:
                # find nearest point between first entry in previous branch and the branch before (i.e. find point in branches[ncurves-2] closest to branches[ncurves-1][0,:])
                nearest_pt = branches[0][np.argmin(np.linalg.norm(branches[0] - branches[1][0], axis=1))]
                geodesic_pts[0,:] = nearest_pt
                geodesic_pts[1,:] = branches[1][0,:]
                init_guess = branches[1][0,:] + np.abs(dthird_param)*(branches[1][0,:] - nearest_pt)/np.linalg.norm(branches[1][0,:] - nearest_pt)
                branch = psa_solver.find_branch(np.array((init_guess[0],)), init_guess[1], ds, ncontinuation_steps, maxiters=maxiters)

                init_guesses[2] = np.copy(init_guess)

                # # visual testing of continuation in third param
                # fig = plt.figure()
                # ax = fig.add_subplot(111) # plotting axis
                # ax.scatter(branches[0][:,0], branches[0][:,1])
                # ax.scatter(branches[1][:,0], branches[1][:,1])
                # ax.scatter(init_guess[0], init_guess[1], c='g', s=50)
                # ax.scatter(nearest_pt[0], nearest_pt[1], c='c', s=50)
                # ax.scatter(branches[1][0,0], branches[1][0,1], c='r', s=50)
                # plt.show()

            # use quadratic interpolation if have three or more branches
            else:
                # find point on previous branch closest to most recent entry in 'geodesic_pts'
                nearest_pt = branches[ncurves-1][np.argmin(np.linalg.norm(branches[ncurves-1] - geodesic_pts[ncurves-2], axis=1))]
                geodesic_pts[ncurves-1] = nearest_pt

                poly_fit_order = 2
                # use previous three points to fit curves
                pts_to_fit = geodesic_pts[ncurves-3:ncurves]
                param1_param3_fit = np.poly1d(np.polyfit(pts_to_fit[:,2], pts_to_fit[:,0], poly_fit_order))
                param2_param3_fit = np.poly1d(np.polyfit(pts_to_fit[:,2], pts_to_fit[:,1], poly_fit_order))
                param1_extrap = param1_param3_fit(third_param_val)
                param2_extrap = param2_param3_fit(third_param_val)

                init_guesses[ncurves] = (param1_extrap, param2_extrap, third_param_val)

                print 'init error:', mm_specialization.f_avg_error(np.array((param1_extrap,)), param2_extrap), 'at branch', ncurves + 1

                branch = psa_solver.find_branch(np.array((param1_extrap,)), param2_extrap, ds, ncontinuation_steps, maxiters=maxiters, abstol=1e-8, bisection_on_maxiter=True)

        except CustomErrors.PSAError:
            # exit if previous level set was not found
            break
        else:
            err = 0
            for pt in branch:
                err = err + np.abs(mm_specialization.f_avg_error(np.array((pt[0],)), pt[1]))
            print 'found', branch.shape[0], 'pts at', third_param_name, 'of', third_param_val, 'with total error of', err
            # add St val to third col of branch array
            fullbranch = np.empty((branch.shape[0], branch.shape[1] + 1))
            fullbranch[:,:-1] = branch
            fullbranch[:,-1] = third_param_val
            branches.append(fullbranch)
            ncurves = ncurves + 1

    branches = np.concatenate(branches)

    fig = plt.figure()
    ax = fig.add_subplot(111) # plotting axis
    ax.scatter(geodesic_pts[:ncurves-1,0], geodesic_pts[:ncurves-1,1], c='g', s=75, marker='^')
    ax.scatter(init_guesses[:ncurves,0], init_guesses[:ncurves,1], c=range(ncurves), s=100, marker='*')
    ax.scatter(branches[:,0], branches[:,1], c=range(branches.shape[0]))
    plt.show()

    file_header = ','.join([key + '=' + str(val) for key, val in params.items()])
    print 'succesfully found', ncurves, 'level sets'
    np.savetxt('./data/output/contour_K_V_St.csv', branches, delimiter=',', header=file_header, comments='')

    # fullbranches = comm.gather(fullbranch, root=0)
    # if rank is 0:
    #     # get rid of those branches that didn't initiate
    #     while 'None' in fullbranches:
    #         fullbranches.remove('None')
    #     print '******************************'
    #     print len(fullbranches)
    #     print '******************************'
    #     fullbranches = np.concatenate(fullbranches)
    #     kept_npts = fullbranches.shape[0]
    #     # create fileheader specifying true param vals
    #     file_header = ','.join([key + '=' + str(val) for key, val in params.items()])
    #     np.savetxt('./data/output/contour_K_V_St.csv', fullbranches, delimiter=',')

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
    # mm_contour_grid()
    # mm_contours()
    # mm_sloppy_plot()
    mm_contour_grid_mpi()
    # test()
