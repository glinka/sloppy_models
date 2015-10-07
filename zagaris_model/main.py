import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint
from mpi4py import MPI

import Z_Model as ZM
import Z_Model_Transformed as ZMT
import algorithms.CustomErrors as CustomErrors
import util_fns as uf

def get_of(sample_traj, true_traj):
    return np.power(np.linalg.norm(sample_traj - true_traj), 2)

def check_params():

    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # specify ode parameters
    (a_true, b_true, lam_true, eps_true) = (0.1, 0.01, 0.1, 0.1)
    params = np.array((a_true, b_true, lam_true, eps_true))
    # create system with given params
    z_system = ZM.Z_Model(params)

    # set up integration times
    t0 = 0
    tfinal = 1/lam_true
    dt = eps_true
    times = np.arange(t0, tfinal, dt)
    ntimes = times.shape[0]

    # get true trajectory based on true initial conditions
    x0_true = np.array((1, a_true))
    x_true_traj = z_system.get_trajectory(x0_true, times)

    # set up sampling grid and storage space for obj. fn. evals
    nsamples = 50
    # a_samples = np.logspace(-2, 1, nsamples)
    # b_samples = np.logspace(-3, 0, nsamples)
    a_samples = np.linspace(-.2, .2, nsamples)
    b_samples = np.linspace(-3, 3, nsamples)
    data = np.empty((nsamples*nsamples, 3)) # space for obj. fn. evals

    count = 0
    for a in uf.parallelize_iterable(a_samples, rank, nprocs):
        for b in b_samples:
            z_system.change_parameters(np.array((a, b, lam_true, eps_true)))
            try:
                x_sample_traj = z_system.get_trajectory(x0_true, times)
            except CustomErrors.IntegrationError:
                continue
            else:
                data[count] = (a, b, get_of(x_sample_traj, x_true_traj))
                count = count + 1

    data = data[:count]
    all_data = comm.gather(data, root=0)

    if rank is 0:
        all_data = np.concatenate(all_data)

        # plot output
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.scatter(all_data[:,0], all_data[:,1], c=all_data[:,2], s=40)
        # ax.set_ylabel(r'$\lambda$')
        # ax.set_xlabel(r'$\epsilon$')
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        plt.show()

def check_transformed_params():
    # specify ode parameters
    (a_true, b_true, c1_true, c2_true) = (0.1, 0.1, 0.1, 0.001)
    params = np.array((a_true, b_true, c1_true, c2_true))
    # create system with given params
    lam_max = 1.2
    epsmax = 1e-1
    S = 1.0
    z_system = ZMT.Z_Model_Transformed(params, lam_max, epsmax, S)

    nsamples = 500
    c1s, c2s = np.meshgrid(np.linspace(-np.sqrt(1 + 1/S), np.sqrt(1 + 1/S), nsamples), np.linspace(-np.sqrt(1 + 1/S), np.sqrt(1 + 1/S), nsamples))
    lambdas, epsilons = np.meshgrid(np.linspace(0, lam_max, nsamples), np.linspace(0, epsmax, nsamples))
    c1 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax)
    c2 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax)
    lam = lambda c1, c2: S*lam_max*(c1*c1+c2*c2+np.arcsin(c2/np.sqrt(c1*c1+c2*c2))/(2*np.pi*S)) # np.arcsin(c2/np.sqrt(c1*c1+c2*c2))
    eps = lambda c1, c2: epsmax*np.arcsin(c2/np.sqrt(c1*c1+c2*c2))/(2*np.pi*S) # np.arctan(c2/c1)
    # plt.scatter(c1(epsilons, lambdas), c2(epsilons, lambdas), lw=0, cmap='YlGnBu') #, eps(c1s, c2s)
    plt.scatter(lam(c1(epsilons, lambdas), c2(epsilons, lambdas)), eps(c1(epsilons, lambdas), c2(epsilons, lambdas)), lw=0, cmap='YlGnBu') #, eps(c1s, c2s)
    plt.show()

    # plt.scatter(c1s, c2s, c=eps(c1s, c2s), lw=0, cmap='YlGnBu') #, eps(c1s, c2s)
    # plt.show()


def check_transformed_params_contours():

    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # specify ode parameters
    (a_true, b_true, c1_true, c2_true) = (0.1, 0.1, 0.1, 0.001)
    params = np.array((a_true, b_true, c1_true, c2_true))
    # create system with given params
    lam_max = 1.2
    epsmax = 1e-1
    S = 5.0
    z_system = ZMT.Z_Model_Transformed(params, lam_max, epsmax, S)

    # set up integration times
    lam_true = S*lam_max*(c1_true*c1_true+c2_true*c2_true+np.arctan(c2_true/c1_true)/(2*np.pi*S))
    eps_true = epsmax*np.arctan(c2_true/c1_true)/(2*np.pi*S)
    print 'lamtrue, epstrue:', lam_true, eps_true
    t0 = 0
    tfinal = 1/lam_true
    dt = eps_true
    times = np.arange(t0, tfinal, dt)
    ntimes = times.shape[0]

    # get true trajectory based on true initial conditions
    x0_true = np.array((1, a_true))
    x_true_traj = z_system.get_trajectory(x0_true, times)

    # set up sampling grid and storage space for obj. fn. evals
    nsamples = 500
    # c1_samples = np.logspace(-3, 3, nsamples)
    # c2_samples = np.logspace(-3, 3, nsamples)
    c1_samples = np.linspace(-np.sqrt(1 + 1/S), np.sqrt(1 + 1/S), nsamples)
    c2_samples = np.linspace(-np.sqrt(1 + 1/S), np.sqrt(1 + 1/S), nsamples)
    data = np.empty((nsamples*nsamples, 3)) # space for obj. fn. evals

    count = 0
    for c1 in uf.parallelize_iterable(c1_samples, rank, nprocs):
        for c2 in c2_samples:
            z_system.change_transformed_parameters(np.array((a_true, b_true, c1, c2)))
            try:
                x_sample_traj = z_system.get_trajectory(x0_true, times)
            except CustomErrors.IntegrationError:
                continue
            else:
                data[count] = (c1, c2, get_of(x_sample_traj, x_true_traj))
                count = count + 1

    data = data[:count]
    all_data = comm.gather(data, root=0)

    if rank is 0:
        all_data = np.concatenate(all_data)
        print 'of min,max:', np.min(data[:,2]), np.max(data[:,2])

        # plot output
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        tol = 1e2
        all_data = all_data[all_data[:,2] < tol]
        ax.scatter(all_data[:,0], all_data[:,1], c=all_data[:,2], s=40, lw=0)
        # ax.set_ylabel(r'$\lambda$')
        # ax.set_xlabel(r'$\epsilon$')
        ax.set_xlabel('c1')
        ax.set_ylabel('c2')
        plt.show()

def check_x0():

    # specify ode parameters
    params = np.array((1.0, 1.0, 0.01, 0.1)) # (a, b, lambda, epsilon)
    (a, b, lam, eps) = params
    # create system with given params
    z_system = ZM.Z_Model(params)

    # set up integration times
    t0 = 0
    tfinal = 20
    dt = 0.1
    times = np.arange(t0, tfinal, dt)
    ntimes = times.shape[0]

    # get true trajectory based on true initial conditions
    x0_true = np.array((1, 0.1))
    x_true_traj = z_system.get_trajectory(x0_true, times)

    # set up sampling grid and storage space for obj. fn. evals
    nsamples_per_axis = 1
    nsamples = nsamples_per_axis**2
    x10s, x20s = np.meshgrid(np.linspace(-1, 1, nsamples_per_axis), np.linspace(-1, 1, nsamples_per_axis))
    x10s.shape = (nsamples,)
    x20s.shape = (nsamples,)
    x0_samples = np.array((x10s, x20s)).T # all samples of initial conditions in two columns
    of_evals = np.empty(nsamples) # space for obj. fn. evals

    # loop through different initial conditions and record obj. fn. value
    for i, x0 in enumerate(x0_samples):
        uf.progress_bar(i, nsamples) # optional progress bar
        x_sample_traj = z_system.get_trajectory(x0, times)
        of_evals[i] = get_of(x_sample_traj, x_true_traj)

        

    # plot grid of sampled points colored by obj. fn. value
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0_samples[:,0], x0_samples[:,1], c=of_evals, zorder=1)
    plt.show()

if __name__=='__main__':
    # check_params()
    # check_rhs()
    check_transformed_params()
    # check_x0()


# S = 10.0
# c1 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax)
# c2 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax)
# epsmax = 1
# lam_max = 1
# eps = np.linspace(0,epsmax,1000)
# lams = np.linspace(0,lam_max,1000)
# eps, lams = np.meshgrid(eps, lams)
# plt.scatter(c1(eps, lams), c2(eps, lams), c=lams, lw=0)
    
