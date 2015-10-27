import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint
from mpi4py import MPI
from mpl_toolkits.mplot3d import Axes3D

import Z_Model as ZM
import algorithms.CustomErrors as CustomErrors
import util_fns as uf

def get_of(sample_traj, true_traj):
    return np.linalg.norm(sample_traj - true_traj)

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

def data_space_plot():
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    params = np.array((0.1, 0.5, 1, 0.001)) # (a, b, lambda, epsilon)
    (a_true, b_true, lam_true, eps_true) = (0.1, 0.5, 1, 1000)
    # create system with given params
    z_system = ZM.Z_Model(params)

    # set up integration times
    t0 = 1.5
    tfinal = 5
    ntimes = 3
    times = np.linspace(t0, tfinal, ntimes)

    x0_true = np.array((1, 0.1))
    if rank is 0:
        # get true trajectory based on true initial conditions
        x0_true = np.array((1, 0.1))
        x_true_traj = z_system.get_trajectory(x0_true, times)
        plt.plot(x_true_traj[:,0], x_true_traj[:,1], lw=5)
        # x_true_traj2 = z_system.get_trajectory(x0_true, np.linspace(0,50,50))
        # plt.plot(np.linspace(0,50,50), x_true_traj2[:,1])
        plt.show()
    
    nsamples = 50
    a_samples = np.linspace(-.2, .2, nsamples)
    b_samples = np.linspace(-3, 3, nsamples)
    data = np.empty((nsamples*nsamples, 5)) # space for obj. fn. evals

    count = 0
    for a in uf.parallelize_iterable(a_samples, rank, nprocs):
        for b in b_samples:
            z_system.change_parameters(np.array((a, b, lam_true, eps_true)))
            try:
                x_sample_traj = z_system.get_trajectory(x0_true, times)
            except CustomErrors.IntegrationError:
                continue
            else:
                data[count,:3] = x_sample_traj[:,1]
                data[count,3:] = (a, b)
                count = count + 1

    data = data[:count]
    all_data = comm.gather(data, root=0)

    if rank is 0:
        all_data = np.concatenate(all_data)

        # plot output
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.scatter(all_data[:,0], all_data[:,1], all_data[:,2], c=all_data[:,3], lw=0)
        # ax.set_ylabel(r'$\lambda$')
        # ax.set_xlabel(r'$\epsilon$')
        ax.set_xlabel('y1')
        ax.set_ylabel('y2')
        ax.set_zlabel('y3')
        plt.show()

        # plot output
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.scatter(all_data[:,0], all_data[:,1], all_data[:,2], c=all_data[:,4], lw=0)
        # ax.set_ylabel(r'$\lambda$')
        # ax.set_xlabel(r'$\epsilon$')
        ax.set_xlabel('y1')
        ax.set_ylabel('y2')
        ax.set_zlabel('y3')
        plt.show()


def check_x0():

    # specify ode parameters
    params = np.array((0.1, 0.5, 0.1, 1000)) # (a, b, lambda, epsilon)
    # create system with given params
    z_system = ZM.Z_Model(params)

    # set up integration times
    t0 = 0
    tfinal = 50
    dt = 0.1
    times = np.arange(t0, tfinal, dt)
    ntimes = times.shape[0]

    # get true trajectory based on true initial conditions
    x0_true = np.array((1, 0.1))
    x_true_traj = z_system.get_trajectory(x0_true, times)
    plt.plot(times, x_true_traj[:,0])
    plt.plot(times, x_true_traj[:,1])
    plt.show()

    # set up sampling grid and storage space for obj. fn. evals
    nsamples_per_axis = 100
    nsamples = nsamples_per_axis**2
    x10s, x20s = np.meshgrid(np.linspace(-2, 2, nsamples_per_axis), np.linspace(-2, 2, nsamples_per_axis))
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
    # check_x0()
    data_space_plot()
