import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint
from mpi4py import MPI
import os

import dmaps
import plot_dmaps
import Z_Model as ZM
import Z_Model_Transformed as ZMT
import algorithms.CustomErrors as CustomErrors
import util_fns as uf

def dmaps_transformed_params():
    """Perform DMAP on nonlinear transformation of parameters lambda/epsilon"""
    
    if os.path.isfile('./data/lam_eps_ofevals.csv'):
        # PERFORM THE DMAP (already generated pts):

        data = np.genfromtxt('./data/lam_eps_ofevals.csv', delimiter=',')

        # extract sloppy parameter combinations
        tol = 0.01
        data = data[data[:,-1] < tol]

        # transform into swirl in c1/c2
        S = 1.0 # for now we require S <= 1 to invert back to lambda/epsilon
        # questionable redefinition of max param values
        lam_max, epsmax = np.max(data[:,:2], axis=0)
        c1 = lambda l, e: np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax)
        c2 = lambda l, e: np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax)

        # do the actual transformation
        cs = np.empty((data.shape[0], 2))
        cs[:,0] = c1(data[:,0], data[:,1])
        cs[:,1] = c2(data[:,0], data[:,1])

        neps = 8
        eps = np.logspace(-3, 3, neps)
        # epsilon_plot(eps, cs)
        eps = 1e-1
        eigvals, eigvects = dmaps.embed_data(cs, k=12, epsilon=eps)
        plot_dmaps.plot_xy(cs[:,0], cs[:,1], color=eigvects[:,1], scatter=True, xlabel=r'$c_1$', ylabel=r'$c_2$')
        # plot_dmaps.plot_embeddings(eigvects, eigvals, k=4)
    else:
        # CREATE DATASET (no dataset exists):
        # init MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        # set up base system
        # specify ode parameters
        (a_true, b_true, lam_true, eps_true) = (0.1, 0.01, 0.1, 0.001)
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
        lam_max = 1.2
        epsmax = 1e-1
        nsamples = 500
        lam_samples = np.linspace(0.9*lam_true, 1.1*lam_true, nsamples)
        eps_samples = np.linspace(0, epsmax, nsamples)
        # eps_samples = np.logspace(-6, np.log10(epsmax), nsamples)
        data = np.empty((nsamples*nsamples, 3)) # space for obj. fn. evals

        count = 0
        for lam in uf.parallelize_iterable(lam_samples, rank, nprocs):
            for eps in eps_samples:
                z_system.change_parameters(np.array((a_true, b_true, lam, eps)))
                try:
                    x_sample_traj = z_system.get_trajectory(x0_true, times)
                except CustomErrors.IntegrationError:
                    continue
                else:
                    data[count] = (lam, eps, get_of(x_sample_traj, x_true_traj))
                    count = count + 1

        data = data[:count]
        all_data = comm.gather(data, root=0)

        if rank is 0:
            all_data = np.concatenate(all_data)
            np.savetxt('./data/lam_eps_ofevals.csv', all_data, delimiter=',')
            print '******************************\n \
            Data saved in ./data/lam_eps_ofevals.csv, rerun to perform DMAP\n \
            ******************************'
            

    

    
    

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

def lam(c1, c2, lam_max, epsmax, S):
    c1 = c1*np.sqrt(e/epsmax + l/(lam_max))*np.cos(2*np.pi*e/epsmax)/(np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax))
    c2 = c2*np.sqrt(e/epsmax + l/(lam_max))*np.sin(2*np.pi*e/epsmax)/(np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax))
    angle = np.arctan2(c2, c1)
    angle[angle < 0] = angle[angle < 0] + 2*np.pi
    return S*lam_max*(c1*c1+c2*c2-angle/(2*np.pi*S)) # np.arcsin(c2/np.sqrt(c1*c1+c2*c2))

def eps(c1, c2, lam_max, epsmax, S):
    c1 = c1*np.sqrt(e/epsmax + l/(lam_max))*np.cos(2*np.pi*e/epsmax)/(np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax))
    c2 = c2*np.sqrt(e/epsmax + l/(lam_max))*np.sin(2*np.pi*e/epsmax)/(np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax))
    angle = np.arctan2(c2, c1)
    angle[angle < 0] = angle[angle < 0] + 2*np.pi
    return epsmax*angle/(2*np.pi*S)    

def check_transformed_params():
    # specify ode parameters
    (a_true, b_true, c1_true, c2_true) = (0.1, 0.1, 0.1, 0.001)
    params = np.array((a_true, b_true, c1_true, c2_true))
    # create system with given params
    lam_max = 1.2
    epsmax = 1e-1
    S = 2.0
    z_system = ZMT.Z_Model_Transformed(params, lam_max, epsmax, S)

    nsamples = 500
    c1s, c2s = np.meshgrid(np.linspace(-np.sqrt(1 + 1/S), np.sqrt(1 + 1/S), nsamples), np.linspace(-np.sqrt(1 + 1/S), np.sqrt(1 + 1/S), nsamples))
    lambdas, epsilons = np.meshgrid(np.linspace(0, lam_max, nsamples), np.linspace(0, epsmax, nsamples))
    c1 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax)
    c2 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax)
    # lam = lambda c1, c2: S*lam_max*(c1*c1+c2*c2+np.arctan2(c1, c2)/(2*np.pi*S)) # np.arcsin(c2/np.sqrt(c1*c1+c2*c2))
    # eps = lambda c1, c2: epsmax*np.arctan2(c1, c2)/(2*np.pi*S)
    plt.scatter(c1(epsilons, lambdas), c2(epsilons, lambdas), c=lambdas, lw=0, cmap='YlGnBu') #, eps(c1s, c2s)
    plt.show()
    plt.scatter(lam(c1(epsilons, lambdas), c2(epsilons, lambdas), lam_max, epsmax, S), eps(c1(epsilons, lambdas), c2(epsilons, lambdas), lam_max, epsmax, S), lw=0) #, cmap='YlGnBu') #, eps(c1s, c2s)
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
    nsamples_per_axis = 100
    nsamples = nsamples_per_axis**2
    x10s, x20s = np.meshgrid(np.linspace(1, 2, nsamples_per_axis), np.linspace(1, 2, nsamples_per_axis))
    x10s.shape = (nsamples,)
    x20s.shape = (nsamples,)
    x0_samples = np.array((x10s, x20s)).T # all samples of initial conditions in two columns
    of_evals = np.empty(nsamples) # space for obj. fn. evals
    x0s = np.empty((nsamples, 2))

    # loop through different initial conditions and record obj. fn. value
    count = 0
    for i, x0 in enumerate(x0_samples):
        uf.progress_bar(i, nsamples) # optional progress bar
        x_sample_traj = z_system.get_trajectory(x0, times)
        temp_eval = get_of(x_sample_traj, x_true_traj)
        if not np.isnan(temp_eval):
            of_evals[count] = temp_eval
            x0s[count] = x0
            count = count + 1
        

    print count
    x0s = x0s[:count]
    of_evals = of_evals[:count]
    # plot grid of sampled points colored by obj. fn. value
    print np.any(np.isnan(of_evals)), np.any(np.isinf(of_evals))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0s[:,0], x0s[:,1])
    plt.show()

if __name__=='__main__':
    # check_params()
    # check_rhs()
    # check_transformed_params()
    check_x0()
    # dmaps_transformed_params()


# S = 10.0
# c1 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax)
# c2 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax)
# epsmax = 1
# lam_max = 1
# eps = np.linspace(0,epsmax,1000)
# lams = np.linspace(0,lam_max,1000)
# eps, lams = np.meshgrid(eps, lams)
# plt.scatter(c1(eps, lams), c2(eps, lams), c=lams, lw=0)
    
