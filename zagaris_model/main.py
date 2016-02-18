"""Investigates sloppiness in Antonios' model"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint
from mpi4py import MPI
from mpl_toolkits.mplot3d import Axes3D
import os

import dmaps
import plot_dmaps
import Z_Model as ZM
import Z_Model_Transformed as ZMT
import algorithms.CustomErrors as CustomErrors
import util_fns as uf

class DMAPS_Data_Kernel:
    """Computes kernel between two points in parameter space, taking into account both the euclidean distance between parameters and the euclidean distance between model predictions at those parameters"""
    def __init__(self, epsilon):
        self._epsilon = epsilon

    def __call__(self, x1, x2):
        """Custom kernel given by: :math:`k(x_1, x_2) = e^{\frac{-(\|log(x_1) - log(x_2) \|^2 + \|m(x_1) - m(x_2)\|^2)}{\epsilon^2} - \frac{(of(x_1) - of(x_2))^2}{\epsilon}}` where :math:`m(x_i)` is the model prediction at parameter set :math:`x_i`

        ..note:
            parameters should be unmodified, do not take a logarithm as this kernel does this automatically

        Args:
            x1 (array): first data point in which x = [(parameters), (predictions)]
            x2 (array): second data point in which x = [(parameters), (predictions)]
        """
        return np.exp(-np.power(np.linalg.norm(np.log10(x1[0]) - np.log10(x2[0])),2)/self._epsilon - np.power(np.linalg.norm(x1[1] - x2[1]), 2)/(self._epsilon*self._epsilon))

def dmaps_two_important_one_sloppy_only_data():
    """Generate parameter combinations in which there are two important (alpha, lambda) and one sloppy (epsilon) parameter(s) and use DMAPS to uncover them, but uses as data only the trajectories and not any information about the objective function"""
    params = np.load('./data/a-lam-eps-of-params-new.pkl')
    trajs = np.load('./data/a-lam-eps-trajs-new.pkl')
    tol = 1.5e-4 # 2e-2 for old data
    trajs = trajs[params[:,3] < tol]
    params = params[params[:,3] < tol]
    params = params[:,:3]
    print 'Have', params.shape[0], 'pts in dataset'
    # epsilons = np.logspace(-3, 1, 5)
    # dmaps.epsilon_plot(epsilons, trajs)
    epsilon = 1e-2 # from epsilon plot
    k = 80
    eigvals, eigvects = dmaps.embed_data(trajs, k, epsilon=epsilon)
    eigvals.dump('./data/dmaps-data-kernel-eigvals.pkl')
    eigvects.dump('./data/dmaps-data-kernel-eigvects.pkl')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.hold(False)
    ax.scatter(np.log10(params[:,0]), np.log10(params[:,1]), np.log10(params[:,2]), c=eigvects[:,i])
    plt.show()
    # for i in range(1,k):
    #     ax.scatter(np.log10(params[:,0]), np.log10(params[:,1]), np.log10(params[:,2]), c=eigvects[:,i])
    #     plt.savefig('./figs/pure-data-space/data-space-dmaps' + str(i) + '.png')
    

def dmaps_two_important_one_sloppy():
    """Generate parameter combinations in which there are two important (alpha, lambda) and one sloppy (epsilon) parameter(s) and use DMAPS with a kernel that accounts for both parameter-space distance and distances in model output, with the aim to uncover the alpha and lambda parameters"""
    if os.path.isfile('./data/a-lam-eps-of-params-new.pkl'):
        # already have data saved, load and trim data to approx 5000 pts for DMAPS
        params = np.load('./data/a-lam-eps-of-params.pkl')
        trajs = np.load('./data/a-lam-eps-trajs.pkl')
        tol = 2e-2
        trajs = trajs[params[:,3] < tol]
        params = params[params[:,3] < tol]
        params = params[:,:3]
        print 'Have', params.shape[0], 'pts in dataset'
        data = zip(params, trajs)
        # epsilons = np.logspace(-3, 1, 5)
        # kernels = [DMAPS_Data_Kernel(epsilon) for epsilon in epsilons]
        # dmaps.kernel_plot(kernels, epsilons, data)
        epsilon = 1e-2 # from epsilon plot
        kernel = DMAPS_Data_Kernel(epsilon)
        k = 30
        eigvals, eigvects = dmaps.embed_data_customkernel(data, k, kernel, symmetric=True)
        eigvals.dump('./data/dmaps-data-kernel-eigvals.pkl')
        eigvects.dump('./data/dmaps-data-kernel-eigvects.pkl')
        for i in range(1,k):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(np.log10(params[:,0]), np.log10(params[:,1]), np.log10(params[:,2]), c=eigvects[:,i])
            plt.savefig('./figs/data-space-dmaps' + str(i) + '.png')
            # plt.show()
    else:
        # need to generate dataset

        # CREATE DATASET (no dataset exists):
        # init MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        # set up base system
        # specify ode parameters
        (a_true, b_true, lam_true, eps_true) = (0.1, 1.0, 0.1, 0.001)
        params = np.array((a_true, b_true, lam_true, eps_true))
        # create system with given params
        z_system = ZM.Z_Model(params)

        # set up integration times
        t0 = 3*eps_true # 0
        tfinal = 1/lam_true
        dt = eps_true
        ntimes = 50
        times = np.linspace(t0, tfinal, ntimes)

        # get true trajectory based on true initial conditions
        x0_true = np.array((1, a_true))
        x_true_traj = z_system.get_trajectory(x0_true, times)

        # set up sampling grid and storage space for obj. fn. evals
        # lam_max = 1.2
        nsamples = 100
        lam_samples = np.linspace(0.9*lam_true, 1.1*lam_true, nsamples)
        a_samples = np.linspace(0.9*a_true, 1.1*a_true, nsamples)
        epsmin = 1e-6
        epsmax = 1e-1
        eps_samples = np.logspace(np.log10(epsmin), np.log10(epsmax), nsamples)
        # add noise to each individual parameter combination to create nice dataset
        params_noise = np.empty((nsamples*nsamples*nsamples, 4))
        params_noise[:,0] = 0.01*np.random.normal(loc=0, size=nsamples*nsamples*nsamples) # same noise for both lam and a
        params_noise[:,1] = 0 # no noise in b
        params_noise[:,2] = 0.01*np.random.normal(loc=0, size=nsamples*nsamples*nsamples)
        params_noise[:,3] = 0.1*np.random.normal(loc=0, size=nsamples*nsamples*nsamples) # noise for eps must vary with scale
        # eps_samples = np.logspace(-6, np.log10(epsmax), nsamples)
        params = np.empty((nsamples*nsamples*nsamples, 4)) # space for obj. fn. evals
        trajs = np.empty((nsamples*nsamples*nsamples, ntimes, 2))

        count = 0
        for lam in uf.parallelize_iterable(lam_samples, rank, nprocs):
            for eps in eps_samples:
                for a in a_samples:
                    new_params = np.array((a, b_true, lam, eps)) + params_noise[count]*np.array((1,1,1,eps))
                    z_system.change_parameters(new_params)
                    try:
                        x_sample_traj = z_system.get_trajectory(x0_true, times)
                    except CustomErrors.IntegrationError:
                        continue
                    else:
                        params[count] = (new_params[0], new_params[2], new_params[3], get_of(x_sample_traj, x_true_traj)) # a, lam, eps
                        trajs[count] = x_sample_traj
                        count = count + 1

        params = params[:count]
        all_params = comm.gather(params, root=0)
        trajs = trajs[:count]
        all_trajs = comm.gather(trajs, root=0)

        if rank is 0:
            all_params = np.concatenate(all_params)
            all_params.dump('./data/a-lam-eps-of-params-new.pkl')
            all_trajs = np.concatenate(all_trajs)
            all_trajs.dump('./data/a-lam-eps-trajs-new.pkl')

            print '******************************\nData saved in ./data/a-lam-eps-...\n******************************'
        

    
    
def henon(x0, y0, n, a, b):
    if n > 0:
        return henon(1 - a*x0*x0 + y0, b*x0, n-1, a, b)
    else:
        return [x0, y0]


def dmaps_transformed_params():
    """Perform DMAP on nonlinear, swirled transformation of parameters lambda/epsilon (important/sloppy)"""
    
    if os.path.isfile('./data/a-lam-ofevals-2016.csv'): # os.path.isfile('./data/lam_eps_ofevals-2016.csv'):
        # PERFORM THE DMAP (already generated pts):

        print 'Already have sloppy data, transforming and embedding'

        data = np.genfromtxt('./data/a-lam-ofevals-2016.csv', delimiter=',')

        # extract sloppy parameter combinations
        tol = 50 # 0.01
        data = data[data[:,-1] < tol]

        # transform into swirl in c1/c2
        S = 1.0 # for now we require S <= 1 to invert back to lambda/epsilon
        # questionable redefinition of max param values
        lam_max, epsmax = np.max(data[:,:2], axis=0)
        # c1 = lambda l, e: np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax)
        # c2 = lambda l, e: np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax)

        y1 = lambda l, e: l + np.power(np.log10(e)- np.average(np.log10(e)), 2)
        y2 = lambda l, e: np.log10(e) - np.average(np.log10(e))

        a = 1.3
        b = 0.3
        
        # do the actual transformation
        cs1 = np.array(henon(data[:,1], data[:,0], 1, a, b)).T
        cs2 = np.array(henon(data[:,1], data[:,0], 2, a, b)).T
        cs3 = np.array(henon(data[:,1], data[:,0], 3, a, b)).T
        cs4 = np.array(henon(data[:,1], data[:,0], 4, a, b)).T

        # look at dataset and subsequent transformations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:,1], data[:,0], c=data[:,2], s=3)
        ax.set_xlabel(r'$x_0 (= \lambda)$', fontsize=72)
        ax.set_ylabel(r'$y_0 (= a)$', fontsize=72)
        fig.subplots_adjust(bottom=0.15)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(cs1[:,0], cs1[:,1], c=data[:,2], s=3)
        ax.set_xlabel(r'$x_1$', fontsize=72)
        ax.set_ylabel(r'$y_1$', fontsize=72)
        fig.subplots_adjust(bottom=0.15)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(cs2[:,0], cs2[:,1], c=data[:,2], s=3)
        ax.set_xlabel(r'$x_2$', fontsize=72)
        ax.set_ylabel(r'$y_2$', fontsize=72)
        fig.subplots_adjust(bottom=0.15)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(cs3[:,0], cs3[:,1], c=data[:,2], s=3)
        ax.set_xlabel(r'$x_3$', fontsize=72)
        ax.set_ylabel(r'$y_3$', fontsize=72)
        fig.subplots_adjust(bottom=0.15)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(cs4[:,0], cs4[:,1], c=data[:,2], s=3)
        ax.set_xlabel(r'$x_4$', fontsize=72)
        ax.set_ylabel(r'$y_4$', fontsize=72)
        fig.subplots_adjust(bottom=0.15)


        plt.show()

        # neps = 8
        # eps = np.logspace(-3, 3, neps)
        # epsilon_plot(eps, cs)
        eps = 1e-1
        eigvals, eigvects = dmaps.embed_data(cs2, k=12, epsilon=eps)
        plot_dmaps.plot_xy(cs2[:,0], cs2[:,1], color=eigvects[:,1], scatter=True, xlabel=r'$y_1$', ylabel=r'$y_2$')
        # plot_dmaps.plot_embeddings(eigvects, eigvals, k=4)
    else:
        # CREATE DATASET (no dataset exists):
        # init MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        # set up base system
        # specify ode parameters
        (a_true, b_true, lam_true, eps_true) = (1.0, 0.01, 1.0, 0.001) # (0.1, 0.01, 0.1, 0.001)
        params = np.array((a_true, b_true, lam_true, eps_true))
        # create system with given params
        z_system = ZM.Z_Model(params)

        # set up integration times
        t0 = 0
        tfinal = 1.0/lam_true
        dt = eps_true
        times = np.arange(t0, tfinal, dt)
        ntimes = times.shape[0]

        # get true trajectory based on true initial conditions
        x0_true = np.array((1, a_true))
        x_true_traj = z_system.get_trajectory_quadratic(x0_true, times)

        # # set up sampling grid and storage space for obj. fn. evals
        # lam_max = 1.2
        # epsmax = 1e-1
        # nsamples = 500
        # lam_samples = np.linspace(0.9*lam_true, 1.1*lam_true, nsamples)
        # eps_samples = np.linspace(0, epsmax, nsamples)
        # eps_samples = np.logspace(-6, np.log10(epsmax), nsamples)
        # data = np.empty((nsamples*nsamples, 3)) # space for obj. fn. evals
        nsamples = 40000
        data = np.empty((nsamples, 3))
        a_lam_samples = np.random.uniform(size=(nsamples, 2))*np.array((1.5,1.5)) + np.array((0.25, 0.25)) # a \in (7, 9) lamb \in (6, 11)

        count = 0
        for a, lam in uf.parallelize_iterable(a_lam_samples, rank, nprocs):
            z_system.change_parameters(np.array((a, b_true, lam, eps_true)))
            try:
                x_sample_traj = z_system.get_trajectory_quadratic(x0_true, times)
            except CustomErrors.IntegrationError:
                continue
            else:
                data[count] = (a, lam, get_of(x_sample_traj, x_true_traj))
                count = count + 1


        # count = 0
        #     for eps in eps_samples:
        #         z_system.change_parameters(np.array((a_true, b_true, lam, eps)))
        #         try:
        #             x_sample_traj = z_system.get_trajectory(x0_true, times)
        #         except CustomErrors.IntegrationError:
        #             continue
        #         else:
        #             data[count] = (lam, eps, get_of(x_sample_traj, x_true_traj))
        #             count = count + 1

        data = data[:count]
        all_data = comm.gather(data, root=0)

        if rank is 0:
            all_data = np.concatenate(all_data)
            np.savetxt('./data/a-lam-ofevals-2016.csv', all_data, delimiter=',')
            print '******************************\n \
            Data saved in ./data/a-lam-ofevals-2016.csv, rerun to perform DMAP\n \
            ******************************'

        # if rank is 0:
        #     all_data = np.concatenate(all_data)
        #     np.savetxt('./data/lam_eps_ofevals-2016.csv', all_data, delimiter=',')
        #     print '******************************\n \
        #     Data saved in ./data/lam_eps_ofevals-2016.csv, rerun to perform DMAP\n \
            # ******************************'
    

def get_of(sample_traj, true_traj):
    return np.power(np.linalg.norm(sample_traj - true_traj), 2)

def check_params():
    """Plots grid of parameters colored by objective function value"""
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # specify ode parameters
    (a_true, b_true, lam_true, eps_true) = (1.0, 0.01, 1.0, 0.001) #(0.1, 0.01, 0.1, 0.1)
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
    x_true_traj = z_system.get_trajectory_quadratic(x0_true, times)

    # set up sampling grid and storage space for obj. fn. evals
    nsamples = 100
    a_samples = np.logspace(-2, 1, nsamples)
    lam_samples = np.logspace(-2, 1, nsamples)
    # b_samples = np.logspace(-4, -1, nsamples)
    # a_samples = np.linspace(-.2, .2, nsamples)
    # b_samples = np.linspace(-3, 3, nsamples)
    data = np.empty((nsamples*nsamples, 3)) # space for obj. fn. evals

    count = 0
    for a in uf.parallelize_iterable(a_samples, rank, nprocs):
        for lam in lam_samples:
            z_system.change_parameters(np.array((a, b_true, lam, eps_true)))
            try:
                x_sample_traj = z_system.get_trajectory_quadratic(x0_true, times)
            except CustomErrors.IntegrationError:
                continue
            else:
                data[count] = (a, lam, get_of(x_sample_traj, x_true_traj))
                count = count + 1

    data = data[:count]
    all_data = comm.gather(data, root=0)

    if rank is 0:
        all_data = np.concatenate(all_data)
        all_data.dump('./data/a-lam-of.pkl')

        # plot output
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.scatter(all_data[:,0], all_data[:,1], c=np.log10(all_data[:,2]), s=40)
        # ax.set_ylabel(r'$\lambda$')
        # ax.set_xlabel(r'$\epsilon$')
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        plt.show()

def data_space_plot():
    """Creates three-dimensional dataspace plot, creating trajectories from different values of a and b (parameters could be changed), and then plots this three-dimensional figure and colors by both selected parameters. **Note: This function uses the quadratic functional form for the fast manifold, as opposed to the linear form used above**"""
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
        x_true_traj = z_system.get_trajectory_quadratic(x0_true, times)
        plt.plot(x_true_traj[:,0], x_true_traj[:,1], lw=5)
        # x_true_traj2 = z_system.get_trajectory_quadratic(x0_true, np.linspace(0,50,50))
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
                x_sample_traj = z_system.get_trajectory_quadratic(x0_true, times)
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
        ax.scatter(np.log10(all_data[:,0]), np.log10(all_data[:,1]), np.log10(all_data[:,2]), c=all_data[:,3], lw=0)
        # ax.set_ylabel(r'$\lambda$')
        # ax.set_xlabel(r'$\epsilon$')
        ax.set_xlabel(r'$log(y_1)$')
        ax.set_ylabel(r'$log(y_2)$')
        ax.set_zlabel(r'$log(y_3)$')
        ax.set_title('colored by a')
        plt.show()

        # plot output
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.scatter(np.log10(all_data[:,0]), np.log10(all_data[:,1]), np.log10(all_data[:,2]), c=all_data[:,4], lw=0)
        # ax.set_ylabel(r'$\lambda$')
        # ax.set_xlabel(r'$\epsilon$')
        ax.set_xlabel(r'$log(y_1)$')
        ax.set_ylabel(r'$log(y_2)$')
        ax.set_zlabel(r'$log(y_3)$')
        ax.set_title('colored by b')
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
    """Demonstrates the nonlinear swirl transformation on a grid of eps/lam, creating a swirled parameter space c1/c2"""
    # specify ode parameters
    (a_true, b_true, c1_true, c2_true) = (0.1, 0.1, 0.1, 0.001)
    # create system with given params
    lam_max = 1.2
    epsmax = 1e-1
    S = 2.0

    nsamples = 500
    lambdas, epsilons = np.meshgrid(np.linspace(0, lam_max, nsamples), np.linspace(0, epsmax, nsamples))
    c1 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.cos(2*np.pi*S*e/epsmax)
    c2 = lambda e, l: np.sqrt(e/epsmax + l/(S*lam_max))*np.sin(2*np.pi*S*e/epsmax)
    plt.scatter(c1(epsilons, lambdas), c2(epsilons, lambdas), c=lambdas, lw=0, cmap='YlGnBu') #, eps(c1s, c2s)
    plt.show()
    plt.scatter(lam(c1(epsilons, lambdas), c2(epsilons, lambdas), lam_max, epsmax, S), eps(c1(epsilons, lambdas), c2(epsilons, lambdas), lam_max, epsmax, S), lw=0) #, cmap='YlGnBu') #, eps(c1s, c2s)
    plt.show()


def check_transformed_params_contours():
    """Attempt to generate a contour plot in c1/c2 space using the transformed model. **Does not work, dmaps_transformed_params successfully performs a similar function**"""
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
    nsamples = 100
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
    """Creates plots in phase plane showing 1.) effect of different epsilons on trajectories and 2.) effects of different initial conditions on trajectories. Mainly reveals how sloppiness arises in singularly perturbed systems"""
    # specify ode parameters
    #
    # NOTE: if 'a' and 'b' are not chosen small enough, the square root evaluated in gen_trajectory will return 'nan'
    #
    params = np.array((0.5, 0.1, 1, 0.1)) # (a, b, lambda, epsilon)
    (a, b, lam, eps) = params
    # create system with given params
    z_system = ZM.Z_Model(params)

    # set up integration times
    t0 = 0.0
    tfinal = 1
    dt = 0.001
    times = np.arange(t0, tfinal, dt)
    ntimes = times.shape[0]

    # get true trajectory based on true initial conditions
    x0_true = np.array((1., 4.))
    x_true_traj = z_system.get_trajectory(x0_true, times)
    # plt.plot(times, )
    plt.show()

    # plot showing increasing sloppiness in epsilon, singular pert param
    epsilons = np.logspace(-1, -5, 5)
    colors = ['k', 'g', 'r', 'b', 'c']
    for i, e in enumerate(epsilons):
        z_system.change_parameters(np.array((a,b,lam,e)))
        traj = z_system.get_trajectory(x0_true, times)
        print np.any(np.isnan(traj))
        plt.scatter(traj[:,0], traj[:,1], lw=0, label=r'$\epsilon$=%1.0e' % e, color=colors[i], s=60)
        # plt.plot(traj[:,0], traj[:,1], label=r'$\epsilon$=' + str(e), c=colors[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)
    plt.show()

    # plot showing sloppiness in initial condition, so long as they lie on the same fast manifold
    # ensure we're in a singularly perturbed regime
    eps = 1e-3
    z_system.change_parameters(np.array((a,b,lam,eps)))
    # find multiple points on a single fast manifold
    x0 = np.array((1., 2.))
    traj = z_system.get_trajectory(x0, times)
    xts = traj[:4]
    # find corresponding x0, y0 that will be mapped to initial points on this fast manifold
    x0s = xts[:,0] - b*xts[:,1]*xts[:,1]
    y0s = -a*b*((np.power(2*xts[:,1]-1/(a*b), 2) - 1/np.power(a*b, 2))/4 + x0s/b)
    x0s = np.array((x0s, y0s)).T
    # increase epsilon to ensure initial points are on slow manifold by the time the second point is recorded
    eps = 1e-4
    z_system.change_parameters(np.array((a,b,lam,eps)))
    # decrease total time
    t0 = 0.0
    tfinal = 0.2
    dt = 0.001
    times = np.arange(t0, tfinal, dt)
    ntimes = times.shape[0]
    # x0s = np.array((1 + 1.0*np.linspace(1,3,5), np.linspace(1,3,5))).T # initial conditions to loop over
    for i, x0 in enumerate(x0s):
        traj = z_system.get_trajectory(x0, times)
        print np.any(np.isnan(traj))
        plt.scatter(traj[:,0], traj[:,1], label=r'$x_0$=%1.1f' % x0[1], color=colors[i], s=100)
        # plt.plot(traj[:,0], traj[:,1], label=r'$\epsilon$=' + str(e), c=colors[i])
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend(loc=4)
    plt.show()
        
    
    
    

    # # set up sampling grid and storage space for obj. fn. evals
    # nsamples_per_axis = 100
    # nsamples = nsamples_per_axis**2
    # x10s, x20s = np.meshgrid(np.linspace(1, 2, nsamples_per_axis), np.linspace(1, 2, nsamples_per_axis))
    # x10s.shape = (nsamples,)
    # x20s.shape = (nsamples,)
    # x0_samples = np.array((x10s, x20s)).T # all samples of initial conditions in two columns
    # of_evals = np.empty(nsamples) # space for obj. fn. evals
    # x0s = np.empty((nsamples, 2))

    # # loop through different initial conditions and record obj. fn. value
    # count = 0
    # for i, x0 in enumerate(x0_samples):
    #     uf.progress_bar(i, nsamples) # optional progress bar
    #     x_sample_traj = z_system.get_trajectory(x0, times)
    #     temp_eval = get_of(x_sample_traj, x_true_traj)
    #     if not np.isnan(temp_eval):
    #         of_evals[count] = temp_eval
    #         x0s[count] = x0
    #         count = count + 1
        

    # print count
    # x0s = x0s[:count]
    # of_evals = of_evals[:count]
    # # plot grid of sampled points colored by obj. fn. value
    # print np.any(np.isnan(of_evals)), np.any(np.isinf(of_evals))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(x0s[:,0], x0s[:,1])
    # plt.show()

if __name__=='__main__':
    pass
    # check_x0()
    # check_transformed_params_contours() # unfunctional
    # check_transformed_params()
    # data_space_plot()
    check_params()
    # dmaps_transformed_params()
    # dmaps_two_important_one_sloppy()
    # dmaps_two_important_one_sloppy_only_data()
