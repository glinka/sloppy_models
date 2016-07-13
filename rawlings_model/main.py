"""Rawlings model-related parameter space investigations"""
import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import Rawlings_Model
import util_fns as uf
import dmaps
import plot_dmaps

def plot_data_dmaps_results():
    """Plots the results from dmaps_param_set_data_kernel (for aiche 2015)"""

    paramdata = np.genfromtxt('./../rawlings_model/data/data-dmaps-params.csv', delimiter=',')
    keff = paramdata[:,0]*paramdata[:,2]/(paramdata[:,1] + paramdata[:,2])
    paramdata = np.log10(paramdata)
    eigvects = np.genfromtxt('./../rawlings_model/data/data-dmaps-eigvects.csv', delimiter=',')
    # color params by dmaps coord
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c=eigvects[:,3], cmap='gnuplot2')
    ax.set_xlabel('log(' + r'$k_1$' + ')')
    ax.set_ylabel('log(' + r'$k_{-1}$' + ')')
    ax.set_zlabel('log(' + r'$k_2$' + ')')
    ax.xaxis._axinfo['label']['space_factor'] = 2.8
    ax.yaxis._axinfo['label']['space_factor'] = 2.8
    ax.zaxis._axinfo['label']['space_factor'] = 2.8
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    plt.show()
    # for ii in xrange(20,220,1):
    #     ax.view_init(azim=ii)
    #     plt.savefig('./figs/data-dmaps/dmaps-coloring' + str(280-ii) + '.png')

    # color params by keff
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c=keff, cmap='gnuplot2')
    ax.set_xlabel('log(' + r'$k_1$' + ')')
    ax.set_ylabel('log(' + r'$k_{-1}$' + ')')
    ax.set_zlabel('log(' + r'$k_2$' + ')')
    ax.xaxis._axinfo['label']['space_factor'] = 2.8
    ax.yaxis._axinfo['label']['space_factor'] = 2.8
    ax.zaxis._axinfo['label']['space_factor'] = 2.8
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    plt.show()
    # for ii in xrange(20,240,1):
    #     ax.view_init(azim=ii)
    #     plt.savefig('./figs/data-dmaps/keff-coloring' + str(220-ii) + '.png')


def qssa_comparison():
    """Shows efficacy of the qssa applied to this system (for gss 2015, fig 3.1)"""
    import matplotlib.pyplot as plt
    A0 = 1.0 # initial concentration of A
    k1_true = 1.0
    kinv_true = 10.0
    k2_true = 10.0
    decay_rate = k1_true*k2_true/(kinv_true + k2_true) # effective rate constant that governs exponential growth rate
    # start at t0 = 0, end at tf*decay_rate = 4
    ntimes = 20 # arbitrary
    times = np.linspace(0, 4/decay_rate, ntimes)
    model = Rawlings_Model(times, A0, k1_true, kinv_true, k2_true)
    Cs_true = model.gen_timecourse(k1_true, kinv_true, k2_true)
    Cs_qssa = A0*(1-np.exp(-times*decay_rate))
    plt.plot(times, Cs_true)
    plt.plot(times, Cs_qssa)
    plt.show()
    
def abc_analytical_contour():
    """Shows line of constant k_eff in k1/kinv/k2 space (for gss 2015, fig 3.2)"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    k1 = lambda kinv, k2: (kinv + k2)/k2
    kinvs, k2s = np.meshgrid(np.logspace(-4,1, 10), np.logspace(-4,1, 10))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(np.log10(k1(kinvs, k2s)), np.log10(kinvs), np.log10(k2s))
    ax.set_xlabel('log(' + r'$k_1$' + ')')
    ax.set_ylabel('log(' + r'$k_{-1}$' + ')')
    ax.set_zlabel('log(' + r'$k_2$' + ')')
    plt.tight_layout()
    for ii in xrange(0,360,1):
        ax.view_init(elev=10.0, azim=ii)
        ax.xaxis._axinfo['label']['space_factor'] = 2.8
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.zaxis._axinfo['label']['space_factor'] = 2.8
        ax.set_xticklabels([r'$10^{%i}$'%i for i in range(7)])
        ax.set_yticklabels([r'$10^{%i}$'%i for i in range(1, -5, -1)])
        ax.set_zticklabels([r'$10^{%i}$'%i for i in range(1, -5, -1)])
        plt.savefig("./figs/ks%i"%ii + ".png")


def sample_param_grid_mpi():
    """Samples k1, kinv, k2 over a lattice and records obj. fn. evaluations. A starting point for future analysis"""
    # set up mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # set up base model
    A0 = 1.0 # initial concentration of A
    k1_true = 0.1
    kinv_true = 1000.0
    k2_true = 1000.0
    eigval_plus_true = 0.5*(-(kinv_true + k1_true + k2_true) + np.sqrt(np.power(k1_true + kinv_true + k2_true, 2) - 4*k1_true*k2_true))
    dt = 0.5/np.abs(eigval_plus_true)
    # start at dt, end at 5*dt
    ntimes = 5 # somewhat arbitrary
    times = np.linspace(dt, 5*dt, ntimes)
    model = Rawlings_Model(times, A0, k1_true, kinv_true, k2_true, using_sympy=False)

    # plot analytical results vs. qssa
    if rank == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(times, model.gen_timecourse(k1_true, kinv_true, k2_true))
        ax.plot(times, A0*(1 - np.exp(-k1_true*k2_true*times/(kinv_true + k2_true))))
        plt.show()

    nks = 10000000
    # k1 \in [10^{-4}, 10^{-1}],    kinv, k2 \in [10^1, 10^4]
    ks = np.power(10, np.random.uniform(size=(nks, 3))*np.array((3, 3, 3)) - np.array((4, -1, -1)))
    nks_per_proc = nks/nprocs + 1 # add one to avoid rounding errors
    params_and_of_evals = np.empty((nks_per_proc, 4))
    count = 0
    for k in uf.parallelize_iterable(ks, rank, nprocs):
        params_and_of_evals[count] = model.lsq_of(k[0], k[1], k[2]), k[0], k[1], k[2]
        count = count + 1

    all_params_and_of_evals = comm.gather(params_and_of_evals[:count], root=0)
    if rank == 0:
        all_params_and_of_evals = np.concatenate(all_params_and_of_evals)
        print all_params_and_of_evals.shape
        all_params_and_of_evals.dump('./data/params-ofevals.pkl')


def dmaps_param_set():
    """Performs DMAP of log(parameter) set that fall within some ob. fn. tolerance"""
    # import data and save only those parameter combinations such that error(k1, kinv, k2) < tol

    data = np.load('./temp.pkl')
    # data = np.genfromtxt('./data/params-ofevals.csv', delimiter=',')

    of_max = 1e-3
    k1_max = 10
    kinv_min = 100
    k2_min = 100
    # data = data[data[:,0] < of_max]
    # data = data[data[:,1] < k1_max]
    # data = data[data[:,2] > kinv_min]
    # data = data[data[:,3] > k2_min]
    # slice = 5000
    # data = data[::data.shape[0]/slice]
    # data.dump('./temp.pkl')

    # of_max = 0.002 # from plotting with scratch.py
    # data = data[data[:,0] < of_max]
    # slice_size = 4 # used to further trim data
    # data = data[::slice_size]
    print 'have', data.shape[0], 'pts in dataset'
    keff = data[:,1]*data[:,3]/(data[:,2] + data[:,3])
    log_params_data = np.log10(data[:,1:])
    # # investigate which epsilon to choose
    # neps = 10 # number of epsilons to investigate
    # epsilons = np.logspace(-3,2, neps)
    # plot_dmaps.epsilon_plot(epsilons, log_params_data)
    # dmap the log data
    epsilon = 0.3 # from epsilon_plot
    k = 12 # number of dimensions for embedding

    # search through files in ./data to see if the embedding has already been computed
    filename_id = 'tol-' + str(of_max) + '-k-' + str(k)
    found_previous_embeddings = False

    eigvals, eigvects = None, None

    for filename in os.listdir('./data'):
        if filename_id in filename:
            # found previously saved data, import and do not recompute
            eigvects = np.genfromtxt('./data/dmaps-eigvects--tol-' + str(of_max) + '-k-' + str(k) + '.csv', delimiter=',')
            eigvals = np.genfromtxt('./data/dmaps-eigvals--tol-' + str(of_max) + '-k-' + str(k) + '.csv', delimiter=',')
            found_previous_embeddings = True
            break

    if found_previous_embeddings is False:
        print 'plotting from previous points'
        eigvals, eigvects = dmaps.embed_data(log_params_data, k, epsilon=epsilon)
        np.savetxt('./data/dmaps-eigvects--tol-' + str(of_max) + '-k-' + str(k) + '.csv', eigvects, delimiter=',')
        np.savetxt('./data/dmaps-eigvals--tol-' + str(of_max) + '-k-' + str(k) + '.csv', eigvals, delimiter=',')

    plot_dmaps.plot_xyz(log_params_data[:,0], log_params_data[:,1], log_params_data[:,2], color=eigvects[:,1], xlabel='\n\n' + r'$\log(k_1)$', ylabel='\n\n' + r'$\log(k_{-1})$', zlabel='\n\n' + r'$\log(k_2)$')
    plot_dmaps.plot_xyz(log_params_data[:,0], log_params_data[:,1], log_params_data[:,2], color=eigvects[:,2], xlabel='\n\n' + r'$\log(k_1)$', ylabel='\n\n' + r'$\log(k_{-1})$', zlabel='\n\n' + r'$\log(k_2)$')
    # plot_dmaps.plot_embeddings(eigvects, eigvals)

class DMAPS_Gradient_Kernel:
    """Class containing function dmaps_kernel, allows for storage of various parameters in controlled namespace"""
    def __init__(self, epsilon, gradient):
        self._epsilon = epsilon
        self._gradient = gradient

    def __call__(self, x1, x2):
        """Exponential kernel including difference in objective function value: :math:`k(x_1, x_2) = e^{\frac{-\|x_1 - x_2 \|^2}{\epsilon^2} - \frac{(of(x_1) - of(x_2))^2}{\epsilon}}`

        Args:
            x1 (array): first data point in which x = (of, k1, kinv, k2)
            x2 (array): second data point in which x = (of, k1, kinv, k2)
        """
        return np.exp(-np.power(np.linalg.norm(x1 - x2),2)/self._epsilon - np.power(np.dot(self._gradient(x1), x1 - x2)/self._epsilon, 2))
    

class DMAPS_Data_Kernel:
    """Computes kernel between two points in parameter space, taking into account both the euclidean distance between parameters and the euclidean distance between model predictions at those parameters"""
    def __init__(self, epsilon):
        self._epsilon = epsilon

    def __call__(self, x1, x2):
        """Custom kernel given by: :math:`k(x_1, x_2) = e^{\frac{-(\|log(x_1) - log(x_2) \|^2 + \|m(x_1) - m(x_2)\|^2)}{\epsilon^2} - \frac{(of(x_1) - of(x_2))^2}{\epsilon}}` where :math:`m(x_i)` is the model prediction at parameter set :math:`x_i`

        Args:
            x1 (array): first data point in which x = [(parameters), (predictions)]
            x2 (array): second data point in which x = [(parameters), (predictions)]
        """
        return np.exp(-np.power(np.linalg.norm(np.log10(x1[0]) - np.log10(x2[0])),2)/self._epsilon - np.power(np.linalg.norm(x1[1] - x2[1]), 2)/(self._epsilon*self._epsilon))
    

def dmaps_param_set_data_kernel():
    """DMAP a collection of sloppy parameter combinations using a kernel which accounts for the model predictions at each parameter combination which will, ideally, uncover the important parameter(s) in the model"""
    # set up base model
    A0 = 1.0 # initial concentration of A
    k1_true = 1.0
    kinv_true = 1000.0
    k2_true = 1000.0
    decay_rate = k1_true*k2_true/(kinv_true + k2_true) # effective rate constant that governs exponential growth rate
    # start at t0 = 0, end at tf*decay_rate = 4
    ntimes = 20 # arbitrary
    times = np.linspace(0, 4/decay_rate, ntimes)
    model = Rawlings_Model(times, A0, k1_true, kinv_true, k2_true, using_sympy=False)
    
    # import existing data
    data = np.genfromtxt('./data/params-ofevals.csv', delimiter=',')

    of_tol = 0.4 # from plotting with scratch.py
    somedata = data[data[:,0] < of_tol]
    # only keep npts points due to computational considerations
    npts = 4000
    slice_size = somedata.shape[0]/npts
    # throw out o.f. evals in first column
    somedata = somedata[::slice_size,1:]
    keff = somedata[:,0]*somedata[:,2]/(somedata[:,1] + somedata[:,2])
    somedata = somedata[keff < 1]
    npts = somedata.shape[0]
    print 'sending a cherry-picked sample of', npts, 'to dmaps'
    trajectories = np.empty((npts, ntimes)) # each row contains [(k1, kinv, k2), (model prediction at k1, kinv, k2)]

    # find model predictions from parameter set
    for i,param_set in enumerate(somedata):
        trajectories[i] = model.gen_timecourse(*param_set)

    # combine into one datastructure
    full_data = zip(somedata, trajectories)

    print 'generated full dataset, proceeding to dmaps'

    # neps = 5 # number of epsilons to evaluate
    # epsilons = np.logspace(-3, 2, neps)
    # kernels = [DMAPS_Data_Kernel(epsilon) for epsilon in epsilons]
    # # investigate proper choice of epsilon
    # dmaps.kernel_plot(kernels, epsilons, full_data) # use un-logged data, as kernel explicitly takes log of parameters

    # perform dmaps, try epsilon=1e-1
    k = 20
    epsilon = 1e-1
    kernel = DMAPS_Data_Kernel(epsilon)
    eigvals, eigvects = dmaps.embed_data_customkernel(full_data, k, kernel)
    np.savetxt('./data/data-dmaps-eigvals.csv', eigvals.real, delimiter=',')
    np.savetxt('./data/data-dmaps-eigvects.csv', eigvects.real, delimiter=',')
    np.savetxt('./data/data-dmaps-params.csv', somedata, delimiter=',')
    print 'saved dmaps output as ./data/data-dmaps...'
    


def dmaps_param_set_grad_kernel():
    """DMAP a collection of sloppy parameter combinations using a kernel which accounts for objective function value and should, ideally, uncover the important parameter(s) in the model"""

    # set up base model
    A0 = 1.0 # initial concentration of A
    k1_true = 1.0
    kinv_true = 1000.0
    k2_true = 1000.0
    decay_rate = k1_true*k2_true/(kinv_true + k2_true) # effective rate constant that governs exponential growth rate
    # start at t0 = 0, end at tf*decay_rate = 4
    ntimes = 20 # arbitrary
    times = np.linspace(0, 4/decay_rate, ntimes)
    model = Rawlings_Model(times, A0, k1_true, kinv_true, k2_true, using_sympy=True)

    # import existing data
    data = np.genfromtxt('./data/params-ofevals.csv', delimiter=',')

    of_tol = 0.4 # from plotting with scratch.py
    somedata = data[data[:,0] < of_tol]
    # only keep npts points due to computational considerations
    npts = 6000
    slice_size = somedata.shape[0]/npts
    somedata = somedata[::slice_size]
    # keff = somedata[:,1]*somedata[:,3]/(somedata[:,2] + somedata[:,3])

    log_params_data = np.log10(somedata[:,1:])
    # add some noise
    noise_level = 0.02
    log_params_data = log_params_data + noise_level*np.random.normal(size=log_params_data.shape)
    # log_params_data = np.log10(data[:,1:])

    # evaluate various epsilons for DMAP kernel
    neps = 5 # number of epsilons to evaluate
    epsilons = np.logspace(-3, 2, neps)
    kernels = [DMAPS_Gradient_Kernel(epsilon, model.sympy_lsq_of_gradient) for epsilon in epsilons]
    # # investigate proper choice of epsilon
    # plot_dmaps.kernel_plot(kernels, epsilons, somedata[:,1:]) # use un-logged data if using gradient of ob. fn. in kernel
    # DMAP with o.f. kernel, appears the epsilon = 20 is appropriate
    epsilon = 20.0
    kernel = DMAPS_Gradient_Kernel(epsilon, model.sympy_lsq_of_gradient)
    k = 15
    eigvals, eigvects = dmaps.embed_data_customkernel(somedata[:,1:], k, kernel)
    plot_dmaps.plot_xyz(somedata[:,1], somedata[:,2], somedata[:,3], color=eigvects[:,1])
    plot_dmaps.plot_xyz(somedata[:,1], somedata[:,2], somedata[:,3], color=eigvects[:,2])

    
def test_sympy_of():
    # set up base model
    A0 = 1.0 # initial concentration of A
    k1_true = 1.0
    kinv_true = 1000.0
    k2_true = 1000.0
    decay_rate = k1_true*k2_true/(kinv_true + k2_true) # effective rate constant that governs exponential growth rate
    # start at t0 = 0, end at tf*decay_rate = 4
    ntimes = 20 # arbitrary
    times = np.linspace(0, 4/decay_rate, ntimes)
    model = Rawlings_Model(times, A0, k1_true, kinv_true, k2_true, using_sympy=True)
    # set up samples of k1, kinv, and k2
    nk1s = 20
    k1s = np.logspace(-2, 2, nk1s)
    nkinvs = 20
    kinvs = np.logspace(0, 5, nkinvs)
    nk2s = 20
    k2s = np.logspace(0, 5, nk2s)
    nks = nk1s*nkinvs*nk2s # total number of parameter space samples
    nks_per_proc = (nks/nk1s + 1)*nkinvs*nk2s # add one to avoid rounding errors
    # iterate over all possible combinations of parameters and record obj. fn. eval
    params_and_of_evals = np.empty((nks_per_proc, 4))
    error = 0
    for k1 in k1s:
        for kinv in kinvs:
            for k2 in k2s:
                error = error + np.abs(model.lsq_of(k1, kinv, k2) - model.sympy_lsq_of_f((k1, kinv, k2)))
    print 'error in sympy function:', error
    

def do_the_right_thing():
    """A completely unecesssary function that does the right thing"""
    sample_param_grid_mpi()
    # dmaps_param_set()
    # test_sympy_of()
    # dmaps_param_set_grad_kernel()
    # dmaps_param_set()
    # abc_analytical_contour()
    # qssa_comparison()
    # plot_data_dmaps_results()
    # dmaps_param_set_data_kernel()
    

if __name__=='__main__':
    do_the_right_thing()
