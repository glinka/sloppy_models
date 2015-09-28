"""Rawlings model-related parameter space investigations

Contact: holiday@alexanderholiday.com
"""
import os
import numpy as np
from mpi4py import MPI

from model import Rawlings_Model
import util_fns as uf
import dmaps
import plot_dmaps

def sample_param_grid_mpi():
    """Samples k1, kinv, k2 over a lattice and records obj. fn. evaluations. A starting point for future analysis"""
    # set up mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

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
    nk1s = 150
    k1s = np.logspace(-2, 2, nk1s)
    nkinvs = 150
    kinvs = np.logspace(0, 5, nkinvs)
    nk2s = 150
    k2s = np.logspace(0, 5, nk2s)
    nks = nk1s*nkinvs*nk2s # total number of parameter space samples
    nks_per_proc = (nks/nk1s + 1)*nkinvs*nk2s # add one to avoid rounding errors
    # iterate over all possible combinations of parameters and record obj. fn. eval
    params_and_of_evals = np.empty((nks_per_proc, 4))
    count = 0
    for k1 in uf.parallelize_iterable(k1s, rank, nprocs):
        for kinv in kinvs:
            for k2 in k2s:
                params_and_of_evals[count] = model.lsq_of(k1, kinv, k2), k1, kinv, k2
                count = count + 1

    all_params_and_of_evals = comm.gather(params_and_of_evals[:count], root=0)
    if rank == 0:
        all_params_and_of_evals = np.concatenate(all_params_and_of_evals)
        print all_params_and_of_evals.shape
        np.savetxt('./data/params_and_of_evals.csv', all_params_and_of_evals, delimiter=',')


def dmaps_param_set():
    """Performs DMAP of log(parameter) set that fall within some ob. fn. tolerance"""
    # import data and save only those parameter combinations such that error(k1, kinv, k2) < tol
    data = np.genfromtxt('./data/params_and_of_evals.csv', delimiter=',')
    of_tol = 0.002 # from plotting with scratch.py
    data = data[data[:,0] < of_tol]
    slice_size = 4 # used to further trim data
    data = data[::slice_size]
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
    filename_id = 'tol-' + str(of_tol) + '-k-' + str(k)
    found_previous_embeddings = False

    eigvals, eigvects = None, None

    for filename in os.listdir('./data'):
        if filename_id in filename:
            # found previously saved data, import and do not recompute
            eigvects = np.genfromtxt('./data/dmaps-eigvects--tol-' + str(of_tol) + '-k-' + str(k) + '.csv', delimiter=',')
            eigvals = np.genfromtxt('./data/dmaps-eigvals--tol-' + str(of_tol) + '-k-' + str(k) + '.csv', delimiter=',')
            found_previous_embeddings = True
            break

    if found_previous_embeddings is False:
        eigvals, eigvects = dmaps.embed_data(log_params_data, k)
        np.savetxt('./data/dmaps-eigvects--tol-' + str(of_tol) + '-k-' + str(k) + '.csv', eigvects, delimiter=',')
        np.savetxt('./data/dmaps-eigvals--tol-' + str(of_tol) + '-k-' + str(k) + '.csv', eigvals, delimiter=',')

    plot_dmaps.plot_xyz(log_params_data[:,0], log_params_data[:,1], log_params_data[:,2], color=eigvects[:,1])
    plot_dmaps.plot_xyz(log_params_data[:,0], log_params_data[:,1], log_params_data[:,2], color=eigvects[:,2])
    # plot_dmaps.plot_embeddings(eigvects, eigvals)

class DMAPS_Kernel:
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
    

def dmaps_param_set_custom_kernel():
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
    data = np.genfromtxt('./data/params_and_of_evals.csv', delimiter=',')

    of_tol = 0.4 # from plotting with scratch.py
    somedata = data[data[:,0] < of_tol]
    # only keep npts points due to computational considerations
    npts = 6000
    slice_size = somedata.shape[0]/npts
    somedata = somedata[::slice_size]
    keff = somedata[:,1]*somedata[:,3]/(somedata[:,2] + somedata[:,3])

    log_params_data = np.log10(somedata[:,1:])
    # add some noise
    noise_level = 0.02
    log_params_data = log_params_data + noise_level*np.random.normal(size=log_params_data.shape)
    log_params_data = np.log10(data[:,1:])

    # evaluate various epsilons for DMAP kernel
    neps = 5 # number of epsilons to evaluate
    epsilons = np.logspace(-3, 2, neps)
    kernels = [DMAPS_Kernel(epsilon, model.sympy_lsq_of_gradient) for epsilon in epsilons]
    # # investigate proper choice of epsilon
    # plot_dmaps.kernel_plot(kernels, epsilons, somedata[:,1:]) # use un-logged data if using gradient of ob. fn. in kernel
    # DMAP with o.f. kernel, appears the epsilon = 20 is appropriate
    epsilon = 20.0
    kernel = DMAPS_Kernel(epsilon, model.sympy_lsq_of_gradient)
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
    # sample_param_grid_mpi()
    # dmaps_param_set()
    # test_sympy_of()
    dmaps_param_set_custom_kernel()

if __name__=='__main__':
    do_the_right_thing()
