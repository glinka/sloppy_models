import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint

# import pyximport; pyximport.install()
import Z_Model as ZM

import util_fns as uf

def get_of(sample_traj, true_traj):
    return np.linalg.norm(sample_traj - true_traj)

def check_rhs():

    # specify ode parameters
    params = np.array((0.1, 0.5, 0.01, 0.1)) # (a, b, lambda, epsilon)
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

    # set up sampling grid and storage space for obj. fn. evals
    nsamples_per_axis = 20
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
    check_rhs()
