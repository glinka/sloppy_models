"""Analyzes sloppy/unidentifiable parameter sets of the system :math:`y=e^{-k_1k_2*t)` using DMAPS"""

import numpy as np
import dmaps
import dmaps_kernels
import matplotlib.pyplot as plt


def get_traj(k1, k2, times):
    """Returns the trajectory of the system at the given 'times' and parameter values 'k1' and 'k2'"""
    return np.exp(-k1*k2*times)

def of(k1, k2, times, data):
    """Returns ob. fn. value based on new 'k1' and 'k2', compared with 'data' derived from true values. The times at which the new values are sampled given in 'times' should be the same as the times 'data' was drawn. **Based on a least squares objective function.**"""
    return np.power(np.linalg.norm(get_traj(k1, k2, times) - data), 2)
        
def get_parameter_sets(k1, k2, times, tol=1e-3):
    """Locates and returns sloppy parameter combinations of the system centered around true values 'k1' and 'k2'"""
    # generate 'experimental data'
    input_data = get_traj(k1, k2, times)
    # set grid for testing
    k1min, k1max, k2min, k2max = (0.5, 1.5, 0.5, 1.5)
    npts = 100000
    k1pts = k1min + (k1max-k1min)*np.random.uniform(size=npts)
    k2pts = k2min + (k2max-k2min)*np.random.uniform(size=npts)
    pts_count = 0
    pts_kept = np.empty((npts, 3)) # storage for x, y, and ob. fn. evaluations
    # loop through random pts and keep if ob. fn. evaluation is low enough
    for i in range(npts):
        of_eval = of(k1pts[i], k2pts[i], times, input_data)
        if of_eval < tol:
            pts_kept[pts_count,:-1] = k1pts[i], k2pts[i]
            pts_kept[pts_count,-1] = of_eval
            pts_count += 1

    print pts_count

    return pts_kept[:pts_count]

def dmaps_sloppy_params():
    # set param values
    k1 = 1; k2 = 1; times = np.linspace(0,4,5)
    tol = 1e-2
    # get sloppy parameter sets
    sloppy_params = get_parameter_sets(k1, k2, times, tol=tol)

    # # visualize param set
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(sloppy_params[:,0], sloppy_params[:,1])
    # ax.set_xlabel(r'$k_1$')
    # ax.set_ylabel(r'$k_2$')
    # ax.set_title(r'$C(k_1, k_2) < ' + str(tol) + '$')
    # plt.show(fig)

    # test different epsilons in the custom kernel
    nepsilons = 10
    epsilons = np.logspace(0, 4, nepsilons)
    kernels = [dmaps_kernels.objective_function_kernel(epsilon) for epsilon in epsilons]
    dmaps.kernel_plot(kernels, epsilons, sloppy_params)

    # perform dmap
    epsilon = 1e1
    of_kernel = dmaps_kernels.objective_function_kernel(epsilon)
    k = 4
    eigvals, eigvects = dmaps.embed_data_customkernel(sloppy_params, k, of_kernel)

    # plot output, color param plane by output eigvectors
    for i in range(1,k):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(sloppy_params[:,0], sloppy_params[:,1], c=eigvects[:,i])
        ax.set_xlabel(r'$k_1$')
        ax.set_ylabel(r'$k_2$')
        ax.set_title(r'$C(k_1, k_2) < ' + str(tol) + '$')
        plt.show(fig)


if __name__=="__main__":
    dmaps_sloppy_params()
