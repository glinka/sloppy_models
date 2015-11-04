"""Analyzes sloppy/unidentifiable parameter sets of the system :math:`y=e^{-k_1k_2*t)` using DMAPS"""

import numpy as np
import dmaps
import plot_dmaps
import dmaps_kernels
import matplotlib.pyplot as plt


class Model:
    """Collects basic info about system into a class, provides helpful functions for objective function analysis

    Attributes:
        k1_true (float): true value of parameter 'k1' with which true data will be generated
        k2_true (float): true value of parameter 'k2' with which true data will be generated
        times (array): times at which to record data
        data (array): trajectory of model: :math:`\{e^{-k_1k_2*t_i)\}_i`
    """
    def __init__(self, k1_true, k2_true, times):
        self._k1 = k1_true
        self._k2 = k2_true
        self._times = times
        self._data = self.get_traj(k1_true, k2_true)
        

    def get_traj(self, k1, k2):
        """Returns the trajectory of the system at the given 'times' and parameter values 'k1' and 'k2'"""
        return np.exp(-k1*k2*self._times)

    def of(self, k1, k2):
        """Returns ob. fn. value based on new 'k1' and 'k2', compared with 'data' derived from true values. The times at which the new values are sampled given in 'times' should be the same as the times 'data' was drawn. **Based on a least squares objective function.**"""
        return np.power(np.linalg.norm(self.get_traj(k1, k2) - self._data), 2)

    def grad(self, ks):
        """Returns gradient of system's least-squares objective function, evaluated at parameters 'ks'

        Args:
            ks (array): array containing parameter values at which to evaluate the gradient, ks = (k1, k2)
        """
        traj = self.get_traj(ks[0], ks[1])
        common_vals = (self._data - traj)*traj
        g1val = ks[1]*self._times
        g2val = ks[0]*self._times
        grad = 2*np.array((np.sum(common_vals*g1val), np.sum(common_vals*g2val)))
        return grad
    
    def get_parameter_sets(self, tol=1e-4):
        """Locates and returns sloppy parameter combinations of the system centered around true values 'k1' and 'k2'"""
        # set grid for testing
        k1min, k1max, k2min, k2max = (0.5, 1.5, 0.5, 1.5)
        npts = 100000
        k1pts = k1min + (k1max-k1min)*np.random.uniform(size=npts)
        k2pts = k2min + (k2max-k2min)*np.random.uniform(size=npts)
        pts_count = 0
        pts_kept = np.empty((npts, 2)) # storage for x, y, and ob. fn. evaluations
        # loop through random pts and keep if ob. fn. evaluation is low enough
        for i in range(npts):
            of_eval = self.of(k1pts[i], k2pts[i])
            if of_eval < tol:
                pts_kept[pts_count] = k1pts[i], k2pts[i]
                pts_count += 1

        print 'generated a sample of', pts_count, 'sloppy parameter combinations'
        return pts_kept[:pts_count]

def dmaps_annulus():
    """Uses Lafon DMAP to generate embedding of annulus such that eigenvectors are constant on level sets, f = x^2 + y^2"""
    # generate dataset
    npts = 3000
    # rs = np.random.uniform(low=0.5, high=1.5, size=npts)
    # thetas = np.random.uniform(high=2*np.pi, size=npts)
    # data = np.array((rs*np.cos(thetas), rs*np.sin(thetas))).T
    data = np.random.uniform(low=0.1, size=(npts,2))
    grad = lambda x: 2*x
    
    # visualize param set
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1])
    plt.show(fig)

    # perform dmap
    epsilon = 1e-5
    kernel = dmaps_kernels.gradient_kernel(epsilon, grad)
    k = 12
    eigvals, eigvects = dmaps.embed_data_customkernel(data, k, kernel)

    # plot output, color param plane by output eigvectors
    for i in range(1,8):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:,0], data[:,1], c=eigvects[:,i])
        plt.show(fig)


def dmaps_sloppy_params():
    # set param values
    k1 = 1; k2 = 1; times = np.linspace(0,4,5)
    tol = 1e-4
    # # get sloppy parameter sets
    sloppy_system = Model(k1, k2, times)
    npts = 3000
    # k1pts = 0.98 + (0.04)*np.random.uniform(size=npts)
    # k2pts = 0.95 + (0.1)*np.random.uniform(size=npts)
    # sloppy_params = np.array((k1pts, k2pts)).T # 0.95 + (0.1)*np.random.uniform(size=(npts,2))

    # load saved data, don't recompute dmap or regenerate data
    eigvals = np.genfromtxt('./data/eigvals.csv', delimiter=',')
    eigvects = np.genfromtxt('./data/eigvects.csv', delimiter=',')
    sloppy_params = np.genfromtxt('./data/sloppy-params.csv', delimiter=',')

    # visualize param set
    fig = plt.figure()
    ax = fig.add_subplot(111)
    of_evals = np.empty(npts)
    grad_evals = np.empty((npts, 2))
    for i, pt in enumerate(sloppy_params):
        of_evals[i] = sloppy_system.of(pt[0], pt[1])
        grad_evals[i] = sloppy_system.grad(pt)
    ax.scatter(sloppy_params[:,0], sloppy_params[:,1], c=of_evals)
    ax.set_xlabel(r'$k_1$')
    ax.set_ylabel(r'$k_2$')
    ax.set_title(r'$C(k_1, k_2) < ' + str(tol) + '$')
    plt.show(fig)

    # # visualize gradients
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for i in range(npts):
    #     ax.plot((sloppy_params[i,0], sloppy_params[i,0] + grad_evals[i,0]), (sloppy_params[i,1], sloppy_params[i,1] + grad_evals[i,1]))
    # ax.set_xlabel(r'$k_1$')
    # ax.set_ylabel(r'$k_2$')
    # ax.set_title(r'$C(k_1, k_2) < ' + str(tol) + '$')
    # plt.show(fig)

    # test different epsilons in the custom kernel

    # nepsilons = 5
    # epsilons = np.logspace(-8, -1, nepsilons)
    # kernels = [dmaps_kernels.gradient_kernel(epsilon, sloppy_system.grad) for epsilon in epsilons]
    # dmaps.kernel_plot(kernels, epsilons, sloppy_params)


    # # test gradient
    # W = np.empty((npts, npts))
    # K = np.empty(npts)
    # test_pt = sloppy_params[1500]
    # plt.scatter(sloppy_params[:,0], sloppy_params[:,1], lw=0)
    # plt.scatter(test_pt[0], test_pt[1], s=100)
    # plt.show()
    # for i in range(npts):
    #     K[i] = np.power(np.dot(sloppy_system.grad(test_pt), test_pt - sloppy_params[i]), 2)
    #     for j in range(npts):
    #         W[i,j] = np.linalg.norm(sloppy_params[i] - sloppy_params[j]) # np.power(np.dot(sloppy_system.grad(sloppy_params[i]), sloppy_params[i] - sloppy_params[j]), 2)
    # plt.hist(K)
    # plt.show()
    # plt.hist(W)
    # plt.show()

    # # perform dmap
    # epsilon = 1e-5
    # kernel = dmaps_kernels.gradient_kernel(epsilon, sloppy_system.grad)
    k = 5
    # eigvals, eigvects = dmaps.embed_data_customkernel(sloppy_params, k, kernel)
    # np.savetxt('./data/eigvals.csv', eigvals, delimiter=',')
    # np.savetxt('./data/eigvects.csv', eigvects, delimiter=',')
    # np.savetxt('./data/sloppy-params.csv', sloppy_params, delimiter=',')

    # plot output, color param plane by output eigvectors
    for i in range(1,k):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(sloppy_params[:,0], sloppy_params[:,1], c=eigvects[:,i])
        ax.set_xlabel(r'$k_1$')
        ax.set_ylabel(r'$k_2$')
        ax.set_title(r'$C(k_1, k_2) < ' + str(tol) + '$')
        plt.show(fig)

    plot_dmaps.plot_embeddings(eigvects, eigvals, k=5, color=of_evals)
        

if __name__=="__main__":
    dmaps_sloppy_params()
    # dmaps_annulus()
