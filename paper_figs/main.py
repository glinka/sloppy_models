# numpy
import numpy as np
# scipy
from scipy.optimize import minimize
# matplotlib
from matplotlib import colors, colorbar, cm, pyplot as plt, gridspec as gs, tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# mpi
from mpi4py import MPI
# utilities
import os

import colormaps
import dmaps
import plot_dmaps
from dmaps_kernels import data_kernel
from util_fns import progress_bar
from rawlings_model.main import dmaps_param_set
from zagaris_model import Z_Model as ZM
from algorithms import Integration
import algorithms.CustomErrors as CustomErrors
import util_fns as uf



def henon(x0, y0, n, a, b):
    if n > 0:
        return henon(1 - a*x0*x0 + y0, b*x0, n-1, a, b)
    else:
        return [x0, y0]

def henon_inv(xn, yn, n, a, b):
    if n > 0:
        return henon_inv(yn/b, xn - 1 + a*np.power(yn/b, 2), n-1, a, b)
    else:
        return [xn, yn]

def henon_3inv(x3, y3, a, b):
    y0 = (x3 - 1 + a*np.power(y3/b, 2))/b - 1 + a*np.power((y3/b - 1 + a*np.power((x3 - 1 + a*np.power(y3/b, 2))/b, 2))/b, 2)
    x0 = (y3/b - 1 + a*np.power((x3 - 1 + a*np.power(y3/b, 2))/b, 2))/b
    return x0, y0

def henon_2inv(x2, y2, a, b):
    y0 = y2/b - 1 + a*np.power((x2 - 1 + a*np.power(y2/b, 2))/b, 2)
    x0 = (x2 - 1 + a*np.power(y2/b, 2))/b
    return x0, y0

class Transform_Zagaris_Model:
    """Uses Henon map to transform params (alpha, lambda), keeps (beta, epsilon) constant, provides coresponding of"""
    def __init__(self, base_params, times, x0, n, a, b):
        self._beta = base_params[1]
        self._epsilon = base_params[3]
        self._system = ZM.Z_Model(base_params)
        self._true_trajectory = self._system.get_trajectory_quadratic(x0, times)
        self._n = n
        self._a = a
        self._b = b
        self._times = times
        self._x0 = x0

    def of(self, params):
        # NOTE THAT LAMBDA IS RETURNED FIRST, THEN ALPHA
        lam, alpha = henon_inv(params[0], params[1], self._n, self._a, self._b)
        self._system.change_parameters(np.array((alpha, self._beta, lam, self._epsilon)))
        try:
            new_traj = self._system.get_trajectory_quadratic(self._x0, self._times)
        except CustomErrors.IntegrationError:
            raise
        else:
            return np.linalg.norm(new_traj - self._true_trajectory)
    
    def henon_2inv_dmaps_metric(self, transformed_params1, transformed_params2):
        """Inverts transformed params twice to uncover alpha, lambda, then take distance there"""
        # NOTE THAT LAMBDA IS RETURNED FIRST, THEN ALPHA
        lam_alpha1 = np.array(henon_inv(transformed_params1[0], transformed_params1[1], self._n, self._a, self._b))
        lam_alpha2 = np.array(henon_inv(transformed_params2[0], transformed_params2[1], self._n, self._a, self._b))
        return np.linalg.norm(lam_alpha2 - lam_alpha1)

class Normal_Zagaris_Model:
    """Uses Henon map to transform params (alpha, lambda), keeps (beta, epsilon) constant, provides coresponding of"""
    def __init__(self, base_params, times, x0):
        self._beta = base_params[1]
        self._epsilon = base_params[3]
        self._system = ZM.Z_Model(base_params)
        self._true_trajectory = self._system.get_trajectory_quadratic(x0, times)
        self._times = times
        self._x0 = x0

    def of(self, params):
        # NOTE THAT LAMBDA IS RETURNED FIRST, THEN ALPHA
        lam, alpha = params
        self._system.change_parameters(np.array((alpha, self._beta, lam, self._epsilon)))
        try:
            new_traj = self._system.get_trajectory_quadratic(self._x0, self._times)
        except CustomErrors.IntegrationError:
            raise
        else:
            return np.linalg.norm(new_traj - self._true_trajectory)


def transformed_param_space_fig():
    """Perform DMAP on nonlinear, henon-mapped parameters lambda/epsilon (important/sloppy)"""
    
    # CREATE DATASET (no dataset exists):
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # set model params
    (a_true, b_true, lam_true, eps_true) = (1.0, 0.01, 1.0, 0.001) # (0.1, 0.01, 0.1, 0.001)
    params = np.array((a_true, b_true, lam_true, eps_true))

    # set up integration times
    t0 = 0
    tf = 1
    dt = 0.1
    times = np.arange(0, tf, dt) + dt
    ntimes = times.shape[0]

    # get true trajectory based on true initial conditions
    x0 = np.array((1, a_true))

    # set henon transform params
    nhenon_transforms = 2
    a = 1.3
    b = 0.3

    # create system with given params
    z_system = Transform_Zagaris_Model(params, times, x0, nhenon_transforms, a, b)

    # assume if not running on multiple procs, then we already have data. otherwise re-generate
    if nprocs > 1:

        # perform optimization in transformed parameter space
        do_transformed_optimization = False
        if do_transformed_optimization:
            if rank == 0:
                print 'running in parallel, generating transformed x2, y2 parameters through repeated optimization'
            # optimize over range of TRANSFORMED initial conditions
            nsamples = 8000
            data = np.empty((nsamples, 3))
            x2_y2_samples = np.random.uniform(size=(nsamples, 2))*np.array((6,1)) + np.array((-4, -0.5)) # a \in (7, 9) lamb \in (6, 11)
            count = 0
            tol = 1.0
            for params in uf.parallelize_iterable(x2_y2_samples, rank, nprocs):
                try:
                    result = minimize(z_system.of, params, method='SLSQP', tol=tol, options={'ftol' : tol})
                except CustomErrors.IntegrationError:
                    continue
                else:
                    if result.success:
                        data[count] = (result.x[0], result.x[1], result.fun)
                        count = count + 1

            data = data[:count]
            all_data = comm.gather(data, root=0)

            if rank is 0:
                all_data = np.concatenate(all_data)
                all_data.dump('./data/x2-y2-ofevals-2016.csv')
                print '******************************'
                print 'Data saved in ./data/x2-y2-ofevals-2016.csv, rerun to perform DMAP'
                print '******************************'


        # perform optimization in normal parameter space
        do_normal_optimization = True
        if do_normal_optimization:
            if rank == 0:
                print 'running in parallel, generating normal alpha, lambda parameters through repeated optimization'

            z_system = Normal_Zagaris_Model(params, times, x0)

            nsamples = 8000
            data = np.empty((nsamples, 3))
            scale = 20
            a_lam_samples = np.random.uniform(size=(nsamples,2))*np.array((scale, scale)) # a \in [0,scale], b \in [0,scale]

            count = 0
            # tol = 0.01
            for params in uf.parallelize_iterable(a_lam_samples, rank, nprocs):
                try:
                    # result = minimize(z_system.of, params, method='SLSQP', tol=tol, options={'ftol' : tol})
                    result = minimize(z_system.of, params, method='SLSQP')
                except CustomErrors.IntegrationError:
                    continue
                else:
                    if result.success:
                        data[count] = (result.x[0], result.x[1], result.fun)
                        count = count + 1

            data = data[:count]
            all_data = comm.gather(data, root=0)

            if rank is 0:
                all_data = np.concatenate(all_data)
                all_data.dump('./data/a-lam-ofevals-2016.csv')
                print '******************************'
                print 'Data saved in ./data/a-lam-ofevals-2016.csv, rerun to perform DMAP'
                print '******************************'
        
    else:
        print 'loading pre-existing data from ./data/x2-y2-ofevals-2016.csv (x2, y2 values from repeated optimization in transformed param space)'

        # perform the dmap on a selection of the optimization results

        # slim data, spurious results due to overflow
        data = np.load('./data/x2-y2-ofevals-2016.csv')
        # extract sloppy parameter combinations
        tol = 2.0
        x2_y2_of = data[data[:,-1] < tol]
        npts = x2_y2_of.shape[0]
        print 'Performing analysis on', npts, 'points found through optimization'

        # # plot output
        scatter_size = 50
        # plot x2, y2 colored by obj. fn.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x2_y2_of[:,0], x2_y2_of[:,1], c=np.log10(x2_y2_of[:,2]), s=scatter_size)
        axins = inset_axes(ax, 6, 5, bbox_to_anchor=(0.45, 0.75), bbox_transform=ax.transAxes)
        axins.scatter(x2_y2_of[:,0], x2_y2_of[:,1], c=np.log10(x2_y2_of[:,2]), s=scatter_size)
        axins.set_xlim((-0.5, 1.25))
        axins.set_ylim((0, 0.35))
        axins.tick_params(axis='both', which='major', labelsize=0)
        ax.set_xlabel(r'$p_1$')
        ax.set_ylabel(r'$p_2$')
        fig.subplots_adjust(bottom=0.15)
        mark_inset(ax, axins, 1, 4 , fc='none', ec='0.5', zorder=3)
        plt.show()

        # # investigate where points lie in (alpha, lambda) space, color by of
        lam, alpha = henon_inv(x2_y2_of[:,0], x2_y2_of[:,1], nhenon_transforms, a, b)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(lam, alpha, c=np.log10(x2_y2_of[:,2]), s=scatter_size)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\alpha$')
        fig.subplots_adjust(bottom=0.15)

        plt.show()

        # do dmaps in transformed space
        # eps = 0.6
        # k = 12
        # eigvals, eigvects = dmaps.embed_data(x2_y2_of[:,:2], k=k, epsilon=eps)
        # for i in range(1,5):
        #     plot_dmaps.plot_xy(x2_y2_of[:,0], x2_y2_of[:,1], color=eigvects[:,i], scatter=True, xlabel=r'$x_2$', ylabel=r'$y_2$', s=scatter_size)

        # # do dmaps in "correct" transformation (i.e. in the original alpha, lambda)
        # eps = 0.7
        # eigvals, eigvects = dmaps.embed_data(x2_y2_of[:,:2], k=k, epsilon=eps, metric=z_system.henon_2inv_dmaps_metric)
        # plot_dmaps.plot_xy(eigvects[:,1], eigvects[:,4], color=np.log10(x2_y2_of[:,2]), scatter=True, xlabel=r'$\Phi_1$', ylabel=r'$\Phi_4$', s=scatter_size)
        # for i in range(1,5):
        #     plot_dmaps.plot_xy(x2_y2_of[:,0], x2_y2_of[:,1], color=eigvects[:,i], scatter=True, xlabel=r'$x_2$', ylabel=r'$y_2$', s=scatter_size)


def comp_coeffs(coeffs, n):
    if n > 0:
        if coeffs.shape[0] - n == 1:
            coeffs[:2] = 1
        else:
            index = coeffs.shape[0] - n + 1
            temp_coeffs = np.empty(index)
            for i in range(index):
                temp_coeffs[i] = coeffs[i-1] + coeffs[i]
            coeffs[:index] = temp_coeffs
        return comp_coeffs(coeffs, n-1)
    else:
        return coeffs


def method_of_lines(L, f, n, dt):
    """Computes 'f' after 'n' steps of size 'dt', based on Laplace eqn. with Neumann BCs, could use recursion bobursion for coolness"""
    coeffs = comp_coeffs(np.zeros(n+1), n)
    print coeffs
    f_t = np.copy(f)
    Lf = np.dot(L, f_t)
    for i in range(1, n+1):
        f = f + coeffs[i]*Lf*np.power(dt, i)
        Lf = np.dot(L, Lf)
    return f
    
def discretized_laplacian_dmaps():
    """Compares the difference between a finite difference approximation of the Laplace operator with the W matrix used in DMAPS"""
    # dmaps, this assumes J = K = 0: the paper specifies J = 0, but doesn't specify what K should be represented with in this case
    n = 100
    dmaps_w = np.zeros((n,n))
    for i in range(1,n-1):
        dmaps_w[i,i+1] = 1
        dmaps_w[i,i-1] = 1
    dmaps_w[0,0] = 1
    dmaps_w[0,1] = 1
    dmaps_w[n-1,n-2] = 1
    dmaps_w[n-1,n-1] = 1
    alphas_dmaps = np.array([(np.pi*i)/n for i in range(n)])
    omegas_dmaps = (np.pi - alphas_dmaps)/2.0
    eigvals_dmaps = 2*np.cos(alphas_dmaps)
    eigvects_dmaps = np.sin(np.outer(alphas_dmaps, np.arange(1,n+1)) + np.outer(omegas_dmaps, np.ones(n))).T
    eigvects_dmaps = eigvects_dmaps/np.linalg.norm(eigvects_dmaps, axis=0)
    print 'error in dmaps eigendecomp:', np.linalg.norm(np.dot(dmaps_w, eigvects_dmaps) - np.dot(eigvects_dmaps, np.diag(eigvals_dmaps)))

    # discrete laplacian
    discrete_laplacian = np.zeros((n,n))
    for i in range(1,n-1):
        discrete_laplacian[i,i+1] = 1
        discrete_laplacian[i,i-1] = 1
    discrete_laplacian[0,1] = 2
    discrete_laplacian[n-1,n-2] = 2
    alphas_lap = np.empty(n)
    alphas_lap[:-1] = [i*np.pi/(n-1) for i in range(n-1)]
    alphas_lap[-1] = np.pi
    omegas_lap = np.empty(n)
    omegas_lap[:-1] = (np.pi - 2*alphas_lap[:-1])/2.0
    omegas_lap[-1] = np.pi/2
    eigvals_lap = 2*np.cos(alphas_lap)
    eigvects_lap = np.sin(np.outer(alphas_lap, np.arange(1,n+1)) + np.outer(omegas_lap, np.ones(n))).T
    eigvects_lap = eigvects_lap/np.linalg.norm(eigvects_lap, axis=0)
    print 'error in discrete laplacian eigendecomp:', np.linalg.norm(np.dot(discrete_laplacian, eigvects_lap) - np.dot(eigvects_lap, np.diag(eigvals_lap)))

    # # plot eigenvalues
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(range(n), eigvals_dmaps, color='b')
    # ax.scatter(range(n), eigvals_dmaps, label='DMAPS', color='b')
    # ax.plot(range(n), eigvals_lap, color='r')
    # ax.scatter(range(n), eigvals_lap, label='Discretization', color='r')
    # plt.show()

    # # plot differences in eigenvectors as measured by norm
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(range(n), np.linalg.norm(eigvects_lap - eigvects_dmaps, axis=0))
    # ax.scatter(range(n), np.linalg.norm(eigvects_lap - eigvects_dmaps, axis=0))
    # plt.show()
    
    # # make the actual matrices
    # dmaps
    delta = 0.1 # delta = exp(-(dx/eps)^2), ideally a small number
    W = delta*dmaps_w
    for i in range(1,n-1):
        W[i,i] = 1 - 2*delta
    W[0,0] = 1 - delta
    W[n-1,n-1] = 1 - delta
    eigvals_dmaps = delta*(eigvals_dmaps + 1/delta - 2)
    # W = W/delta

    # laplacian
    L = discrete_laplacian
    for i in range(n):
        L[i,i] = -2
    dx = 0.04
    L = L/(dx*dx)
    eigvals_lap = (eigvals_lap - 2)/(dx*dx)

    # examine result of applying these operators to 1-d data

    dt = 0.1
    npts = 100
    xmax = 4
    xs = np.linspace(0,xmax,npts)
    T0 = np.exp(xs - xmax)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, T0, c='c')
    ax.scatter(xs, method_of_lines(L, T0, 1, dt), c='b')
    ax.scatter(xs, np.dot(W, T0), c='r')
    plt.show()
    
    

class FormatAxis:
    def __init__(self, ax):
        self._ax = ax

    def format(self, axis, data, format_string, offset=0, nticks=5):
        # dictionary of relevant functions/attributes
        d = {'x':{'ax':self._ax.xaxis, 'set-tick-pos':self._ax.set_xticks, 'set-tick-labels':self._ax.set_xticklabels, 'tick-pos':self._ax.xaxis.get_majorticklocs()},
             'y':{'ax':self._ax.yaxis, 'set-tick-pos':self._ax.set_yticks, 'set-tick-labels':self._ax.set_yticklabels, 'tick-pos':self._ax.yaxis.get_majorticklocs()},
             'z':{'ax':self._ax.zaxis, 'set-tick-pos':self._ax.set_zticks, 'set-tick-labels':self._ax.set_zticklabels, 'tick-pos':self._ax.get_zticks()}}
        # tick positions are constant, regardless of offset
        maxval, minval = np.min(data), np.max(data)
        increment = (maxval - minval)/(nticks-1)
        d[axis]['set-tick-pos']([minval + i*increment for i in range(nticks)])
        # subtract offset from data if using
        if offset != 0:
            if offset < 0:
                offset_str =  '- %1.2f' % abs(offset) # unicode dash u'\u2012'
            else:
                offset_str = '+ %1.2f' % abs(offset)
            # go through the terrible process of figuring out where to put the damn thing
            loc = {'x':0, 'y':0, 'z':0}
            for i,key in enumerate(d.keys()):
                if key is axis:
                    if axis is 'x':
                        loc[key] = np.min(d[key]['tick-pos']) - 0.00*(np.max(d[key]['tick-pos']) - np.min(d[key]['tick-pos']))
                    else:
                        loc[key] = np.max(d[key]['tick-pos']) + 0.00*(np.max(d[key]['tick-pos']) - np.min(d[key]['tick-pos']))
                else:
                    if key is 'x':
                        loc[key] = np.max(d[key]['tick-pos']) + 0.2*(np.max(d[key]['tick-pos']) - np.min(d[key]['tick-pos']))
                    else:
                        loc[key] = np.min(d[key]['tick-pos']) - 0.2*(np.max(d[key]['tick-pos']) - np.min(d[key]['tick-pos']))
            self._ax.text(loc['x'], loc['y'], loc['z'], offset_str, fontsize=12) #maxval-0.05*(maxval-minval)
            data = data - offset
        # set axis tick labels
        minval = np.min(data)
        d[axis]['set-tick-labels']([format_string % (minval + i*increment) for i in range(nticks)])


def two_effective_one_neutral_dmaps_fig():
    """Plots the mixed kernel dmaps results of antonios' system fixing 'a' and varying 'a', 'eps' and 'lambda', thus a system with two effective and one neutral parameter. the mixed dmaps first uncovers the effective 'a', 'lambda' params. based on zagaris_model.main.py's dmaps_two_important_one_sloppy_only_data()"""
    data_dir = '/home/alexander/workspace/sloppy_models/zagaris_model/data/'
    params = np.load(data_dir + 'a-lam-eps-of-params-new.pkl')
    trajs = np.load(data_dir + 'a-lam-eps-trajs-new.pkl')
    tol = 1.5e-4 # 2e-2 for old data
    trajs = trajs[params[:,3] < tol]
    params = params[params[:,3] < tol]
    params = params[:,:3]
    print 'Have', params.shape[0], 'pts in dataset'
    # epsilons = np.logspace(-3, 1, 5)
    # dmaps.epsilon_plot(epsilons, trajs)
    epsilon = 1e-2 # from epsilon plot
    k = 12
    # eigvals, eigvects = dmaps.embed_data(trajs, k, epsilon=epsilon)
    # eigvals.dump(data_dir + 'dmaps-data-kernel-eigvals.pkl')
    # eigvects.dump(data_dir + 'dmaps-data-kernel-eigvects.pkl')
    eigvals = np.load(data_dir + 'dmaps-data-kernel-eigvals.pkl')
    eigvects = np.load(data_dir + 'dmaps-data-kernel-eigvects.pkl')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(np.log10(params[:,0]), np.log10(params[:,1]), np.log10(params[:,2]), c=eigvects[:,1])
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(np.log10(params[:,0]), np.log10(params[:,1]), np.log10(params[:,2]), c=eigvects[:,2])
    for ax in [ax1, ax2]:
        # prevent overly crowded axis ticks and labels
        formatter = FormatAxis(ax)
        formatter.format('x', np.log10(params[:,0]), '%1.3f', nticks=2)
        formatter.format('y', np.log10(params[:,1]), '%1.3f', nticks=2)
        formatter.format('z', np.log10(params[:,2]), '%1.1f')
        # labels
        ax.set_xlabel(r'$a$')
        ax.set_ylabel(r'$\lambda$')
        ax.set_zlabel(r'$\epsilon$')
        # white bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # move labels to avoid overlap with numbers
        ax.xaxis._axinfo['label']['space_factor'] = 2.2
        ax.yaxis._axinfo['label']['space_factor'] = 2.2
        ax.zaxis._axinfo['label']['space_factor'] = 2.2
        # # offset size
        # ax.yaxis.get_children()[1].set_size(20)
    plt.show()


def calc_tri_avg(triangles, data):
    """Calculates the average value of 'data' at each triangle specified in 'triangles'.

    ***'triangles' and 'data' must be of the same length***"""
    avg_data = np.empty(triangles.shape[0])
    for i, t in enumerate(triangles):
        avg_data[i] = np.average(data[t])
    return avg_data

def rawlings_3d_dmaps_fig():
    """Plots **thick** sloppy manifold in parameter space (here found through sampling), which is colored by the first and second DMAP eigenvectors in two separate subfigures on the right, adapted from rawlings_model.main's dmaps_param_set()"""    
    paramdata = np.genfromtxt('/home/alexander/workspace/sloppy_models/rawlings_model/data/data-dmaps-params.csv', delimiter=',')
    keff = paramdata[:,0]*paramdata[:,2]/(paramdata[:,1] + paramdata[:,2])
    paramdata = np.log10(paramdata)
    eigvects = np.genfromtxt('/home/alexander/workspace/sloppy_models/rawlings_model/data/data-dmaps-eigvects.csv', delimiter=',')
    # color params by dmaps coord
    gspec = gs.GridSpec(2,2)
    fig = plt.figure()
    # set up axes
    ax_b = fig.add_subplot(gspec[:,0], projection='3d')
    ax_keff = fig.add_subplot(gspec[0,1], projection='3d')
    ax_dmaps = fig.add_subplot(gspec[1,1], projection='3d')
    # add scatter plots with correct colorings
    ax_b.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c='b', cmap='gnuplot2')
    ax_keff.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c=keff, cmap='gnuplot2')
    ax_dmaps.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c=eigvects[:,3], cmap='gnuplot2')
    for ax in [ax_b, ax_keff, ax_dmaps]:
        # label axes
        ax.set_xlabel('log(' + r'$k_1$' + ')')
        ax.set_ylabel('log(' + r'$k_{-1}$' + ')')
        ax.set_zlabel('log(' + r'$k_2$' + ')')
        # move labels to avoid overlap with numbers
        ax.xaxis._axinfo['label']['space_factor'] = 2.8
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.zaxis._axinfo['label']['space_factor'] = 2.8
        # white bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # # no grid
        # ax.grid(False)
    plt.show()

def rawlings_2d_dmaps_fig():
    """Plots sloppy manifold in parameter space (here found through sampling), which is colored by the first and second DMAP eigenvectors in two separate subfigures on the right, adapted from rawlings_model.main's dmaps_param_set()"""
    # import data
    # eigvals.dump('./data/rawlings-2d-dmaps-eigvals.pkl')
    # eigvects.dump('./data/rawlings-2d-dmaps-eigvects.pkl')
    # log_params_data.dump('./data/rawlings-2d-dmaps-logparams.pkl')
    log_params_data = np.load('./data/rawlings-2d-dmaps-logparams.pkl')
    eigvals = np.load('./data/rawlings-2d-dmaps-eigvals.pkl')
    eigvects = np.load('./data/rawlings-2d-dmaps-eigvects.pkl')

    # # plot figure (three subfigs total)
    gspec = gs.GridSpec(2,2)
    fig = plt.figure()
    xlabel = r'$k_1$'
    ylabel = r'$k_{-1}$'
    zlabel = r'$k_2$'
    stride = 15 # effectively thins array that is plotted
    cmap = cm.ScalarMappable(cmap='jet')
    # create triangulation in k_2, k_{-1}
    triangles = tri.Triangulation(log_params_data[::stride,2], log_params_data[::stride,1]).triangles # 2, 1 was found to be best
    triangle_vertices = np.array([np.array((log_params_data[::stride][T[0]], log_params_data[::stride][T[1]], log_params_data[::stride][T[2]])) for T in triangles]) # plot in k1, k-1, k2
    # create polygon collections that will be plotted (seems impossible to completely erase edges)
    collb = Poly3DCollection(triangle_vertices, facecolors='b', linewidths=0, edgecolors='b')
    coll1 = Poly3DCollection(triangle_vertices, facecolors=cmap.to_rgba(calc_tri_avg(triangles, eigvects[::stride,1])), linewidths=0, edgecolors=np.array((0,0,0,0)))
    coll2 = Poly3DCollection(triangle_vertices, facecolors=cmap.to_rgba(calc_tri_avg(triangles, eigvects[::stride,2])), linewidths=0, edgecolors=np.array((0,0,0,0)))
    # create axes for each subplot
    ax_b = fig.add_subplot(gspec[:,0], projection='3d')
    ax_d1 = fig.add_subplot(gspec[0,1], projection='3d')
    ax_d2 = fig.add_subplot(gspec[1,1], projection='3d')
    # add collections to axes
    ax_b.add_collection(collb) # plain blue dataset
    ax_d1.add_collection(coll1) # colored by dmaps1
    ax_d2.add_collection(coll2) # colored by dmaps2
    # add labels and otherwise adjust plots
    axes = [ax_b, ax_d1, ax_d2]
    for ax in axes:
        # axis limits
        ax.set_xlim((0.99*np.min(log_params_data[::stride,0]), 1.01*np.max(log_params_data[::stride,0])))
        ax.set_ylim((0.99*np.min(log_params_data[::stride,1]), 1.01*np.max(log_params_data[::stride,1])))
        ax.set_zlim((0.99*np.min(log_params_data[::stride,2]), 1.01*np.max(log_params_data[::stride,2])))
        # axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # # hide labels, too squashed
        # plt.tick_params(axis='both', which='major', labelsize=0)
        # # hide grid
        # ax.grid(False)
    plt.tight_layout()
    plt.show()

def dmaps_2d_epsilon():
    """Examines the effect of epsilon on the embedding's eigenvalues"""
    # TODO: change the wonky indexing. In some places you remove the trivial eigenvalue but not the corresponding eigenvector
    # calculate dmap on [a,b] with relatively small epsilon
    # a = 2e-3
    # b = 1e-3
    # npts = 100
    # dmaps_epsilon = 1e-3 # dmaps uses eps^2, theory uses eps
    # theory_epsilon = dmaps_epsilon*dmaps_epsilon
    a = 4
    b = 2
    npts_xaxis = 100
    npts_yaxis = 40
    npts = npts_xaxis*npts_yaxis
    dmaps_epsilon = 0.4
    theory_epsilon = dmaps_epsilon*dmaps_epsilon
    # xdata, ydata = np.meshgrid(a*np.random.uniform(size=npts_xaxis), b*np.random.uniform(size=npts_yaxis)) # np.meshgrid(np.linspace(0,a,npts_xaxis), np.linspace(0,b,npts_yaxis))
    # xdata = xdata.flatten()
    # ydata = ydata.flatten()
    xdata = a*np.random.uniform(size=npts)
    ydata = b*np.random.uniform(size=npts)
    data = np.empty((npts,2))
    data[:,0] = xdata
    data[:,1] = ydata
    k = 12
    # don't always calculate
    dmaps_eigvals, eigvects = dmaps.embed_data(data, k, epsilon=dmaps_epsilon, embedding_method=dmaps._compute_embedding_laplace_beltrami)
    dmaps_eigvals.dump('./data/temp-eigvals2.pkl')
    eigvects.dump('./data/temp-eigvects2.pkl')
    # dmaps_eigvals = np.load('./data/temp-eigvals2.pkl')
    # eigvects = np.load('./data/temp-eigvects2.pkl')
    # calculate theoretical eigenvalues and raise to e^{-eigval*theory_eps}, hopefully match dmaps eps
    theory_eigvals = np.empty(36)
    count = 0
    for i in range(6):
        for j in range(6):
            theory_eigvals[count] = np.power(i*np.pi/a, 2) + np.power(j*np.pi/b, 2)
            count = count + 1
    # sort them in increasing order, discard eigval=0
    theory_eigvals_argsort = np.argsort(theory_eigvals)
    theory_eigvals = theory_eigvals[theory_eigvals_argsort]
    # random scaling factor from operator_0 to operator (pg 28 lafon's thesis)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # only valid for density of 2500 pts per [0,0.2] x [0,0.2] square
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plot exp(-eps*theory_eigvals) vs dmaps_eigvals
    scaling_factor = 4.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1,k), dmaps_eigvals[1:k], color='r', label='DMAPS', lw=4)
    ax.scatter(range(1,k), np.exp(-theory_epsilon*theory_eigvals[1:k]/scaling_factor), c='b', label='Analytical', s=100)
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\lambda$')
    ax.legend()
    plt.savefig('./figs/analytical-dmaps-comparison/eigvals-comparison.png')

    # examine eigenfunctions vs eigenvectors
    theory_eigenfunctions = np.empty((36, npts))
    xs = np.linspace(0,a,npts_xaxis)
    ys = np.linspace(0,b,npts_yaxis)
    count = 0
    for i in range(6):
        for j in range(6):
            theory_eigenfunctions[count] = np.cos(i*data[:,0]*np.pi/a)*np.cos(j*data[:,1]*np.pi/b)
            count = count + 1
    theory_eigenfunctions = theory_eigenfunctions[theory_eigvals_argsort]
    xgrid, ygrid = np.meshgrid(xs, ys)
    for i in range(1,k):
        # attempt to get consistent coloring by swapping sign of eigvector if necessary
        if np.sign(eigvects[0,i]) != np.sign(theory_eigenfunctions[i,0]):
            eigvects[:,i] = -1*eigvects[:,i]
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(data[:,0], data[:,1], c=eigvects[:,i], s=100)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('DMAPS')
        fig1.tight_layout()
        plt.savefig('./figs/analytical-dmaps-comparison/eigvect' + str(i) + '.png')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.scatter(data[:,0], data[:,1], c=theory_eigenfunctions[i], s=100)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Analytical')
        fig2.tight_layout()
        plt.savefig('./figs/analytical-dmaps-comparison/eigfn' + str(i) + '.png')
        

    # show quantitative error between eigfns and eigvects
    relative_error = np.empty(k-1)
    for i in range(1,k):
        scale = np.linalg.norm(theory_eigenfunctions[i])
        relative_error[i-1] = np.linalg.norm(scale*eigvects[:,i] - theory_eigenfunctions[i])/scale
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(1,k), relative_error, s=100)
    ax.set_xlabel(r'$\Phi_i$')
    ax.set_ylabel('relative error in DMAP')
    ax.set_ylim(bottom=0)
    plt.savefig('./figs/analytical-dmaps-comparison/eigvects-error.png')
        
    

class diffusion_sln:

    def __init__(self, imax, xmax, ymax, x0, y0, D, epsilon):
        self._indices = np.arange(1,imax+1)
        self._indices.shape = (imax,1)
        self._ymax = ymax
        self._xmax = xmax
        self._x0 = x0
        self._y0 = y0
        self._A = 4.0*(np.sin(self._indices*np.pi*x0/xmax)*np.sin(self._indices*np.pi*y0/ymax).T)/(ymax*xmax)
        self._E = np.exp(-np.power(D*np.pi*self._indices/xmax, 2))*np.exp(-np.power(epsilon*D*np.pi*self._indices/ymax, 2)).T

    def __call__(self,x,y,t):
        siny = np.sin(self._indices*np.pi*y/self._ymax)
        sinx = np.sin(self._indices*np.pi*x/self._xmax)
        E_t = np.power(self._E, t)
        
        return np.dot(siny.T, np.dot(self._A*E_t, sinx))

def analytical_anisotropic_diffusion():
    """Plots the results of two-dimensionnal anisotropic diffusion on a square"""
    imax = 10
    xmax = 3
    ymax = 2
    x0 = 1
    y0 = 1.5
    D = 1
    epsilon = 0.5
    diffusion_f = diffusion_sln(imax, xmax, ymax, x0, y0, D, epsilon)
    t = 1
    npts = 100
    sln = np.empty((npts,npts))
    xs = np.linspace(0,xmax,npts)
    ys = np.linspace(0,ymax,npts)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            sln[i,j] = diffusion_f(x, y, t)
    xgrid, ygrid = np.meshgrid(xs, ys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xgrid, ygrid, c=sln)
    plt.show()

def analytical_anisotropic_diffusion_eigenfns():
    """Plots two-dimensional eigenfunctions of anisotropic diffusion operator"""
    # create indices sorted by increasing corresponding eigenvalue
    xmax = 1.0
    ymax = 1.0
    epsilon = 1e-1
    imax = 100
    # ivals, jvals = np.meshgrid(range(1,imax+1), range(1,imax+1))
    ivals, jvals = np.meshgrid(range(imax), range(imax))
    eigvals = np.power(ivals/xmax, 2) + epsilon*np.power(jvals/ymax, 2)
    sorted_eigval_indices = np.argsort(eigvals.flatten())
    eigvals = eigvals.flatten()[sorted_eigval_indices]
    ivals = ivals.flatten()[sorted_eigval_indices]
    jvals = jvals.flatten()[sorted_eigval_indices]
    # create grid for sampling fn
    npts = 100
    xgrid, ygrid = np.meshgrid(np.linspace(0,xmax,npts), np.linspace(0,ymax,npts))
    # plot first k eigenfns
    k = 40
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(False)
    for i in range(k):
        # f = np.sin(ivals[i]*np.pi*xgrid/xmax)*np.sin(jvals[i]*np.pi*ygrid/ymax)
        f = np.cos(ivals[i]*np.pi*xgrid/xmax)*np.cos(jvals[i]*np.pi*ygrid/ymax)
        ax.scatter(xgrid, ygrid, c=f)
        ax.set_title(r'$\lambda=$' + str(eigvals[i]))
        plt.savefig('./figs/dmaps/diffusion-eigenfn' + str(i) + '.png')


def dmaps_plane():
    """Tests whether analytical results for anisotropic diffusion on a plane match numerical results"""
    npts = 50
    xdata, ydata = np.meshgrid(np.linspace(0,1,npts), np.linspace(0,1,npts))
    xdata = xdata.flatten()
    ydata = ydata.flatten()
    data = zip(xdata,ydata)
    epsilon = 1e-1
    alpha = 1.0
    kernel = data_kernel(epsilon, alpha)
    k = 14
    eigvals, eigvects = dmaps.embed_data_customkernel(data, k, kernel, symmetric=True)
    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(False)
    for i in range(1,k):
        ax.scatter(xdata, ydata, c=eigvects[:,i], cmap='jet', s=50)
        plt.savefig('./figs/dmaps/dmaps-plane-' + str(i) + '.png')


def dmaps_1d_dataspace():
    """Examines the result of different weightings on the DMAP of the combination of some function f:R2 -> R1, where the kernel is a weighted data-space kernel"""
    # sample (x,y) deterministically on (0,1)^2
    sqrt_npts = 50
    npts = sqrt_npts*sqrt_npts
    x = np.linspace(0,1,sqrt_npts)
    y = np.linspace(0,1,sqrt_npts)
    xgrid, ygrid = np.meshgrid(x,y)
    domain_data = np.array((xgrid.flatten(), ygrid.flatten())).T # shape = (npts,2)
    # simply take f = x + y
    fdata = np.sum(domain_data, axis=1)/2.0
    # combine preimage and image to make complete dataset
    dataset = zip(domain_data, fdata)
    # # plot dataset
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(domain_data[:,0], domain_data[:,1], fdata)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel(r'$f(x,y)$')
    # plt.show(fig)
    # # examine epsilons affect on domain and range separately
    # neps = 6
    # epsilons = np.logspace(-4, 0, neps)
    # dmaps.epsilon_plot(epsilons, domain_data, filename='./figs/domaindata-eps-plot.png')
    # dmaps.epsilon_plot(epsilons, fdata, filename='./figs/fdata-eps-plot.png')
    # eps { 0.01 -> 0 looks reasonalbe
    epsilon = 1e-1
    alpha = 4 # 1e-8
    kernel = data_kernel(epsilon, alpha)
    k = 24
    eigvals, eigvects = dmaps.embed_data_customkernel(dataset, k, kernel)
    fig = plt.figure()
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)
    ax3d.hold(False)
    ax2d.hold(False)
    for i in range(1,k):
        ax3d.scatter(domain_data[:,0], domain_data[:,1], fdata, c=eigvects[:,i])
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel(r'$f(x,y)$')
        ax2d.scatter(fdata, np.ones(npts), c=eigvects[:,i])
        plt.savefig('./figs/dmaps/dmaps-1d-dataspace-alpha-' + str(alpha) + '-eps-' + str(epsilon) + '-' + str(i) + '.png')


def dmaps_2d_dataspace():
    """Examines the result of different weightings on the DMAP of the combination of some function f:R2 -> R2, where the kernel is a weighted data-space kernel"""
    # sample (x,y) deterministically on (0,1)^2
    sqrt_npts = 70
    npts = sqrt_npts*sqrt_npts
    x = np.linspace(0,1,sqrt_npts)
    y = np.linspace(0,1,sqrt_npts)
    xgrid, ygrid = np.meshgrid(x,y)
    domain_data = np.array((xgrid.flatten(), ygrid.flatten())).T # shape = (npts,2)
    # set up transformation matrix
    lam1 = np.sqrt(0.5)
    lam2 = np.sqrt(0.5)
    theta1 = 0
    theta2 = theta1 + np.pi/2 # creates orthogonal eigenvectors
    V = np.array(((np.cos(theta1), np.cos(theta2)), (np.sin(theta1), np.sin(theta2)))) # orthonormal basis for R2
    L = np.diag(np.array((lam1, lam2))) # diagonal eigenvalue matrix
    A = np.dot(V, np.dot(L, np.linalg.inv(V)))
    print A
    # f = lambda x: np.dot(A, x)
    fdata = np.dot(A, domain_data.T).T
    # # # plot dataset
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(domain_data[:,0], domain_data[:,1], c='b', s=40)
    # ax.scatter(fdata[:,0], fdata[:,1], c='r')
    # plt.show()
    # # examine epsilons affect on domain and range separately
    # neps = 6
    # epsilons = np.logspace(-4, 0, neps)
    # dmaps.epsilon_plot(epsilons, domain_data, filename='./figs/domaindata-eps-plot.png')
    # dmaps.epsilon_plot(epsilons, fdata, filename='./figs/fdata-eps-plot.png')

    # data = zip(domain_data, fdata)
    # epsilon = 0.1 # 1e-1
    # alpha = 10 # 1e-8
    # kernel = data_kernel(epsilon, alpha)
    # k = 24
    # eigvals, eigvects = dmaps.embed_data_customkernel(data, k, kernel)

    # for antonios
    alpha = 10
    epsilon = 0.1
    k = 24
    data = np.empty((npts, 4))
    data[:,:2] = domain_data
    data[:,2:] = fdata
    eigvals, eigvects = dmaps.embed_data(data, k, epsilon=epsilon)

    fig = plt.figure()
    axp = fig.add_subplot(121) # preimage
    axi = fig.add_subplot(122) # image
    axp.hold(False)
    axi.hold(False)
    for i in range(1,k):
        axp.scatter(domain_data[:,0], domain_data[:,1], c=eigvects[:,i])
        axp.set_title('preimage')
        axi.scatter(1 + fdata[:,0], fdata[:,1], c=eigvects[:,i])
        axi.set_title('image')
        plt.savefig('./figs/dmaps/dmaps-2d-dataspace-lam2-' + str(lam2) + '-alpha-'+ str(alpha) + '-eps-' + str(epsilon) + '-' + str(i) + '.png')


def dmaps_line():
    """Performs DMAPS on a one-dimensional dataset at different values of epsilon to investigate this parameter's effect on the resulting embedding"""
    # create dataset on line, no randomness to prevent run-to-run variability
    npts = 500
    dataset = np.linspace(0,1,npts) # np.sort(np.random.uniform(size=npts))
    # create set of epsilons to test
    neps = 40
    epsilons = np.logspace(-4, -3, neps) # np.logspace(-4, 0, neps)
    k = 5
    # space to save dmaps results
    embeddings = np.empty((neps, npts))
    eigvals_save = np.empty(neps)
    for i, eps in enumerate(epsilons):
        progress_bar(i, neps)
        eigvals, eigvects = dmaps.embed_data(dataset, k, epsilon=eps)
        embeddings[i] = eigvals[1]*eigvects[:,1]
        eigvals_save[i] = eigvals[1]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # real adhoc test to get colorings to line up properly later on
        # some of the eigenvectors are flipped compared to others, so we unflip them for consistency
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if embeddings[i,0] > embeddings[i,-1]:
            embeddings[i] = embeddings[i,::-1]
    # fig setup, including consistent colornorm for all results to compare across different epsilons
    emax = np.max(np.abs(embeddings))
    colornorm = colors.Normalize(vmin=-emax, vmax=emax) # colors.Normalize(vmin=np.min(embeddings), vmax=np.max(embeddings))
    colormap = cm.ScalarMappable(norm=colornorm, cmap='PuOr')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    for i in range(neps):
        ax.scatter(dataset, epsilons[i]*np.ones(npts), c=colormap.to_rgba(embeddings[i]))
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\epsilon$')
    ax.set_ylim((0.9*np.min(epsilons), 1.1*np.max(epsilons)))
    ax.set_yscale('log')
    plt.savefig('./figs/1d-dmaps' + str(i) + '.png')
    print eigvals_save
        
def sing_pert_contours():
    """Plots contours of system used in 'sing_pert_data_space' in y0 and epsilon"""
    x0 = 1
    lam = 1
    # define times at which to sample data. Always take three points as this leads to three-dimensional data space
    # t0 = 0.01; tf = 3
    # times = np.linspace(t0,tf,3) # for linear eqns
    # x trajectory is constant as x0 and lambda are held constant
    times = np.linspace(0.01, 0.31, 3) # for nonlinear eqns

    # generate true/base trajectory
    y0_true = 2
    eps_true = 1e-3
    yprime = lambda t, y: -y*(1+1/(2*np.sin(y)))/eps_true
    y_integrator = Integration.Integrator(yprime)
    traj_true = y_integrator.integrate(np.array((y0_true,)), times)[:,0]


    npts = 200
    y0s = np.linspace(-1, 3, npts)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # may help visually to set epss = np.logspace(-1, -6, npts), though the resulting plot is not practically useful for the paper
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # epss = np.logspace(3, -3, npts)
    epss = np.logspace(-3.5, -1, npts)
    y0mesh, epsmesh = np.meshgrid(y0s, epss)
    of_evals = np.empty((npts,npts))
    # record data space output at each value of (y0, eps)
    for i in range(npts):
        for j in range(npts):
            yprime = lambda t, y: -y*(1+1/(2*np.sin(y)))/epsmesh[i,j]
            y_integrator = Integration.Integrator(yprime)
            traj = y_integrator.integrate(np.array((y0mesh[i,j],)), times)[:,0]
            of_evals[i,j] = np.power(np.linalg.norm(traj - traj_true), 2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.scatter(epsmesh, y0mesh, c=np.log10(of_evals), s=40)
    ax.set_xlim((np.min(epsmesh), np.max(epsmesh)))
    ax.set_ylim((np.min(y0mesh), np.max(y0mesh)))
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$y_0$')
    plt.show()

def sing_pert_data_space_fig():
    """Plots three dimensional data space of the classic singularly perturbed ODE system: x' = -lambda*x, y' = -y/epsilon. Probably will end up plotting {y(t1), y(t2), y(t3)} though may also plot norms {|| (x(t1), y(t1)) ||, ... } though this doesn't represent data space in the traditional sense. In particular we're interested in observing the transition from 2 to 1 to 0 dimensional parameter -> data space mappings, i.e. 2 stiff to 1 stiff to 0 stiff parameters."""
    times = np.array((0.01, 0.1, 0.5))
    npts = 40
    count = 0
    lam = 2.0
    epss = np.logspace(-1, 0, npts)
    x0s = np.ones((npts,2))
    x0s[:,1] = np.linspace(0,10,npts)
    xt1 = np.empty((npts,npts))
    xt2 = np.empty((npts,npts))
    xt3 = np.empty((npts,npts))
    epss_copy = np.empty((npts,npts))
    y0s = np.empty((npts,npts))
    for i, x0 in enumerate(x0s):
        for j, eps in enumerate(epss):
            f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
            integrator = Integration.Integrator(f)
            xt1[i,j], xt2[i,j], xt3[i,j] = integrator.integrate(x0, times)[:,1]
            epss_copy[i,j] = eps
            y0s[i,j] = x0[1]
            
    # # save the data
    # xt1.dump('./data/sing-pert-x1.pkl')
    # xt2.dump('./data/sing-pert-x2.pkl')
    # xt3.dump('./data/sing-pert-x3.pkl')
    # epss_copy.dump('./data/sing-pert-eps-copy.pkl')
    # y0s.dump('./data/sing-pert-y0s.pkl')

    # xt1 = np.load('./data/sing-pert-x1.pkl')
    # xt2 = np.load('./data/sing-pert-x2.pkl')
    # xt3 = np.load('./data/sing-pert-x3.pkl')
    # epss_copy = np.load('./data/sing-pert-eps-copy.pkl')
    # y0s = np.load('./data/sing-pert-y0s.pkl')
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # print 'loading previously generated data for plots'
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    # # plot points in 2d parameter space
    x0_grid, epss_grid = np.meshgrid(x0s, epss)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epss_grid, x0_grid, s=5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$y(t_0)$')
    ax.set_xlim((0.09, 1.1))
    ax.set_ylim((-1, 11))

    # # plot eps and y0 coloring of manifold
    gspec = gs.GridSpec(5,5)
    cmap = colormaps.inferno


    fig1 = plt.figure()
    fig2 = plt.figure()
    # axes
    ax_eps = fig1.add_subplot(gspec[:,:4], projection='3d')
    ax_y0 = fig2.add_subplot(gspec[:,:4], projection='3d')
    # make color data
    eps_colors = (1/epss_copy)/np.max(1/epss_copy)
    x0_colors = y0s/np.max(y0s)
    light = LightSource(0, 65)
    illuminated_surface_eps = light.shade(eps_colors, cmap=cm.gist_earth, blend_mode='soft')
    illuminated_surface_y0 = light.shade(x0_colors, cmap=plt.get_cmap('winter'), blend_mode='soft')
    # add scatter plots

    
    vv.use('gtk')
    vv.figure()
    a1 = vv.subplot(111)
    m1 = vv.surf(xt1, xt2, xt3, eps_colors)
    m1.colormap = vv.CM_HOT
    app = vv.use()
    app.Run()


    # ax_eps.plot_surface(xt1, xt2, xt3, facecolors=illuminated_surface_eps, linewidth=0, antialiased=False, cstride=1, rstride=1)
    # ax_y0.plot_surface(xt1, xt2, xt3, facecolors=illuminated_surface_y0, edgecolors=np.array((0,0,0,0)), linewidth=0, antialiased=False, cstride=1, rstride=1)#, alpha=0.5)
    # # ax_eps.plot_surface(xt1, xt2, xt3, facecolors=cmap(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.5)
    # # ax_y0.plot_surface(xt1, xt2, xt3, facecolors=cmap(x0_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.5)
    # # add unique titles
    # # ax_eps.set_title('Colored by ' + r'$\epsilon$')
    # # ax_y0.set_title('Colored by ' + r'$y(t_0)$')
    # # add colorbars
    # ax_eps_cb = fig1.add_subplot(gspec[:,4])
    # ax_y0_cb = fig2.add_subplot(gspec[:,4])
    # eps_cb = colorbar.ColorbarBase(ax_eps_cb, cmap=cmap, norm=colors.Normalize(np.min(1/epss_copy), np.max(1/epss_copy)), ticks=np.logspace(np.log10(np.min(1/epss_copy)), np.log10(np.max(1/epss_copy)), 2))
    # ax_eps_cb.text(0.4, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)
    # y0_cb = colorbar.ColorbarBase(ax_y0_cb, cmap=cmap, norm=colors.Normalize(np.min(y0s), np.max(y0s)), ticks=np.linspace(np.min(x0s), np.max(y0s), 5))
    # ax_y0_cb.text(0.4, 1.05, r'$y(t_0)$', transform=ax_y0_cb.transAxes, fontsize=48)
            
    # # # plot a ball in model space
    # # find points in ball
    # eps_center = 0.5 # center of ball is (eps_center, y0_center) in parameter space
    # y0_center = 5.0
    # model_space_radius_squared = 0.1
    # f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps_center)*(1+10/(1.5-np.sin(x[1])))))), x)
    # integrator = Integration.Integrator(f)
    # xt1_center, xt2_center, xt3_center = integrator.integrate(np.array((1,y0_center)), times)[:,1]
    # surface_colors = np.zeros((npts,npts,4))
    # # also keep track of which parameters the model manifold points correspond to
    # npts_in_ball = 0
    # eps_in_ball = np.empty(npts*npts)
    # y0_in_ball = np.empty(npts*npts)
    # for i, x0 in enumerate(x0s):
    #     for j, eps in enumerate(epss):
    #             if np.power(xt1[i,j] - xt1_center, 2) + np.power(xt2[i,j] - xt2_center, 2) + np.power(xt3[i,j] - xt3_center, 2) < model_space_radius_squared:
    #                 surface_colors[i,j,3] = 0.5 # non-zero alpha
    #                 eps_in_ball[npts_in_ball] = eps
    #                 y0_in_ball[npts_in_ball] = x0[1]
    #                 npts_in_ball = npts_in_ball + 1
    # eps_in_ball = eps_in_ball[:npts_in_ball]
    # y0_in_ball = y0_in_ball[:npts_in_ball]
    # print 'found', npts_in_ball, 'points intersecting ball'

    # # plot intersection of full model manifold and ball, colored by x0
    # gspec = gs.GridSpec(5,5)
    # fig = plt.figure()
    # ax_ball = fig.add_subplot(gspec[:,:4], projection='3d')
    # ax_ball.hold(True)
    # # plot surface
    # ax_ball.plot_surface(xt1, xt2, xt3, facecolors=cmap(x0_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.5)
    # # plot intersection
    # ax_ball.plot_surface(xt1, xt2, xt3, facecolors=surface_colors, linewidth=0, cstride=1, rstride=1, antialiased=False)
    # # plot ball
    # u = np.linspace(0, 2*np.pi, 500)
    # v = np.linspace(0, np.pi, 500)
    # ball_x = np.sqrt(model_space_radius_squared)*np.outer(np.cos(u), np.sin(v)) + xt1_center
    # ball_y = np.sqrt(model_space_radius_squared)*np.outer(np.sin(u), np.sin(v)) + xt2_center
    # ball_z = np.sqrt(model_space_radius_squared)*np.outer(np.ones(np.size(u)), np.cos(v)) + xt3_center
    # ax_ball.plot_surface(ball_x, ball_y, ball_z, color='b', alpha=0.4, linewidth=0)
    # # colorbar
    # ax_cb = fig.add_subplot(gspec[:,4])
    # y0_cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=colors.Normalize(np.min(y0s), np.max(y0s)), ticks=np.linspace(np.min(x0s), np.max(y0s), 5))
    # ax_cb.text(0.4, 1.05, r'$y(t_0)$', transform=ax_cb.transAxes, fontsize=48)


    # # # adjust all figure's properties at once as they're all the same
    # for ax in [ax_eps, ax_y0, ax_ball]:
    #     # axis limits
    #     ax.set_xlim((0.99*np.min(xt1), 1.01*np.max(xt1)))
    #     ax.set_ylim((0.99*np.min(xt2), 1.01*np.max(xt2)))
    #     ax.set_zlim((0.99*np.min(xt3), 1.01*np.max(xt3)))
    #     # axis labels
    #     ax.set_xlabel(r'$y(t_1)$')
    #     ax.set_ylabel(r'$y(t_2)$')
    #     ax.set_zlabel(r'$y(t_3)$')
    #     # white out the bg
    #     ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #     ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #     ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #     # # hide grid
    #     # ax.grid(False)
    #     # # hide tick labels
    #     # plt.tick_params(axis='both', which='major', labelsize=0)
    #     # move labels to avoid overlap with numbers
    #     ax.xaxis._axinfo['label']['space_factor'] = 2.2
    #     ax.yaxis._axinfo['label']['space_factor'] = 2.2
    #     ax.zaxis._axinfo['label']['space_factor'] = 2.2

    # ax_ball.set_zlim((-0.5, 0.5)) # for better visualization of ball

    # # map model manifold ball intersection back to parameter space
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(eps_in_ball, y0_in_ball, lw=0)
    # ax.set_xlabel(r'$\epsilon$')
    # ax.set_ylabel(r'$y(t_0)$')
    # ax.set_title('Preimage of model manifold ball')
    # plt.tight_layout()
    # plt.show()

def main():
    # dmaps_2d_epsilon()
    # sing_pert_contours()
    # dmaps_line()
    # dmaps_2d_dataspace()
    # dmaps_1d_dataspace()
    # analytical_anisotropic_diffusion()
    # analytical_anisotropic_diffusion_eigenfns()
    # dmaps_plane()
    # sing_pert_data_space_fig()
    # rawlings_2d_dmaps_fig()
    # rawlings_3d_dmaps_fig()
    # two_effective_one_neutral_dmaps_fig()
    # discretized_laplacian_dmaps()
    transformed_param_space_fig()
    # test()
    
if __name__=='__main__':
    main()
