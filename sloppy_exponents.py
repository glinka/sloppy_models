import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.gridspec as gs
import sympy
import dmaps
import PseudoArclengthContinuation as PSA
import ObjectiveFunction as of

class ab_fn:
    """A test sloppy function, suitable for use with PSA module to find contours. Models an objective function used to estimate parameters from data drawn from a three-dimensional system as defined in 'get_sloppy_traj'"""
    def __init__(self, data, times, contour, epsilon):
        """
        Args:
        data (array): arrray of shape (ndimensions, npts) to be fit
        times (array): vector of shape (npts,) in which times[i] is the time at which data[:,i] was taken
        contour (float): contour value of interest
        epsilon (float): characteristic neighborhood used in dmaps kernel
        """
        self._data = data
        self._times = times
        self._contour = contour
        self._epsilon = epsilon

    def f(self, alpha, beta):
        return np.array((np.sum(np.linalg.norm(self._data - get_sloppy_traj(beta, alpha, self._times), axis=0)**2) - self._contour,))

    def Df(self, alpha, beta):
        expalpha = np.exp(-alpha*self._times)
        a11 = 2*np.sum(self._times*expalpha*(2*(self._data[0,:] - expalpha) + beta*(self._data[1,:] - beta*expalpha)))
        a12 = 2*np.sum(expalpha*(beta*expalpha - self._data[1,:]))
        return np.array(((a11, a12),))
        
    def gradient_dmaps_kernel(self, pt1, pt2):
        return np.exp(-np.linalg.norm(pt1 - pt2)**2/self._epsilon - np.power(np.dot(self.Df(*pt1), pt1 - pt2)/self._epsilon, 2))
    

def gradient_dmaps():
    """Testing the effects of including gradient information in the DMAPS kernel on the resulting embedding."""
    # set values for data generation and algorithm performance
    k1_true = 0.1
    kinv_true = 0.1
    k2_true = 10000.0
    alpha_true = k1_true*k1_true/(kinv_true*kinv_true + k2_true)
    beta_true = k2_true/(kinv_true*kinv_true + k2_true)
    alpha_true = np.array((alpha_true,))
    beta_true = np.array((beta_true,))
    # the contour value for which data will be generated
    contour = 1e-1
    # psa stepsize
    ds = 1e-3
    times = np.linspace(1, 5, 10)
    data = get_sloppy_traj(beta_true, alpha_true, times)
    of = ab_fn(data, times, contour, ds)
    # set up psa solver
    psa_solver = PSA.PSA(of.f, of.Df)
    # perturb beta to ensure nonsingular jacobian in psa routine
    beta_perturbed = 1.001*beta_true
    ab_contour = psa_solver.find_branch(alpha_true, beta_perturbed, ds, nsteps=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ab_contour[:,0], ab_contour[:,1])
    plt.savefig('./figs/embeddings/dmaps_data.png')
    k = 6
    eigvals, dmaps_embedding = dmaps.embed_data_customkernel(ab_contour, k, of.gradient_dmaps_kernel)
    for i in range(1, k):
        for j in range(i+1, k):
            ax.cla()
            ax.scatter(eigvals[i]*dmaps_embedding[:,i], eigvals[j]*dmaps_embedding[:,j])
            plt.savefig('./figs/embeddings/dmaps' + str(i) + str(j) + '.png')

def test_psa():
    k1_true = 0.1
    kinv_true = 0.1
    k2_true = 10000.0
    alpha_true = k1_true*k1_true/(kinv_true*kinv_true + k2_true)
    beta_true = k2_true/(kinv_true*kinv_true + k2_true)
    alpha_true = np.array((alpha_true,))
    beta_true = np.array((beta_true,))
    times = np.linspace(1, 5, 10)
    data = get_sloppy_traj(beta_true, alpha_true, times)
    ncontours = 5
    contours = np.logspace(-2, 0, ncontours)
    fig = plt.figure()
    ax = fig.add_subplot(gspec[:,:5])
    ax_cb = fig.add_subplot(gspec[:,5])
    colornorm = colors.Normalize(vmin=np.log10(contours[0]), vmax=np.log10(contours[-1]))
    colormap = cm.ScalarMappable(norm=colornorm, cmap='jet')
    ds = 1e-5
    for i in range(ncontours):
        myfn = ab_fn(data, times, contours[i])
        psa_solver = PSA.PSA(myfn.f, myfn.Df)
        # perturb beta to ensure nonsingular jacobian in psa routine
        beta_perturbed = 1.001*beta_true
        ab_contour = psa_solver.find_branch(alpha_true, beta_perturbed, ds, nsteps=50000)
        ax.scatter(ab_contour[:,0], ab_contour[:,1])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    cb = colorbar.ColorbarBase(ax_cb, cmap='jet', norm=colornorm, orientation='vertical')
    cb.set_label('log(obj. fn. value)')
    plt.show(fig)


def run_dmaps():
    # k1_true = 0.1
    # kinv_true = 0.1
    # k2_true = 1000.0
    # k1_true = 10.0
    # kinv_true = 10.0
    # k2_true = 100.0
    k1_true = 1.0
    kinv_true = 1.0
    k2_true = 100.0
    beta_true = k2_true/(kinv_true*kinv_true + k2_true)
    alpha_true = k1_true*k1_true/(kinv_true*kinv_true + k2_true)
    # define objective function using sympy
    # create sympy matrix from numpy array
    times = np.linspace(1, 5, 10)
    ntimes = times.shape[0]
    data = sympy.Matrix(get_sloppy_traj(beta_true, alpha_true, times))
    x,y,z = sympy.symbols('x,y,z', real=True)
    # convert times array to sympy type, then take element-wise exponential
    times = sympy.Matrix(times).transpose()
    k1,k2,kinv = sympy.symbols('k1,k2,kinv')
    ks = [k1,k2,kinv]
    # redefine beta to include (0,40), jk
    beta = k2/(kinv*kinv + k2)
    alpha = k1*k1/(kinv*kinv + k2)
    exp_alpha = -alpha*times
    exp_alpha = exp_alpha.applyfunc(sympy.exp)
    ys = sympy.zeros(3, ntimes)
    ys[0,:] = exp_alpha
    ys[1,:] = beta*exp_alpha
    ys[2,:] = sympy.ones(1, ntimes) - exp_alpha
    # make the sympy obj. fn., essentially the squared frobenius norm of the matrix created by stacking vectors
    # at different sampling times together
    f = sum((ys - data).applyfunc(lambda x: x*x))
    sloppy_of = of.ObjectiveFunction(f, ks)
    # print of.gradient([k1_true, k2_true, kinv_true]), of.hessian([k1_true, k2_true, kinv_true])
    # ought to be a smarter way by converting from sympy to numpy instead of redefining
    # data = np.array(data)
    # times = np.array(times)

    # contour = 6e-2 w/ ks = [10,10,100]
    contour = 1e-2
    beta_true = np.array((beta_true,))
    alpha_true = np.array((alpha_true,))
    times = np.linspace(1, 5, 10)
    data = get_sloppy_traj(beta_true, alpha_true, times)
    ds = 5e-4
    altof = ab_fn(data, times, contour, ds)
    # set up psa solver
    psa_solver = PSA.PSA(altof.f, altof.Df)
    # perturb beta to ensure nonsingular jacobian in psa routine
    beta_perturbed = 1.001*beta_true
    ab_contour = psa_solver.find_branch(alpha_true, beta_perturbed, ds, nsteps=400)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ab_contour[:,1], ab_contour[:,0])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    plt.show(fig)

    npts = ab_contour.shape[0]
    alphas = ab_contour[:,1]
    betas = ab_contour[:,0]
    nks = 10
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    npts_toplot = 100
    spacing = npts/npts_toplot
    # for each sampled alpha/beta pair, of which there will be 'npts_toplot + 1', we find 'nks' values of k1, k2, kinv
    dmaps_data = np.empty(((npts_toplot+1)*nks, 3))
    for i in range(npts):
        if i % spacing == 0:
            k1s = np.abs(np.logspace(0.1, 100, nks) + np.random.normal(size=nks))
            k2s = betas[i]*k1s*k1s/(alphas[i])
            kinvs = np.sqrt((1 - betas[i])/(alphas[i]*betas[i]))*k1s
            dmaps_data[nks*i/spacing:nks*(i/spacing+1),:] = np.array((k1s, k2s, kinvs)).T
            # try log to scale
            # dmaps_data[nks*i/spacing:nks*(i/spacing+1),:] = np.log(np.array((k1s, k2s, kinvs)).T)
            ax3d.scatter(k1s, k2s, kinvs, c='b')
    # # see what data set looks like
    # ax3d.set_xscale('log')
    # ax3d.set_yscale('log')
    # ax3d.set_zscale('log')
    ax3d.set_xlim(left=0)
    ax3d.set_ylim(bottom=0)
    ax3d.set_zlim(bottom=0)
    ax3d.set_xlabel(r'$k_1$')
    ax3d.set_ylabel(r'$k_2$')
    ax3d.set_zlabel(r'$k_{-1}$')
    # ax3d.set_xlabel(r'$log(k_1)$')
    # ax3d.set_ylabel(r'$log(k_2)$')
    # ax3d.set_zlabel(r'$log(k_{-1})$')
    plt.show(fig3d)
    # plt.savefig('./figs/ks_3d.png')
    # do dmaps
    k = 6
    # add tiny amount to ensure all-positive eigenvalues
    H = sloppy_of.hessian([k1_true, k2_true, kinv_true]) + 1e-16*np.identity(3)
    H_inv = np.linalg.inv(H)
    print 'eigvals of hessian at min:', np.linalg.eigvals(H)
    # use metric defined by sqrt(xAx) where A is pos. def.
    metric = lambda x,y: np.sqrt(np.dot(x-y, np.dot(H, x-y)))
    eigvals, eigvects = dmaps.embed_data(dmaps_data, k, metric=metric)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(1, k):
        for j in range(i+1, k):
            ax.cla()
            ax.scatter(eigvects[:,i], eigvects[:,j], c=dmaps_data[:,2])
            # plt.show()
            plt.savefig('./figs/embeddings/dmaps/qssa_' + str(i) + '_' + str(j) + '.png')

def get_sloppy_traj(beta, alpha, times):
    exp_alpha = np.exp(-alpha*times)
    cas = exp_alpha
    cbs = beta*exp_alpha
    ccs = (1 - exp_alpha)
    return np.array((cas, cbs, ccs))

# def plot_contours():
    # k1 = 10.0
    # k1inv = 10.0
    # k2 = 100.0
    # for i in range(npts):
    #     if i % spacing == 0:
    #         beta = contour_pts[i,0]
    #         alpha = contour_pts[i,1]
    #         ax3d.plot(k1s, beta/alpha*k1s*k1s, np.sqrt((1 - beta)/(alpha*beta))*k1s, c='b')
    # ax3d.set_xlabel(r'$k_1$')
    # ax3d.set_ylabel(r'$k_2$')
    # ax3d.set_zlabel(r'$k_{-1}$')

# def of_contours():
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # axtwin = ax.twinx()
    # cs = ['g', 'b', 'r']
    # labels = [r'$y_1$', r'$y_2$', r'$y_3$']
    # y1 = ax.plot(times, data[0,:], c=cs[0], label=labels[0])
    # y2 = ax.plot(times, data[1,:], c=cs[1], label=labels[1])
    # y3 = axtwin.plot(times, data[2,:], c=cs[2], label=labels[2])
    # ax.scatter(times, data[0,:], c=cs[0], s=50)
    # ax.scatter(times, data[1,:], c=cs[1], s=50)
    # axtwin.scatter(times, data[2,:], c=cs[2], s=50)
    # ys = y1 + y2 + y3
    # labs = [y.get_label() for y in ys]
    # ax.legend(ys, labs, loc=0)
    # ax.set_xlim((times[0],times[-1]))
    # ax.set_xlabel('time')
    # ax.set_ylabel(r'$y_1$, $y_2$')
    # axtwin.set_ylabel(r'$y_3$')
    # plt.tight_layout()
    # plt.show()

if __name__=='__main__':
    # gradient_dmaps()
    # import time
    # t0 = time.time()
    # test_psa()
    # t1 = time.time()
    # print 'PSA took', t1 - t0
    # t0 = time.time()
    # plot_contours()
    # t1 = time.time()
    # print 'shit method took', t1 - t0
    run_dmaps()
    # of_contours()
