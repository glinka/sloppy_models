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

# """ Basic pseudo-arclength continuation, requires analytic expression for Jacobian """
# class PSA:

#     def __init__(self, f, Df):
#         """ **Implementation of naive and basic pseudo-arclength continuation**

#         **Args**:
#         f: function :math:'f(x, \lambda)' from :math:'R^{n+1} \rightarrow R^n` on which continuation will be performed, and in which :math:'\lambda' is the continuation parameter
#         Df: function :math:'Df(x, \lambda)' from :math:'R^{n+1} \rightarrow R^{n(n+1)}' which is the Jacobian of f
        
        
        
        
# """

class ab_fn:
    def __init__(self, data, times, contour):
        self._data = data
        self._times = times
        self._contour = contour

    def f(self, alpha, beta):
        return np.array((np.sum(np.linalg.norm(self._data - get_sloppy_traj(beta, alpha, self._times), axis=0)**2) - self._contour,))

    def Df(self, alpha, beta):
        expalpha = np.exp(-alpha*self._times)
        a11 = 2*np.sum(self._times*expalpha*(2*(self._data[0,:] - expalpha) + beta*(self._data[1,:] - beta*expalpha)))
        a12 = 2*np.sum(expalpha*(beta*expalpha - self._data[1,:]))
        return np.array(((a11, a12),))
        
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
    ax = fig.add_subplot(111)
    ds = 1e-4
    for i in range(ncontours):
        myfn = ab_fn(data, times, contours[i])
        psa_solver = PSA.PSA(myfn.f, myfn.Df)
        # perturb beta to ensure nonsingular jacobian in psa routine
        beta_perturbed = 1.001*beta_true
        ab_contour = psa_solver.find_branch(alpha_true, beta_perturbed, ds, nsteps=50000)
        ax.scatter(ab_contour[:,0], ab_contour[:,1])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    plot_contours(ax)
    plt.show(fig)


def run_dmaps():
    k1_true = 100.0
    kinv_true = 0.1
    k2_true = 10000.0
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
    # redefine beta to include (0,40)
    beta = 40*k2/(kinv*kinv + k2)
    alpha = k1*k1/(kinv*kinv + k2)
    exp_alpha = -alpha*times
    exp_alpha = exp_alpha.applyfunc(exp)
    ys = zeros(3, ntimes)
    ys[0,:] = exp_alpha
    ys[1,:] = beta*exp_alpha
    ys[2,:] = ones(1, ntimes) - exp_alpha
    # make the sympy obj. fn., essentially the squared frobenius norm of the matrix created by stacking vectors
    # at different sampling times together
    f = sum((ys - data).applyfunc(lambda x: x*x))
    sloppy_of = of.ObjectiveFunction(f, ks)
    # print of.gradient([k1_true, k2_true, kinv_true]), of.hessian([k1_true, k2_true, kinv_true])
    # ought to be a smarter way by converting from sympy to numpy instead of redefining
    # data = np.array(data)
    # times = np.array(times)
    times = np.linspace(1, 5, 10)
    data = get_sloppy_traj(beta_true, alpha_true, times)
    ds = 5e-3
    contour = 0.229 # np.power(10.0, 1.0)
    ab_contour = sloppy_continuation(data, times, contour, ds, beta_true, alpha_true, nsteps=1200)

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
            k1s = np.linspace(0.1, 10, nks) + np.random.normal(size=nks)
            k2s = betas[i]*k1s*k1s/(alphas[i]*40)
            kinvs = np.sqrt((1 - betas[i]/40.0)/(alphas[i]*betas[i]))*k1s
            dmaps_data[nks*i/spacing:nks*(i/spacing+1),:] = np.array((k1s, k2s, kinvs)).T
            # dmaps_data[nks*i/spacing:nks*(i/spacing+1),1] = k2s
            # dmaps_data[nks*i/spacing:nks*(i/spacing+1),2] = kinvs
            ax3d.scatter(k1s, k2s, kinvs, c='b')
    # see what data set looks like
    ax3d.set_xlabel(r'$k_1$')
    ax3d.set_ylabel(r'$k_2$')
    ax3d.set_zlabel(r'$k_{-1}$')
    # do dmaps
    eigvals, eigvects = sloppy_dmaps(dmaps_data, np.copy(sloppy_of.hessian([k1_true, k2_true, kinv_true])))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(eigvects[:,1], eigvects[:,2])
    plt.show()
    print eigvals[:5]
    

def sloppy_dmaps(data, hessian):
    npts = data.shape[0]
    norms = np.zeros((npts,npts))
    average_norm = 0.0
    for i in range(npts):
        for j in range(i+1, npts):
            # note, now using hessian only at minimum where it will be positive semi definite and thus define a proper norm
            norms[i,j] = np.sqrt(np.dot(np.dot(data[i] - data[j], hessian), data[i] - data[j])) # hessian(data[i])
            average_norm = average_norm + norms[i,j]
    # divide by number of pairwise distances: nptsChoose2
    average_norm = average_norm/(npts*(npts-1)/2)
    epsilon = np.median(norms[norms > 0]) # average_norm
    print 'median:', np.median(norms[norms > 0]), 'avg:', average_norm
    print 'eigvals of hessian at min:', np.linalg.eigvals(hessian)
    W = np.empty((npts,npts))
    for i in range(npts):
        W[i,i] = 1
        for j in range(i+1, npts):
            W[i,j] = np.exp(-np.power(norms[i,j], 2.0)/epsilon)
            W[j,i] = W[i,j]
    D_half_inv = np.identity(npts)/np.sqrt(np.sum(W,1))
    S = np.dot(np.dot(D_half_inv, W), D_half_inv)
    eigvals, eigvects = np.linalg.eigh(S)
    eigvects = np.dot(D_half_inv, eigvects)
    sorted_indices = np.argsort(np.abs(eigvals))
    # reverse to sort eigvals in decreasing magnitude
    sorted_indices = sorted_indices[::-1]
    eigvals = eigvals[sorted_indices]
    eigvects = eigvects[:,sorted_indices]
    return [eigvals, eigvects]


def get_exact_traj(k1, k1inv, k2, times, ca0):
    alpha = 0.5*(k1 + k1inv + k2 + np.sqrt(np.power(k1 + k1inv + k2, 2) - 4*k1*k2))
    beta = 0.5*(k1 + k1inv + k2 - np.sqrt(np.power(k1 + k1inv + k2, 2) - 4*k1*k2))
    exp_alpha = np.exp(-alpha*times)
    exp_beta = np.exp(-beta*times)
    cas = ca0*(exp_alpha*k1*(alpha - k2)/(alpha*(alpha - beta)) + exp_beta*k1*(k2 - beta)/(beta*(alpha - beta)))
    cbs = ca0*(exp_alpha*k1/(beta - alpha) + exp_beta*k1/(alpha - beta))
    ccs = ca0*(k1*k2/(alpha*beta) + exp_alpha*k1*k2/(alpha*(alpha - beta)) - exp_beta*k1*k2/(beta*(alpha - beta)))
    return np.array((cas, cbs, ccs))

def get_qssa_traj(k1, k1inv, k2, times, ca0):
    kexp = k1*k1/(k1inv*k1inv + k2)
    kscale = k2/(k1inv*k1inv + k2)
    exp_kexp = np.exp(-kexp*times)
    cas = ca0*exp_kexp
    cbs = ca0*kscale*exp_kexp
    ccs = ca0*(1 - exp_kexp)
    # returns 3 x ntimes array
    return np.array((cas, cbs, ccs))
    # kexp = k1*k2/(k1inv + k2)
    # kscale = k1/(k1inv + k2)
    # exp_kexp = np.exp(-kexp*times)
    # cas = ca0*exp_kexp
    # cbs = ca0*kscale*exp_kexp
    # ccs = ca0*(1 - exp_kexp)
    # return [cas, cbs, ccs]

def get_sloppy_traj(beta, alpha, times):
    exp_alpha = np.exp(-alpha*times)
    cas = exp_alpha
    cbs = beta*exp_alpha
    ccs = (1 - exp_alpha)
    return np.array((cas, cbs, ccs))

def leastsq_of(data, k1, k1inv, k2, times, ca0):
    return np.sum(np.linalg.norm(data - get_qssa_traj(k1, k1inv, k2, times, ca0), axis=0))
    # return np.sum(np.linalg.norm(data - get_exact_traj(k1, k1inv, k2, times, ca0), axis=0))

# evaluates f = (lsq_err, arclength error)
def f_lsq(data, times, contour, beta, alpha, beta_previous, alpha_previous, betaprime, alphaprime, ds):
    lsq_error = np.sum(np.linalg.norm(data - get_sloppy_traj(beta, alpha, times), axis=0)**2) - contour
    arclength_error = (beta - beta_previous)*betaprime + (alpha - alpha_previous)*alphaprime - ds
    return np.array((lsq_error, arclength_error))

def f_lsq_jacobian(data, times, beta, alpha, betaprime, alphaprime):
    expalpha = np.exp(-alpha*times)
    a11 = 2*np.sum(expalpha*(beta*expalpha - data[1,:]))
    a12 = 2*np.sum(times*expalpha*(2*(data[0,:] - expalpha) + beta*(data[1,:] - beta*expalpha)))
    a21 = betaprime
    a22 = alphaprime
    return np.array(((a11, a12), (a21, a22)))

# find alpha, beta value constrained by arclength
def sloppy_newton(data, times, contour, beta0, alpha0, beta_previous, alpha_previous, betaprime, alphaprime, ds):
    x = np.array((beta0, alpha0))
    maxiters = 10000
    iters = 0
    tol = 1e-8
    error = np.linalg.norm(f_lsq(data, times, contour, x[0], x[1], beta_previous, alpha_previous, betaprime, alphaprime, ds))
    while error > tol and iters < maxiters:
        feval = f_lsq(data, times, contour, x[0], x[1], beta_previous, alpha_previous, betaprime, alphaprime, ds)
        dx = -np.dot(np.linalg.inv(f_lsq_jacobian(data, times, x[0], x[1], betaprime, alphaprime)), feval)
        error = np.linalg.norm(feval)
        x = x + dx
        iters = iters + 1
    if iters == maxiters:
        print 'failed to converge to tolerance:', tol, 'within', maxiters, 'iters'
        return False
    return x
                                     
def sloppy_newton_init(data, times, contour, beta0, alpha0):
    maxiters = 100000
    iters = 0
    tol = 1e-8
    beta = beta0
    alpha = alpha0
    error = np.sum(np.linalg.norm(data - get_sloppy_traj(beta, alpha, times), axis=0)**2) - contour
    expalpha = np.exp(-alpha*times)
    while np.abs(error) > tol and iters < maxiters:
        dbeta = -error/(2*np.sum(expalpha*(beta*expalpha - data[1,:])))
        beta = beta + dbeta
        error = np.sum(np.linalg.norm(data - get_sloppy_traj(beta, alpha, times), axis=0)**2) - contour
        iters = iters + 1
    if iters == maxiters:
        print '****************************************'
        print '** initial point on contour not found **'
        print '****************************************'
        print 'failed to converge to tolerance:', tol, 'within', maxiters, 'iters'
        return False
    return beta
    

def sloppy_continuation(data, times, contour, ds, beta_true, alpha_true, nsteps=4000):
    # find first two points on contour to establish betaprime, alphaprime
    # insert initialization code here
    alpha = alpha_true
    # got to add some noise
    beta = beta_true + 1e-3*beta_true
    beta = sloppy_newton_init(data, times, contour, beta, alpha)
    # arbitrary direction/scaling at first
    xprime = np.dot(np.linalg.inv(f_lsq_jacobian(data, times, beta, alpha, 0, 1)), np.array((0,1)))
    xprime = xprime/np.linalg.norm(xprime)
    beta_init = beta
    alpha_init = alpha
    betaprime = xprime[0]
    alphaprime = xprime[1]
    # continue along contour both forwards and backwards nsteps
    # forward
    contour_points = np.empty(((2*nsteps), 2))
    for k in range(2):
        ds = -ds
        beta = beta_init
        alpha = alpha_init
        for i in range(nsteps):
            beta0 = beta + betaprime*ds
            alpha0 = alpha + alphaprime*ds
            beta_previous = beta
            alpha_previous = alpha
            xnew = sloppy_newton(data, times, contour, beta0, alpha0, beta_previous, alpha_previous, betaprime, alphaprime, ds)
            beta = xnew[0]
            alpha = xnew[1]
            xprimenew = np.dot(np.linalg.inv(f_lsq_jacobian(data, times, beta, alpha, betaprime, alphaprime)), np.array((0,1)))
            xprime = xprime/np.linalg.norm(xprime)
            betaprime = xprimenew[0]
            alphaprime = xprimenew[1]
            contour_points[k*nsteps+i,:] = np.copy(xnew)
    return contour_points

def plot_contours(ax):
    # def works:
    k1 = 0.1
    k1inv = 0.1
    k2 = 10000.0
    # k1 = 10.0
    # k1inv = 10.0
    # k2 = 100.0
    beta = k2/(k1inv*k1inv + k2)
    alpha = k1*k1/(k1inv*k1inv + k2)
    times = np.linspace(1, 5, 10)
    data = get_sloppy_traj(beta, alpha, times)
    ds = 1e-4
    ncontours = 5
    contours = np.logspace(-2, 0, ncontours)
    gspec = gs.GridSpec(6,6)
    fig = plt.figure()
    # ax = fig.add_subplot(gspec[:,:5])
    ax_cb = fig.add_subplot(gspec[:,5])
    colornorm = colors.Normalize(vmin=np.log10(contours[0]), vmax=np.log10(contours[-1]))
    colormap = cm.ScalarMappable(norm=colornorm, cmap='jet')

    # perturb beta
    beta = beta*1.001
    for i in range(ncontours):
        contour_val = contours[i]
        contour_pts = sloppy_continuation(data, times, contour_val, ds, beta, alpha)
        ax.scatter(contour_pts[:,1], contour_pts[:,0], c=colormap.to_rgba(np.log10(contours[i])), s=20, zorder=2)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    cb = colorbar.ColorbarBase(ax_cb, cmap='jet', norm=colornorm, orientation='vertical')
    cb.set_label('log(obj. fn. value)')

    # 3d plot in parameter space
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    k1s = np.logspace(1, 3.5, 40)
    npts = contour_pts.shape[0]
    npts_to_plot = 100
    spacing = npts/npts_to_plot
    # attempt at wireframe
    # k1ss = []
    # k2ss = []
    # kinvss = []
    # ax3d.plot_trisurf(np.array(k1ss).flatten(), np.array(k2ss).flatten(), np.array(kinvss).flatten())
    #         k1ss.append(k1s)
    #         k2ss.append(beta/alpha*k1s*k1s)
    #         kinvss.append(np.sqrt((1 - beta)/(alpha*beta))*k1s)
    for i in range(npts):
        if i % spacing == 0:
            beta = contour_pts[i,0]
            alpha = contour_pts[i,1]
            ax3d.plot(k1s, beta/alpha*k1s*k1s, np.sqrt((1 - beta)/(alpha*beta))*k1s, c='b')
    ax3d.set_xlabel(r'$k_1$')
    ax3d.set_ylabel(r'$k_2$')
    ax3d.set_zlabel(r'$k_{-1}$')
    # # contours, proof of accuracy
    # npts = 500
    # alphas = np.linspace(0.4, 0.6, npts)
    # betas = np.linspace(0.2, 0.8, npts)
    # agrid, bgrid = np.meshgrid(alphas, betas)
    # of_evals = np.empty((npts,  npts))
    # for i in range(npts):
    #     for j in range(npts):
    #         of_evals[i,j] = np.sum(np.linalg.norm(data - get_sloppy_traj(bgrid[i,j], agrid[i,j], times), axis=0)**2)
    # nlevels = 20
    # ax.contour(agrid, bgrid, np.log10(of_evals), nlevels, norm=colornorm, zorder=1)

    plt.show()
    # plt.savefig('./contours.png', bbox_inches='tight')

def of_contours():
    # set values of reaction rate constants and initial concentrations
    k1 = 0.1
    k1inv = 0.1
    k2 = 10000.0
    ca0 = 1.0
    cb0 = 0
    cc0 = 0 
    times = np.linspace(1, 5, 10)
    data = get_qssa_traj(k1, k1inv, k2, times, ca0)

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
    # ax.set_xlim((1,5))
    # ax.set_xlabel('time')
    # ax.set_ylabel(r'$y_1$, $y_2$')
    # axtwin.set_ylabel(r'$y_3$')
    # plt.tight_layout()
    # plt.show()

    npts = 30
    # k1s = np.logspace(-4, 0, npts)
    # k2s = np.logspace(4, 5, npts)
    # k1s = np.linspace(0.99999, 1.00001, npts)
    k2s = np.logspace(3, 6, npts)
    k1s = np.logspace(-2, 0, npts)
    k1invs = np.logspace(-5, 0, npts)
    dmaps_data = np.empty((3, npts*npts*npts))
    of_evals = np.empty(npts*npts*npts)
    counter = 0
    of_upper_threshold = 5.5e-5
    of_lower_threshold = 5.4e-5
    for i in range(npts):
        for j in range(npts):
            for k in range(npts):
                # !!!!!!!!!!!!!!!!!!!!
                # take log of k1inv and k2, otherwise have scaling issues
                # !!!!!!!!!!!!!!!!!!!!
                of_evals[counter] = leastsq_of(data, k1s[i], k1invs[j], k2s[k], times, ca0)
                if  of_evals[counter] < of_upper_threshold and of_evals[counter] > of_lower_threshold:
                    dmaps_data[:, counter] = np.array((np.log10(k1s[i]), np.log10(k1invs[j]), np.log10(k2s[k])))
                    counter = counter + 1

    # k1k2 = 10000
    # k2s = np.linspace(9000, 2*k1k2, npts)
    # k1s = k1k2/k2s
    # k1invs = 1.0*np.ones(npts)
    # kexps = k1s*k2s/(k1invs + k2s)
    # kscales = k1s/(k1invs + k2s)
    # kexp_grid, kscale_grid = np.meshgrid(kexps, kscales)
    # of_evals = np.empty((npts, npts))
    # of_threshold = 1e-1
    # dmaps_data = np.empty((3, npts*npts))
    # counter = 0
    # for i in range(npts):
    #     for j in range(npts):
    #         k2temp = kexp_grid[i,j]/kscale_grid[i,j]
    #         k1temp = k1k2/k2temp
    #         k1invtemp = 1.0
    #         of_evals[i,j] = leastsq_of(data, k1temp, k1invtemp, k2temp, times, ca0)
    #         if of_evals[i,j] < of_threshold:
    #             dmaps_data[:, counter] = np.array((k1temp, k1invtemp, k2temp))
    #             counter = counter + 1

    ndmapspts = counter
    of_evals = of_evals[:ndmapspts]
    dmaps_data = dmaps_data[:, :ndmapspts]
    print 'found', counter, 'points within obj. fn. threshold of[', of_upper_threshold, ',', of_lower_threshold, ']'
    print 'min of val found:', np.amin(of_evals)

    data_distances = np.empty(ndmapspts*(ndmapspts - 1)/2)
    counter = 0
    for i in range(ndmapspts):
        for j in range(i+1, ndmapspts):
            data_distances[counter] = np.linalg.norm(dmaps_data[:, i] - dmaps_data[:, j])
            counter = counter + 1
    epsilon = np.median(data_distances)
    print epsilon
    eigvals, eigvects = dmaps.dmaps_nongeneral(dmaps_data, epsilon)
    print eigvals[:5]
    dmaps.plot_data(dmaps_data[0,:], dmaps_data[1,:], dmaps_data[2,:], eigvects[:,1], r'$\Phi_1$', cmap='jet')
    dmaps.plot_data(dmaps_data[0,:], dmaps_data[1,:], dmaps_data[2,:], eigvects[:,2], r'$\Phi_2$', cmap='jet')
    # dmaps.plot_data(dmaps_data[0,:], dmaps_data[1,:], dmaps_data[2,:], eigvects[:,3], cmap='jet')
    k1s = np.power(10, dmaps_data[0,:])
    k1invs = np.power(10, dmaps_data[1,:])
    k2s = np.power(10, dmaps_data[2,:])
    # kexp = np.power(10, dmaps_data[0,:])*np.power(10, dmaps_data[0,:])/(np.power(10, dmaps_data[1,:])*np.power(10, dmaps_data[1,:]) + np.power(10, dmaps_data[2,:]))
    # kscale = np.power(10, dmaps_data[2,:])/(np.power(10, dmaps_data[1,:])*np.power(10, dmaps_data[1,:]) + np.power(10, dmaps_data[2,:]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    s = ax.scatter(eigvects[:,1], eigvects[:,2], c=np.log10(k1s))
    ax.set_xlabel(r'$\Phi_1$')
    ax.set_ylabel(r'$\Phi_2$')
    ax.set_title('colored by ' + r'$log(k_1)$' + ' values')
    cb = fig.colorbar(s, ticks=np.linspace(np.amin(np.log10(k1s)), np.amax(np.log10(k1s)), 7))
    cb.set_label(r'$log(k_{1})$')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    s = ax.scatter(eigvects[:,1], eigvects[:,2], c=np.log10(k1invs))
    ax.set_xlabel(r'$\Phi_1$')
    ax.set_ylabel(r'$\Phi_2$')
    ax.set_title('colored by ' + r'$log(k_{-1})$' + ' values')
    cb = fig.colorbar(s, ticks=np.linspace(np.amin(np.log10(k1invs)), np.amax(np.log10(k1invs)), 7))
    cb.set_label(r'$log(k_{-1})$')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    s = ax.scatter(eigvects[:,1], eigvects[:,2], c=np.log10(k2s))
    ax.set_xlabel(r'$\Phi_1$')
    ax.set_ylabel(r'$\Phi_2$')
    ax.set_title('colored by ' + r'$log(k_{2})$' + ' values')
    cb = fig.colorbar(s, ticks=np.linspace(np.amin(np.log10(k2s)), np.amax(np.log10(k2s)), 7))
    cb.set_label(r'$log(k_2)$')
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(np.log(k1s), np.log(k1invs), c=eigvects[:,1])
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(kexp, kscale, c=eigvects[:,2])
    # plt.show()
    # dmaps.plot_data(np.power(10, dmaps_data[0,:]), np.power(10, dmaps_data[1,:]), np.power(10, dmaps_data[2,:]), eigvects[:,3], cmap='jet')
    

    # # plot contours
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # nlevels = 50
    # c = ax.contourf(kexp_grid, kscale_grid, of_evals, levels=np.logspace(-2,-1,10))#nlevels)
    # ax.scatter(k1*k2/(k1inv + k2), k1/(k1inv + k2), c='r', lw=0)
    # ax.set_xlabel(r'$k_{exp}$')
    # ax.set_ylabel(r'$k_{scale}$')
    # ax.set_xlim((np.amin(kexp_grid), np.amax(kexp_grid)))
    # ax.set_ylim((np.amin(kscale_grid), np.amax(kscale_grid)))
    # fig.colorbar(c)
    # plt.show(fig)
        

    # test accuracy of equations, reproducing fig on pg 223
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # times = np.linspace(1, 5, 10)
    # exact_cs = get_exact_traj(k1, k1inv, k2, times, ca0)
    # qssa_cs = get_qssa_traj(k1, k1inv, k2, times, ca0)
    # cs = ['c', 'b', 'r']
    # for i in range(3):
    #     ax.plot(times, exact_cs[i], c=cs[i])
    # ax.plot(times, exact_cs[2], c=cs[2])
    # ax.plot(times, np.abs((exact_cs[2] - qssa_cs[2])/exact_cs[2]))
    # ax.set_yscale('log')
    # plt.show(fig)

    

if __name__=='__main__':
    test_psa()
    # run_dmaps()
    # plot_contours()
    # of_contours()
             
