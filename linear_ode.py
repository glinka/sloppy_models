import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import util_fns as uf

## numerically integrates ode
# returns values of ode evaluated at different times
# from a specific set of initial conditions using forward euler
# @param f handle to autonomous function
# @param x0 initial condition for iteration
# @param t0 initial time for iteration
# @param times vector of times at which to store values
# @param dt numerical timestep
def integrate(f, x0, params, times, dt=1e-6, t0=0):
    npts = times.shape[0]
    evals = np.empty((npts, x0.shape[0]))
    times = np.insert(times, 0, t0)
    time_intervals = times[1:] - times[:-1]
    nsteps_per_interval = (time_intervals/dt).astype(int)
    x = np.copy(x0)
    for i in range(npts):
        for j in range(nsteps_per_interval[i]):
            k1 = f(x, params)
            k2 = f(x + 0.5*k1, params)
            k3 = f(x + 0.5*k2, params)
            k4 = f(x + k3, params)
            x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
            # silly euler
            # x = x + f(x, params)*dt
        evals[i,:] = x
    return evals
        
## linear ode test function
# returns evaluation of rhs fn. at given value of state variable
# "x" and parameter "sigma"
# @param x state variable at which function will be evaluated
# @param sigma parameter
def f_test(x, sigma):
    R = np.array(((np.cos(np.pi/4), -np.sin(np.pi/4)),
                  (np.sin(np.pi/4), np.cos(np.pi/4))))
    A = np.array(((-1, 0),
                  (0, -sigma)))
    return np.dot(np.dot(R, np.dot(A, np.linalg.inv(R))), x)

## linear separable ode test function
# returns evaluation of rhs fn. at given value of state variable
# "x" and parameter "sigma"
# @param x state variable at which function will be evaluated
# @param sigma parameter
def f_sep(x, sigma):
    A = np.array(((-1, 0),
                  (0, -sigma)))
    return np.dot(A, x)

def lsq_error(params, times, data):
    # unpack params
    x0 = params[:2]
    sigma = params[2]
    return np.linalg.norm(data - integrate(f_test, x0, sigma, times), axis=1)

def find_min():
    x0 = np.array((1,4))
    sigma = 100
    npts = 50
    times = np.linspace(0, 1, npts)
    data = integrate(f_test, x0, sigma, times)

    # generate initial points uniformly distributed +/- "max_offset" around true value
    ntrials = 2
    max_offset = 0.5
    init_offsets = max_offset*np.random.uniform(low=-1, high=1, size=(ntrials,3))
    true_val = np.hstack((x0, sigma))
    opt_vals = np.empty((ntrials, 3))
    errs = np.empty(ntrials)
    for i in range(ntrials):
        init_guess = true_val + init_offsets[i]
        out = opt.leastsq(lsq_error, init_guess, args=(times, data), full_output=True)
        opt_vals[i] = out[0]
        errs[i] = np.sum(out[2]['fvec'])
        uf.progress_bar(i+1, ntrials)
    np.savetxt('./data/optvals.csv', opt_vals, delimiter=',')
    np.savetxt('./data/errs.csv', errs, delimiter=',')

def lsq_error2(params, x10, times, data):
    x20 = params[0]
    sigma = params[1]
    npts = times.shape[0]
    trial = np.empty((npts,2))
    trial[:,0] = x10*np.exp(-times)
    trial[:,1] = x20*np.exp(-sigma*times)
    return np.linalg.norm(data - trial, axis=1)

def find_min(data, times, x0, sigma):
    out = opt.leastsq(lsq_error2, (x0[1], sigma), args=(x0[0], times, np.copy(data)), full_output=True)
    print 'error in least square fitting', np.sum(out[2]['fvec']**2)
    return out[0]


def find_contours(noise=False):
    npts = 10
    times = np.linspace(0, 0.05, npts)
    x10_truth = 1
    x20 = 4
    SIGMA = 1000
    sigma = SIGMA
    data = np.empty((npts,2))
    if noise is False:
        noise_stdev = np.array((0,))
    else:
        noise_stdev = np.logspace(-4, -2, 8)
    for i in range(noise_stdev.shape[0]):
        if noise:
            x20 = 4
            data[:,0] = x10_truth*np.exp(-times) + np.random.normal(loc=0, scale=noise_stdev[i], size=npts)
            data[:,1] = x20*np.exp(-SIGMA*times) + np.random.normal(loc=0, scale=noise_stdev[i], size=npts)
            x10_truth = data[0,0]
            x20 = data[0,1]
        else:
            data[:,0] = x10_truth*np.exp(-times)
            data[:,1] = x20*np.exp(-SIGMA*times)            
        x0 = np.array((x10_truth, x20))
        x20_truth, sigma_truth = find_min(data, times, x0, SIGMA)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:,0], data[:,1], c='b')
        ax.scatter(x10_truth*np.exp(-times), x20_truth*np.exp(-sigma_truth*times), c='r', s=80)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('phase plot')
        plt.show(fig)

        init_err = F(data, times, x10_truth, x20_truth, sigma_truth, 0)
        print 'numerical error in integration/fitting:', init_err


        if noise:
            nconts = 12
            contours = np.linspace(init_err+1e-8, init_err+1e-4, nconts)
        else:
            nconts = 16
            contours = np.logspace(-4, -8, nconts)
        # colornorm = colors.Normalize(vmin=1e-8, vmax=1e-4)
        colornorm = colors.Normalize(vmin=0, vmax=nconts-1)
        colormap = cm.ScalarMappable(norm=colornorm, cmap='jet')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        iters = 0
        ylims = [1e8, -1e8]
        maxinner_iters = 100000
        dsigma_min = 1e-10
        x20_max = 0.001
        for c in contours:

            # fig = plt.figure()
            # ax = fig.add_subplot(111)

            sigmas = []
            x20s = []
            # increase sigma
            x20 = x20_truth
            sigma = sigma_truth
            inner_iters = 0
            dsigma = 0.01
            while x20 is not False and np.abs(x20-4) < x20_max and inner_iters < maxinner_iters:
                x20 = find_contour_point(data, times, x10_truth, x20, sigma, c)
                if x20 is not False:
                    x20s.append(x20)
                    sigmas.append(sigma)
                elif dsigma > dsigma_min:
                    sigma = sigma - dsigma
                    dsigma = 0.1*dsigma
                    x20 = x20s[-1]
                sigma = sigma + dsigma
                inner_iters = inner_iters + 1
            # decrease sigma
            x20 = x20_truth
            sigma = sigma_truth
            inner_iters = 0
            dsigma = 0.01
            while x20 is not False and np.abs(x20-4) < x20_max and inner_iters < maxinner_iters:
                x20 = find_contour_point(data, times, x10_truth, x20, sigma, c)
                if x20 is not False:
                    x20s.append(x20)
                    sigmas.append(sigma)
                elif dsigma > dsigma_min:
                    sigma = sigma + dsigma
                    dsigma = 0.1*dsigma
                    x20 = x20s[-1]
                sigma = sigma - dsigma
                inner_iters = inner_iters + 1
            print 'found', len(x20s), 'pts at contour value', c
            print F(data, times, x10_truth, x20s[-1], sigmas[-1], c)
            ax.scatter(sigmas, x20s, lw=0, c=colormap.to_rgba(iters), label='f=%1.2e' % c, zorder=3)
            if np.amin(x20s) < ylims[0]:
                ylims[0] = np.amin(x20s)
            if np.amax(x20s) > ylims[1]:
                ylims[1] = np.amax(x20s)
            iters = iters + 1
        # gca().get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        # mpl.rcParams['axes.formatter.useoffset'] = False
        ax.set_ylim(ylims)

        # # TESTING
        # npts = 10
        # times = np.linspace(0, 0.075, npts)
        # x10 = 1
        # x20 = 4
        # sigma = 100
        # data = np.empty((npts,2))
        # data[:,0] = x10*np.exp(-times)
        # data[:,1] = x20*np.exp(-sigma*times)            
        # nspts = 100
        # sigmas = np.linspace(99.4, 100.6, nspts)
        # nxpts = 100
        # x20s = np.linspace(3.999, 4.001, nxpts)
        # x20grid, sigmagrid = np.meshgrid(x20s, sigmas)
        # Z = np.empty((nspts, nxpts))
        # for i in range(nspts):
        #     for j in range(nxpts):
        #         Z[i,j] = F(data, times, x10, x20grid[i,j], sigmagrid[i,j], 0)
        # # fig = plt.figure()
        # # ax = fig.add_subplot(111)
        # nlevels = 40
        # ax.contourf(sigmagrid, x20grid, Z, nlevels, cmap='jet', norm=colornorm, zorder=2)
        # # plt.show(fig)
        # # END TESTING

        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel('x20')
        if noise is True:
            ax.set_title('normally distributed error with stdev: ' + str(noise_stdev[i]))
        else:
            ax.set_title('contours with perfect data')
        ax.legend(loc=4, fontsize=20)
        plt.show(fig)
        



def mesh_contour():
    npts = 10
    times = np.linspace(0.006, 0.05, npts)
    x10 = 1
    x20 = 4
    sigma = 1000
    data = np.empty((npts,2))
    noise = False
    if noise:
        noise_stdev = np.logspace(-3, -3, 4)
        ynoise_stdev = np.logspace(-3, -3, 4)
    else:
        noise_stdev = np.array((1,))
        ynoise_stdev = np.array((1,))
    for k in range(noise_stdev.shape[0]):
        data[:,0] = x10*np.exp(-times) + int(noise)*np.random.normal(loc=0, scale=noise_stdev[k], size=npts)
        data[:,1] = x20*np.exp(-sigma*times) + int(noise)*np.random.normal(loc=0, scale=ynoise_stdev[k], size=npts)

        out = opt.leastsq(lsq_error2, (x20, sigma), args=(x10, times, np.copy(data)), full_output=True, ftol=1e-14, xtol=1e-14)
        print 'opt params:', out[0]

        optval = F(data, times, x10, out[0][0], out[0][1], 0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:,0], data[:,1])
        # nspts = 250
        # sigmas = np.linspace(out[0][1] - 2, out[0][1] + 2, nspts) # 700, 1600, nspts)
        # nxpts = 250
        # x20s = np.linspace(out[0][0] - 0.001, out[0][0] + 0.001, nspts) # 3.95, 4.05, nxpts)
        optsigma = out[0][1]
        optx20 = out[0][0]
        nspts = 600
        # sigmas = np.linspace(optsigma - 1000, optsigma + 200, nspts) # 700, 1600, nspts)
        # nxpts = 200
        # x20s = np.linspace(optx20 - 6, optx20 + 0, nspts) # 3.95, 4.05, nxpts)
        sigmas = np.linspace(100, 250, nspts) # 700, 1600, nspts)
        nxpts = 600
        x20s = np.linspace(-0.25, 0.25, nspts) # 3.95, 4.05, nxpts)
        x20grid, sigmagrid = np.meshgrid(x20s, sigmas)
        Z = np.empty((nspts, nxpts))
        
        # # OPTIONAL REFINEMENT OF OPTIMAL VALUES
        # for r in range(1):
        #     for i in range(nspts):
        #         for j in range(nxpts):
        #             if F(data, times, x10, x20grid[i,j], sigmagrid[i,j], optval) < 0:
        #                 optval = F(data, times, x10, x20grid[i,j], sigmagrid[i,j], 0)
        #                 optsigma = np.copy(sigmagrid[i,j])
        #                 optx20 = np.copy(x20grid[i,j])
        #     print optsigma - out[0][1], optx20 - out[0][0]        
        #     nspts = 600
        #     sigmas = np.linspace(optsigma - 20, optsigma + 20, nspts) # 700, 1600, nspts)
        #     nxpts = 600
        #     x20s = np.linspace(optx20 - 1, optx20 + 1, nspts) # 3.95, 4.05, nxpts)
        #     x20grid, sigmagrid = np.meshgrid(x20s, sigmas)
        # Z = np.empty((nspts, nxpts))
        # # END OPTIONAL REFINEMENT


        for i in range(nspts):
            for j in range(nxpts):
                Z[i,j] = F(data, times, x10, x20grid[i,j], sigmagrid[i,j], optval)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        nlevels = 60
        print 'min contour:', np.amin(Z)
        c = ax.contourf(sigmagrid, x20grid, Z - np.amin(Z), levels = np.logspace(-5, -2, 50)) 
        if noise:
            ax.set_title('contour with N(0,%1.2e) noise in data' % noise_stdev[k])
        else:
            ax.set_title('contour of objective function')
        fig.colorbar(c)
        plt.show(fig)

def g1(x10, t):
    return x10*np.exp(-t)

def g2(x20, sigma, t):
    return x20*np.exp(-sigma*t)

def F(data, times, x10, x20, sigma, c):
    return np.sum((g1(x10, times) - data[:,0])**2 + (g2(x20, sigma, times) - data[:,1])**2) - c

# shape (2,1) function with arclength evaluation as (0,1) entry
def F_arclength(data, times, x10, x20, sigma, c, dx20dS, dSigmadS, x20_init, sigma_init, ds):
    return np.array((F(data, times, x10, x20, sigma, c), (x20 - x20_init)*dx20dS + (sigma - sigma_init)*dSigmadS - ds))

def DF(data, times, x20, sigma, dx20dS, dSigmadS):
    return np.array(((dFdx20(data, times, x20, sigma), dFdsigma(data, times, x20, sigma)),(dx20dS, dSigmadS)))

def dFdx20(data, times, x20, sigma):
    return 2*np.sum((g2(x20, sigma, times) - data[:,1])*np.exp(-sigma*times))

def dFdsigma(data, times, x20, sigma):
    return -2*x20*np.sum(times*np.exp(-sigma*times)*(g2(x20, sigma, times) - data[:,1]))

def find_zero_arclength(data, times, x10, x20, sigma, x20_prev, sigma_prev, contour_val, ds, dx20dS, dSigmadS):
    # save initial values for use NR
    # zeroth order continuation LOL
    maxiters = 1000
    tol = 1e-10
    iters = 0
    Feval = F_arclength(data, times, x10, x20, sigma, contour_val, dx20dS, dSigmadS, x20_prev, sigma_prev, ds)
    DFeval = DF(data, times,x20, sigma, dx20dS, dSigmadS)
    err = np.linalg.norm(Feval)
    while err > tol and iters < maxiters:
        dsigma = (DFeval[1,0]*Feval[0]/DFeval[0,0]- Feval[1])/(DFeval[1,1] - DFeval[1,0]*DFeval[0,1]/DFeval[0,0])
        sigma = sigma + dsigma
        x20 = x20 + (DFeval[0,1]*dsigma - Feval[0])/DFeval[0,0]
        Feval = F_arclength(data, times, x10, x20, sigma, contour_val, dx20dS, dSigmadS, x20_prev, sigma_prev, ds)
        DFeval = DF(data, times, x20, sigma, dx20dS, dSigmadS)
        err = np.linalg.norm(Feval)
        iters = iters + 1
    if iters == maxiters:
        # print 'failed to converge in', maxiters, 'iterations at arclength', ds, ', exiting'
        return False, False
    else:
        return x20, sigma
        

def find_contour_arclength():
    npts = 10
    times = np.linspace(0, 0.05, npts)
    x10 = 1
    x20 = 4
    sigma = 1000
    data = np.empty((npts,2))
    data[:,0] = x10*np.exp(-times)# + np.random.normal(loc=0, scale=1e-3, size=npts)
    data[:,1] = x20*np.exp(-sigma*times)# + np.random.normal(loc=0, scale=1e-2, size=npts)
    contour_val = 1e-4
    # find initial point on contour
    x20 = find_contour_point(data, times, x10, x20 + 1, sigma, contour_val)
    # calculate initial value of derivates with respect to arclength, from Kelley
    # wut is the correct sign of d\lambda dS? don't matter none at the beginning
    dSigmadS = -np.sqrt((dFdsigma(data, times, x20, sigma)/dFdx20(data, times, x20, sigma))**2 + 1)
    dx20dS = -dFdsigma(data, times, x20, sigma)/dFdx20(data, times, x20, sigma)*dSigmadS
    nsteps = 200
    contour = np.empty((nsteps, 2))
    converged = True
    i = 0
    ds = 1
    ds_min = 1e-8
    x20_prev = x20
    sigma_prev = sigma
    while i < nsteps and converged is True:
        x20, sigma = find_zero_arclength(data, times, x10, x20, sigma, x20_prev, sigma_prev, contour_val, ds, dx20dS, dSigmadS)
        # if not converged and reached min arclength step, exit
        if int(x20 + sigma) == 0 and ds < ds_min:
            converged = False
        # if not converged and min arclength not yet reached, decrease arclength and try again
        # so adaptive
        elif int(x20 + sigma) == 0:
            ds = ds/10.0
            x20, sigma = contour[i-1]
        # otherwise converged, proceed as normal
        else:
            # print F(data, times, x10, x20, sigma, contour_val)
            ds = 1
            contour[i] = (x20, sigma)
            dFdSig = dFdsigma(data, times, x20, sigma)
            dFdX2 = dFdx20(data, times, x20, sigma)
            # re-estimate derivates instead of exactly calculating to remove worries about sign changes (?)
            dSigmadS = 1/(dSigmadS - dFdSig*dx20dS/dFdX2)
            dx20dS = -dFdSig*dSigmadS/dFdX2
            norm = np.sqrt(dSigmadS**2 + dx20dS**2)
            dSigmadS = dSigmadS/norm
            dx20dS = dx20dS/norm
            # first order continuation
            x20_prev = x20
            sigma_prev = sigma
            x20 = x20 + dx20dS*ds
            sigma = sigma + dSigmadS*ds
            i = i + 1
    print 'min y', np.amin(contour[:i,0])
    i = i - 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(contour[:i,1], contour[:i,0], label="f=%1.3e" % contour_val, c=np.linspace(0,1,i))
    ax.legend()
    plt.savefig('./figs/alc.png')
    plt.show(fig)
        

def find_contour_point(data, times, x10, x20, sigma, c):
    maxiters = 1000
    tol = 1e-14
    iters = 0
    err = F(data, times, x10, x20, sigma, c)
    while err > tol and iters < maxiters:
        x20 = x20 - err/dFdx20(data, times, x20, sigma)
        err = F(data, times, x10, x20, sigma, c)
        iters = iters + 1
    if iters == maxiters:
        # print 'failed to converge in', maxiters, 'iterations, exiting'
        return False
    else:
        return x20 

def two_dim_fit():
    addnoise = True
    npts = 10
    noise = np.random.normal(scale=0.0001, size=npts)
    times = np.linspace(0.1, 0.2, npts)
    sigma = 1000
    slope = 5
    fevals = np.exp(-sigma*times) + np.sin(slope*times) +  + int(addnoise)*noise
    obj_fn = lambda sig, sl: np.sum(np.power(fevals - np.exp(-sig*times) - np.sin(sl*times), 2))
    gridsize = 100
    sigmas = np.linspace(90, 1100, gridsize)
    slopes = np.linspace(4.999, 5.001, gridsize)
    slopegrid, sigmagrid = np.meshgrid(slopes, sigmas)
    obj_fn_evals = np.empty((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize):
            obj_fn_evals[i,j] = obj_fn(sigmagrid[i,j], slopegrid[i,j])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(times, fevals)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nlevels = 50
    ax.contourf(sigmagrid, slopegrid, obj_fn_evals, nlevels) #levels=np.logspace(-, -7, nlevels))
    print 'max contour val:', np.amax(obj_fn_evals)
    print 'min contour val:', np.amin(obj_fn_evals)
    plt.show(fig)
    

def one_dim_exp_fit():
    npts = 10
    times = np.linspace(0.007, 0.05, npts)
    # times = np.linspace(0.05, 0.1, npts)
    noise = np.random.normal(scale=0.0005, size=npts)
    xnoise = 0*np.random.normal(scale=0.002, size=npts)
    sigma = 1000
    sigma1 = 1000
    sigma2 = 500
    addnoise = True
    obj_fn = lambda s: [np.sum(np.power(np.exp(-sigma*times) + int(addnoise)*noise - np.exp(-i*times), 2)) for i in s]
    # obj_fn = lambda s: [np.sum(np.power(np.exp(-sigma*(times + int(addnoise)*xnoise)) + (times + int(addnoise)*xnoise)/sigma + int(addnoise)*noise - np.exp(-i*times) - times/i, 2)) for i in s]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(times + int(addnoise)*xnoise, np.exp(-sigma*times) + times/sigma + int(addnoise)*noise, s=50)
    ax.set_ylabel('y')
    ax.set_xlabel('t')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'$f(\sigma)$')
    ax.set_xlabel(r'$\sigma$')
    upperlimit = 1000
    lowerlimit = 300
    ax.plot(np.linspace(sigma-lowerlimit, sigma+upperlimit, 1000), obj_fn(np.linspace(sigma-lowerlimit, sigma+upperlimit, 1000)), lw=2)
    # ax.set_ylim((0.5, 0.7))
    # ax.set_ylim((0, 4))
    plt.show(fig)

    # obj_fn = lambda s1, s2: np.sum(np.power(np.exp(-sigma1*times) + np.exp(-sigma2*times) - np.exp(-s1*times) - np.exp(-s2*times), 2))
    # # offset = 600
    # nspts = 200
    # s1s = np.linspace(sigma1 - 600, sigma1 + 100, nspts) # 700, 1600, nspts)
    # nxpts = 200
    # s2s = np.linspace(sigma2 - 100, sigma2 + 600, nspts) # 3.95, 4.05, nxpts)
    # Z = np.empty((nspts, nxpts))
    # for i in range(nspts):
    #     for j in range(nxpts):
    #         Z[i,j] = obj_fn(s1s[i], s2s[j])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # nlevels = 100
    # ax.contourf(s1s, s2s, Z, levels=np.logspace(-9,-1,50))#nlevels)
    # plt.show(fig)


if __name__=="__main__":
    # mesh_contour()
    # find_contours()
    # find_contour_arclength()
    one_dim_exp_fit()
    # two_dim_fit()
