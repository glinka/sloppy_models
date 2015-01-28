import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

## numerically integrates ode
# returns values of ode evaluated at different times
# from a specific set of initial conditions using forward euler
# @param f handle to autonomous function
# @param x0 initial condition for iteration
# @param t0 initial time for iteration
# @param times vector of times at which to store values
# @param dt numerical timestep
def integrate(f, x0, params, times, dt=1e-3, t0=0):
    npts = times.shape[0]
    evals = np.empty((npts, x0.shape[0]))
    times = np.insert(times, 0, t0)
    time_intervals = times[1:] - times[:-1]
    nsteps_per_interval = (time_intervals/dt).astype(int)
    x = np.copy(x0)
    for i in range(npts):
        for j in range(nsteps_per_interval[i]):
            x = x + f(x, params)*dt
        evals[i,:] = x
    return evals
        
# def test():
#     R = np.array(((np.cos(np.pi/4), -np.sin(np.pi/4)),
#                   (np.sin(np.pi/4), np.cos(np.pi/4))))
#     epsilon = 1e-1
#     A = np.array(((-1, 0),
#                   (0, -1/epsilon)))
#     x = np.array((1,4))
#     f = lambda xval: np.dot(np.dot(R, np.dot(A, np.linalg.inv(R))), xval)
#     # f = lambda xval: np.dot(A, xval)
#     dt = 1e-4
#     nsteps = 10000
#     evals = np.empty((nsteps, 2))
#     for i in range(nsteps):
#         x = x + f(x)*dt
#         evals[i] = np.copy(x)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(evals[:,0], evals[:,1], lw=0)
#     plt.show(fig)

def f(x, epsilon):
    R = np.array(((np.cos(np.pi/4), -np.sin(np.pi/4)),
                  (np.sin(np.pi/4), np.cos(np.pi/4))))
    A = np.array(((-1, 0),
                  (0, -1/epsilon)))
    return np.dot(np.dot(R, np.dot(A, np.linalg.inv(R))), x)

def lsq_error(params, times, data):
    # unpack params
    x0 = params[:2]
    epsilon = params[2]
    return np.linalg.norm(data - integrate(f, x0, epsilon, times), axis=1)

def find_min():
    x0 = np.array((1,4))
    epsilon = 1e-2
    npts = 50
    times = np.linspace(0, 1, npts)
    data = integrate(f, x0, epsilon, times)

    # generate initial points uniformly distributed +/- "max_offset" around true value
    ntrials = 10
    max_offset = 0.5
    init_offsets = max_offset*np.random.uniform(low=-1, high=1, size=(ntrials,3))
    true_val = np.hstack((x0, epsilon))
    opt_vals = np.empty((ntrials, 3))
    errs = np.empty(ntrials)
    for i in range(ntrials):
        init_guess = true_val + init_offsets[i]
        out = opt.leastsq(lsq_error, init_guess, args=(times, data), full_output=True)
        opt_vals[i] = out[0]
        errs[i] = np.sum(out[2]['fvec'])
    # testing
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=30, lw=1, c='w', label='true data')
    colornorm = colors.Normalize(vmin=0, vmax=ntrials-1)
    colormap = cm.ScalarMappable(norm=colornorm, cmap='jet')
    for i in range(ntrials):
        min_prof = integrate(f, opt_vals[i][:2], opt_vals[i][2], times)
        ax.scatter(min_prof[:,0], min_prof[:,1], lw=0, c=colormap.to_rgba(i), label=str(errs[i]))
    ax.legend()
    plt.show(fig)
    

if __name__=="__main__":
    find_min()
