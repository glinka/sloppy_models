import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint

def rhs(t, x, params):
    a,b,lam,eps = params
    x = -1/((1-4*a*b*x[0]*x[1])*eps)*np.array((2*b*x[1]*(x[1]-a*x[0]**2) + eps*lam*(x[0]-b*x[1]**2), (x[1]-a*x[0]**2) + 2*eps*a*lam*x[0]*(x[0]-b*x[1]**2)))
    return x

def get_trajectory(integrator, x0, times):
    # assume x0 = x(0)
    integrator.set_initial_value(x0, 0.0)
    trajectory = np.empty((times.shape[0], x0.shape[0]))
    # if we record initial condition, remove t = 0 from times, otherwise numpy will fail to integrate. tack on initial conditions after integration
    times_to_integrate = times
    offset = 0
    if times[0] == 0:
        trajectory[0] = x0
        offset = 1
        times_to_integrate = times[1:]
    for i, t in enumerate(times_to_integrate):
        trajectory[i+offset] = integrator.integrate(t)
        if not integrator.successful():
            print 'failed to integrate'
            return False
    return trajectory
