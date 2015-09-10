import scipy.integrate as spint
import numpy as np
cimport numpy as np
cimport cython

import algorithms.CustomErrors as CustomErrors

cdef class Z_Model:
    """Implementation of Antonios' custom-built model with four parameters: a, b, lambda and epsilon. 'a' controls the curvature of the slow manifold, 'b' the curvature of the fast manifold, 'lambda' the rate of motion along the slow manifold and 'epsilon' the rate of motion along the fast manifold"""

    cdef double _a, _b, _lam, _eps
    cdef object _integrator

    def __init__(self, np.ndarray[np.float64_t] params):
        """Sets up integrator and system's parameters

        Args:
            params (array): the system parameters in the form (a, b, lambda, epsilon)
        """
        # # this cannot work with cython as classes are stored as structs and the following would require variable-length structs, thus we unpack to the individual parameters
        # cdef np.ndarray[np.float64_t] self._params = params

        # unpack params for clarity
        self._a = params[0]
        self._b = params[1]
        self._lam = params[2]
        self._eps = params[3]
        self._integrator = spint.ode(self._rhs)
        self._integrator.set_integrator('lsoda')
        # self._integrator.set_f_params(params)

    # @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef np.ndarray[np.float64_t] _rhs(self, float t, np.ndarray[np.float64_t] x):
        """Returns the right-hand-side of the ODE sytem at a given system state with given parameters

        Args:
            t (float): time
            x (array): system state

        Returns:
            x (array): evaluation of system at 'x' and 'params'
        """
        # evaluate rhs and return
        return -1/((1-4*self._a*self._b*x[0]*x[1])*self._eps)*np.array((2*self._b*x[1]*(x[1]-self._a*x[0]**2) + self._eps*self._lam*(x[0]-self._b*x[1]**2), (x[1]-self._a*x[0]**2) + 2*self._eps*self._a*self._lam*x[0]*(x[0]-self._b*x[1]**2)))

    # @cython.boundscheck(False) # turn of bounds-checking for entire function
    def get_trajectory(self, np.ndarray[np.float64_t] x0, np.ndarray[np.float64_t] times):
        """Returns complete trajectory of 'x' at 'times' starting from x(0) = x0

        Args:
            x0 (array): initial system state, **assumed to be given at t=0**
            times (array): the times at which to collect points

        Returns:
            trajectory (array): values of system state 'x' at 'times'
        """
        # assume x0 = x(0)
        self._integrator.set_initial_value(x0, 0.0)
        cdef np.ndarray[np.float64_t, ndim=2] trajectory = np.empty((times.shape[0], x0.shape[0]))
        # if we record initial condition, remove t = 0 from times, otherwise numpy will fail to integrate. tack on initial conditions after integration
        cdef np.ndarray[np.float64_t] times_to_integrate = times
        cdef int offset = 0
        cdef int i
        cdef double t
        if times[0] == 0:
            trajectory[0] = x0
            offset = 1
            times_to_integrate = times[1:]
        for i, t in enumerate(times_to_integrate):
            trajectory[i+offset] = self._integrator.integrate(t)
            if not self._integrator.successful():
                raise CustomErrors.IntegrationError
        return trajectory

    cpdef void change_parameters(self, np.ndarray[np.float64_t] new_params):
        """Changes the parameters of the ode system to 'new_params'

        Args:
            new_params (array): new parameter set to use in form (a, b, lambda, epsilon)
        """
        self._a = new_params[0]
        self._b = new_params[1]
        self._lam = new_params[2]
        self._eps = new_params[3]
