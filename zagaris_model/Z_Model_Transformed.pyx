import scipy.integrate as spint
import numpy as np
cimport numpy as np
cimport cython

import algorithms.CustomErrors as CustomErrors
from Z_Model cimport Z_Model

cdef class Z_Model_Transformed(Z_Model):
    """Implementation of Antonios' custom-built model with four parameters, **two of which have been artificially transformed to create nonlinear contours**: a, b, c1 and c2. 'a' controls the curvature of the slow manifold, 'b' the curvature of the fast manifold, while 'c1' and 'c2' together influence rates of movement along the fast and slow manifold"""

    cdef double _lam_max, _epsmax, _S

    def __init__(self, np.ndarray[np.float64_t] params, double lam_max, double epsmax, double S):
        """Sets up integrator and system's parameters

        Args:
            params (array): the system parameters in the form (a, b, **c1, c2**)
        """
        a, b, c1, c2 = params
        self._lam_max = lam_max
        self._epsmax = epsmax

        # transform c1, c2 back to lam/eps to set up superclass
        self._S = S # constant swirling factor
        lam = self._S*self._lam_max*(c1*c1+c2*c2-np.arctan(c2/c1)/(2*np.pi*self._S))
        eps = self._epsmax*np.arctan(c2/c1)/(2*np.pi*self._S)
        Z_Model.__init__(self, np.array((a, b, lam, eps)))

        # self._c1 = np.sqrt(self._eps/self._epsmax + self._lam/(self._S*self._lam_max))*np.cos(2*np.pi*self._S*self._eps/self._epsmax)
        # self._c2 = np.sqrt(self._eps/self._epsmax + self._lam/(self._S*self._lam_max))*np.sin(2*np.pi*self._S*self._eps/self._epsmax)
        
    cpdef void change_transformed_parameters(self, np.ndarray[np.float64_t] new_params):
        """Changes the parameters of the ode system to 'new_params'

        Args:
            new_params (array): new parameter set to use in form (a, b, **c1, c2**)
        """
        # self._a, self._b, c1, c2 = new_params
        # # transform c1, c2 back to lam/eps to set up parent class
        # self._lam = self._S*self._lam_max*(c1*c1+c2*c2-np.arctan(c2/c1)/(2*np.pi*self._S))
        # self._eps = self._epsmax*np.arctan(c2/c1)/(2*np.pi*self._S)
        # let f(eps) = eps
        w = 1.0 # spin angular velocity
        self._a, self._b, c1, c2 = new_params
        f1 = np.arctan(c2/c1)/w
        
