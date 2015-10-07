import scipy.integrate as spint
import numpy as np

import algorithms.CustomErrors as CustomErrors
from Z_Model_no_cython import Z_Model

def class Z_Model_Transformed(Z_Model):
    """Implementation of Antonios' custom-built model with four parameters, **two of which have been artificially transformed to create nonlinear contours**: a, b, c1 and c2. 'a' controls the curvature of the slow manifold, 'b' the curvature of the fast manifold, while 'c1' and 'c2' together influence rates of movement along the fast and slow manifold"""


    def __init__(self, params, lam_max, epsmax, S):
        """Sets up integrator and system's parameters

        Args:
            params (array): the system parameters in the form (a, b, **c1, c2**)
        """
        a, b, c1, c2 = params
        self._lam_max = lam_max
        self._epsmax = epsmax

        # transform c1, c2 back to lam/eps to set up superclass
        self._S = S # constant swirling factor
        lam = self._S*self._lam_max*(c1*c1+c2*c2+np.arctan(c2/c1)/(2*np.pi*self._S))
        eps = self._epsmax*np.arctan(c2/c1)/(2*np.pi*self._S)
        Z_Model.__init__((a, b, lam, eps))

        # self._c1 = np.sqrt(self._eps/self._epsmax + self._lam/(self._S*self._lam_max))*np.cos(2*np.pi*self._S*self._eps/self._epsmax)
        # self._c2 = np.sqrt(self._eps/self._epsmax + self._lam/(self._S*self._lam_max))*np.sin(2*np.pi*self._S*self._eps/self._epsmax)
        
    def void change_transformed_parameters(self, new_params):
        """Changes the parameters of the ode system to 'new_params'

        Args:
            new_params (array): new parameter set to use in form (a, b, **c1, c2**)
        """
        self._a, self._b, c1, c2 = params
        # transform c1, c2 back to lam/eps to set up superclass
        self._lam = self._S*self._lam_max*(c1*c1+c2*c2+np.arctan(c2/c1)/(2*np.pi*self._S))
        self._eps = self._epsmax*np.arctan(c2/c1)/(2*np.pi*self._S)
