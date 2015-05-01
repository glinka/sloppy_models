"""A class and functions related to Michaelis Menten kinetics"""

import numpy as np
import scipy.integrate as spint
from sympy import Function, dsolve, Eq, Derivative, symbols

class EnzymeOF:
    """Represents the objective function created by the reactions described by Prof. Zagaris in the form :math:`C(K, V, \sigma, \epsilon, \kappa) = \sum_i^N (P(t_i; K, V, \sigma, \epsilon, \kappa) - \hat{P_i})^2` or :math:`C(k_{inv}, k_1, k_2, S_t, E_t) = \sum_i^N (P(t_i; k_{inv}, k_1, k_2, S_t, E_t) - \hat{P_i})^2` depending on which form of the parameters are being used"""

    def __init__(self, data, times):
        """Completely specifies the objective fn. value given some set of test parameters"""
        self._data = data
        self._times = times

    def of(self, params):
        """Returns value of *original* least-squares objective fn. used to fit concentration of P, i.e. in terms of :math:`(k_{inv}, k_1, k_2, S_t, E_t)`

        Args:
            params (array): the parameter set :math:`(k_{inv}, k_1, k_2, S_t, E_t)` at which to evaluate the objective function, 

        Returns:
            of_eval (float): value of objective function, given by :math:`C(k_{inv}, k_1, k_2, S_t, E_t) = \sum_i^N (P(t_i; k_{inv}, k_1, k_2, S_t, E_t) - \hat{P_i})^2` where :math:`\hat{P_i}` is the :math:`i^{th}` data point to be fit
        """
        # transform params from (kinv, k1, k2, St, Et) -> (K, V, sigma, epsilon, kappa)
        transformed_params = transform_params(params)
        of_eval = self.of_t(transformed_params)
        return of_eval

    def of_t(self, params):
        """Returns value of *transformed* least-squares objective fn. used to fit concentration of P, i.e. in terms of :math:`(K, V, \sigma, \epsilon, \kappa)`

        Args:
            params (array): the parameter set :math:`(K, V, \sigma, \epsilon, \kappa)` at which to evaluate the objective function, 

        Returns:
            of_eval (float): value of objective function, given by :math:`C(K, V, \sigma, \epsilon, \kappa) = \sum_i^N (P(t_i; K, V, \sigma, \epsilon, \kappa) - \hat{P_i})^2` where :math:`\hat{P_i}` is the :math:`i^{th}` data point to be fit"""
        # as these initial values seems set by eqns (8) and (9), do not leave as params but set here within fn.
        # initial concentrations used in integration
        # S0 = K*sigma gives
        # unpack params for clarity
        (K, V, sigma, epsilon, kappa) = params
        S0 = K*sigma
        C0 = 0
        P0 = 0
        # initial concentrations
        Cs0 = np.array((S0, C0, P0))
        # pack up enzyme parameters to pass
        enzyme_profile = gen_profile(Cs0, self._times, params)
        of_eval = np.sum(np.power(enzyme_profile[:,2] - self._data[:,2], 2))
        return of_eval

################################################################################
################################################################################
# end class EnzymeOF, define other helpful functions in the MM namespace
################################################################################
################################################################################


def transform_params(params):
    """Transforms params from (kinv, k1, k2, St, Et) -> (K, V, sigma, epsilon, kappa)"""
    # unpack *original* parameters
    kinv, k1, k2, St, Et = params
    # transform
    K = (kinv + k2)/k1
    V = k2*Et
    sigma = St/K
    epsilon = Et/(St + K)
    kappa = kinv/k2
    return np.array((K, V, sigma, epsilon, kappa))

def enzyme_rhs(Cs, t, K, V, sigma, epsilon, kappa):
        """Function passed to scipy integrate routine to find concentration profile

        Args:
            Cs (array): shape (3,) array containing concentrations of (S, C, P)
            t (float): time
            K, V, sigma, epsilon, kappa (floats): model params

        Returns:
            rhs (array): evaluation of rates of change of three species S, C and P

        >>> import MM
        >>> params = np.array((1.0, 1.0, 1.0, 1e-2, 10.0))  # params = (K, V, sigma, epsilon, kappa)
        >>> Cs0 = np.array((params[0]*params[2], 0, 0)) #  Cs0 = np.array((S0, C0, P0))
        >>> tscale = (params[2] + 1)*params[0]/params[1]
        >>> npts = 20
        >>> times = tscale*np.linspace(1,npts,npts)/5.0
        >>> enzyme_data = MM.gen_profile(Cs0, times, params)
        >>> [plt.plot(times, enzyme_data[:,i]) for i in range(3)]
        >>> plt.show()
        """
        S = Cs[0]
        C = Cs[1]
        P = Cs[2]
        Sprime = (kappa + 1)*V*(-S + C*S/(epsilon*(sigma + 1)*K) + C*kappa/(epsilon*(sigma + 1)*(kappa + 1)))/K
        Cprime = (kappa + 1)*V*(S - C*S/(epsilon*(sigma + 1)*K) - C/(epsilon*(sigma + 1)))/K
        Pprime = V*C/(epsilon*(sigma + 1)*K)
        rhs = np.array((Sprime, Cprime, Pprime))
        return rhs

def gen_profile(Cs0, times, params):
    """Generates the concentration profiles of the MM species based on the initial concentrations 'Cs0' and parameters 'params', all evaluated at the times given in 'times'

    Args:
        Cs0 (array): initial concentrations of the species, (S0, C0, P0)
        times (array): times at which to evaluate the concentration profile
        params (array): parameters for which to evaluate the concentration profile

    .. note:: 'params' refers to the *transformed* parameters :math:`(K, V, \sigma, \epsilon, \kappa)`, **not** the original :math:`(k_{inv}, k_1, k_2, S_t, E_t)`

    Returns:
        profile (array): shape ('number of times', 3) array in which each column contains the concentrations of S, C, and P respectively at the times specified in 'times'
    """
    # necessary to decrease tolerance
    tol = 1e-13
    profile = spint.odeint(enzyme_rhs, Cs0, times, args=tuple(params), atol=tol, rtol=tol)
    return profile

def symbolic_enzyme_integrationparams(K, V, sigma, epsilon, kappa):
    """Attempts but ultimately fails to symbolically integrate the enzyme kinetic odes. Worth a shot."""
    S, C, P = symbols('S,C,P', function=True)
    t = symbols('t')
    odes = (
        Eq(S(t).diff(t), (kappa + 1)*V*(-S(t) + C(t)*S(t)/(epsilon*(sigma + 1)*K) + C(t)*kappa/(epsilon*(sigma + 1)*(kappa + 1)))/K),
        Eq(C(t).diff(t), (kappa + 1)*V*(S(t) - C(t)*S(t)/(epsilon*(sigma + 1)*K) - C(t)/(epsilon*(sigma + 1)))/K),
        Eq(P(t).diff(t), V*C(t)/(epsilon*(sigma + 1)*K))
    )
    print odes
    print dsolve(odes)
