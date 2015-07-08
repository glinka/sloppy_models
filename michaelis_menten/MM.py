"""A class and functions related to Michaelis Menten kinetics"""

import algorithms.CustomErrors as CustomErrors
import numpy as np
import scipy.integrate as spint
from sympy import Function, dsolve, Eq, Derivative, symbols
from collections import OrderedDict

import matplotlib.pyplot as plt

class MM_System:
    """Represents the objective function created by the reactions described by Prof. Zagaris in the form :math:`C(K, V, \sigma, \epsilon, \kappa) = \sum_i^N (P(t_i; K, V, \sigma, \epsilon, \kappa) - \hat{P_i})^2` or :math:`C(k_{inv}, k_1, k_2, S_t, E_t) = \sum_i^N (P(t_i; k_{inv}, k_1, k_2, S_t, E_t) - \hat{P_i})^2` depending on which form of the parameters are being used

    Attributes:
        _data (array): shape (3, npts) array that contains the trajectories of S, C and P in the rows, respectively, that will be fit. In fact, only the 'observable' trajectory, P, is used
        _times (array): shape (npts,) array containing the times at which concentration data was collected
        param_transform (string): indicates the parameter transformation used, as enumerated below
            * 'o': (kinv, k1, k2, St, Et)
            * 't1': (K, V, sigma, epsilon, kappa)
            * 't2': (K, V, St, epsilon, kappa)
        _param_dict (dict): dictionary of different types of parameter sets available for use. These are transformed into the 'original'/bare parameters during compution.
    """

    def __init__(self, Cs0, times, params, param_transform):
        """Completely specifies the objective fn. value given some set of test parameters"""
        # set up stiff integrator, mimicking ode15s from MATLAB
        self._integrator = spint.ode(self._enzyme_rhs)
        self._integrator.set_integrator('vode', method='bdf', order=15, nsteps=10000)
        self._Cs0 = Cs0
        self._times = times
        self.param_transform = param_transform
        all_param_dicts = {'o':['kinv', 'k1', 'k2', 'St', 'Et'],
                            't1':['K', 'V', 'sigma', 'epsilon', 'kappa'],
                            't2':['K', 'V', 'St', 'epsilon', 'kappa']}
        self._param_list = all_param_dicts[param_transform]
        self._true_param_dict = OrderedDict(zip(self._param_list, params))
        self._data = self.gen_profile(self._Cs0, self._times, self._true_param_dict.values())
        

    def of(self, params):
        """Returns value of *original* least-squares objective fn. used to fit concentration of P, i.e. in terms of :math:`(k_{inv}, k_1, k_2, S_t, E_t)`

        Args:
            params (array): the parameter set at which to evaluate the objective function, expressed according to the paramter set 'param_transform'

        Returns:
            of_eval (float): value of objective function, given by :math:`C(k_{inv}, k_1, k_2, S_t, E_t) = \sum_i^N (P(t_i; k_{inv}, k_1, k_2, S_t, E_t) - \hat{P_i})^2` where :math:`\hat{P_i}` is the :math:`i^{th}` data point to be fit
        """
        # as these initial values seems set by eqns (8) and (9), do not leave as params but set here within fn.
        # initial concentrations used in integration
        # S0 = K*sigma gives
        # params = (kinv, k1, k2, St, Et)
        true_params = self.inverse_transform_params(params)
        S0 = true_params[3]
        C0 = 0
        P0 = 0
        # initial concentrations
        Cs0 = np.array((S0, C0, P0))
        # pack up enzyme parameters to pass
        # check if enzyme_profile was succesfully computed
        try:
            enzyme_profile = self.gen_profile(Cs0, self._times, params)
        except CustomErrors.IntegrationError:
            raise
        else:
            of_eval = np.sum(np.power(enzyme_profile[:,2] - self._data[:,2], 2))
            return of_eval

    def gen_profile(self, Cs0, times, params):
        """Generates the concentration profiles of the MM species based on the initial concentrations 'Cs0' and parameters 'params', all evaluated at the times given in 'times'

        Args:
            Cs0 (array): initial concentrations of the species, (S0, C0, P0)
            times (array): times at which to evaluate the concentration profile
            params (array): parameters for which to evaluate the concentration profile, :math:`(k_{inv}, k_1, k_2, S_t, E_t)`

        .. note:: 'params' refers to the *original* parameters :math:`(k_{inv}, k_1, k_2, S_t, E_t)`, **not** the transformed :math:`(K, V, \sigma, \epsilon, \kappa)`

        Returns:
            profile (array): shape ('number of times', 3) array in which each column contains the concentrations of S, C, and P respectively at the times specified in 'times'
        """
        # old method, perhaps not suitable for such a stiff problem
        # # necessary to decrease tolerance
        # tol = 4e-14
        # profile = spint.odeint(self.enzyme_rhs, Cs0, times, args=tuple(self.inverse_transform_params(params)), atol=tol, rtol=tol)
        # new method with customized 'bdf' integrator
        # set up new initial conditions
        self._integrator.set_initial_value(Cs0, 0.0)
        self._integrator.set_f_params(*self.inverse_transform_params(params))
        profile = np.empty((times.shape[0], 3))
        for i, t in enumerate(times):
            profile[i] = self._integrator.integrate(t)
            if not self._integrator.successful():
                raise CustomErrors.IntegrationError
        return profile

        
    def _enzyme_rhs(self, t, Cs, kinv, k1, k2, St, Et):
        """ Silly private rhs function with different parameter order from 'enzyme_rhs', necessary because spint.ode and spint.odeint require different argument orders in their rhs functions
        """
        return self.enzyme_rhs(Cs, t, kinv, k1, k2, St, Et)

    def enzyme_rhs(self, Cs, t, kinv, k1, k2, St, Et):
        """Function passed to scipy integrate routine to find concentration profile

        Args:
            Cs (array): shape (3,) array containing concentrations of (S, C, P)
            t (float): time
            kinv, k1, k2, St, Et (floats): model params

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
        Sprime = -k1*(Et - C)*S + kinv*C
        Cprime = k1*(Et - C)*S - (kinv + k2)*C
        Pprime = k2*C
        rhs = np.array((Sprime, Cprime, Pprime))
        return rhs


    # def of_t(self, params):
    #     """Returns value a *transformed* least-squares objective fn. used to fit concentration of P. The actual transformation is defined by the attribute 'param_transform', e.g. 't1' cooresponds to :math:`(K, V, \sigma, \epsilon, \kappa)`.

    #     Args:
    #         params (array): the parameter set at which to evaluate the objective function, 

    #     Returns:
    #         of_eval (float): value of objective function, given by :math:`C(K, V, \sigma, \epsilon, \kappa) = \sum_i^N (P(t_i; K, V, \sigma, \epsilon, \kappa) - \hat{P_i})^2` where :math:`\hat{P_i}` is the :math:`i^{th}` data point to be fit"""
    #     # transform params from (K, V, sigma, epsilon, kappa) -> (kinv, k1, k2, St, Et)
    #     transformed_params = self.inverse_transform_params(params)
    #     of_eval = self.of(transformed_params)
    #     return of_eval

    def inverse_transform_params(self, params):
        """Transforms params based on 'param_transform' from some transformed set to the intermediate (K, V, sigma, epsilon, kappa) and finally to (kinv, k1, k2, St, Et), or, if in original transformation simply return params"""
        if self.param_transform is 'o':
            # params are already in original form (kinv, k1, k2, St, Et), return as is
            return np.copy(params)
        else:
            if self.param_transform is 't1':
                # (K, V, sigma, epsilon, kappa)
                K, V, sigma, epsilon, kappa = params
            elif self.param_transform is 't2':
                # (K, V, St, epsilon, kappa)
                K, V, St, epsilon, kappa = params
                sigma = St/K
            # invert back to original params, based on intermediate (K, V, sigma, epsilon, kappa)
            kinv = kappa*V/(epsilon*K*(sigma + 1))
            k1 = (kappa + 1)*V/(epsilon*K*K*(sigma + 1))
            k2 = V/(epsilon*K*(sigma + 1))
            St = sigma*K
            Et = epsilon*(sigma + 1)*K
            return np.array((kinv, k1, k2, St, Et))
            
    def transform_params(self, params):
        """Transforms params from (kinv, k1, k2, St, Et) to intermediate (K, V, sigma, epsilon, kappa) and finally to set specified by 'param_transform'"""
        # unpack *original* parameters
        kinv, k1, k2, St, Et = params
        K = (kinv + k2)/k1
        V = k2*Et
        sigma = St/K
        epsilon = Et/(St + K)
        kappa = kinv/k2
        if self.param_transform is 't1':
            # (K, V, sigma, epsilon, kappa)
            return np.array((K, V, sigma, epsilon, kappa))
        elif self.param_transform is 't2':
            # (K, V, St, epsilon, kappa)
            St = sigma*K
            return np.array((K, V, St, epsilon, kappa))


################################################################################
################################################################################
# end class EnzymeOF, define other helpful functions in the MM namespace
################################################################################
################################################################################







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
