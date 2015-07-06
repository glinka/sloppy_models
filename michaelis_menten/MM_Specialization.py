"""Provides functions that evaluate the objective function and corresponding Jacobian/gradient while keeping a specific set of parameters constant. Used with PseudoArclengthContinuation module to find contours of the MM system"""

from MM import MM_System
from algorithms.Derivates import gradient
from collections import OrderedDict
import numpy as np
    
class MM_Specialization(MM_System):
    """Allows for the manipulation of specific MM parameters while keeping others constant

    Attributes:
        _state_params (list): strings specifying which parameters will be varied during operation, considered to comprise the state of the system
        _continuation_param (str): specifies the parameter that will be changed during continuation methods
        _contour_val (float): value of the objective function contour being explored
    """

    def __init__(self, Cs0, times, params, param_transform, state_params, continuation_param, contour_val):
        MM_System.__init__(self, Cs0, times, params, param_transform)
        # ensure all parameters specified as state are in specified parameter set
        for param in state_params:
            if param not in self._param_list:
                print 'Parameter transformation specified by "param_transform" does not contain one of params in "variable_params, exiting"'
                exit()
        if continuation_param not in self._param_list:
            print 'Continuation param "continuation_param" not part of specified parameter transform group, exiting"'
            exit()
        self._state_params = state_params
        self._continuation_param = continuation_param
        self._const_params = []
        for param in self._param_list:
            if param not in state_params and param is not continuation_param:
                self._const_params.append(param)
        self._variable_param_dict = OrderedDict.fromkeys(self._param_list)
        self._contour_val = contour_val

                
    def f(self, params, continuation_param):
        """Objective function using 'params' and 'continuation_param' as variables. The remaining parameters, specified in '_const_params', will be set to values in '_params' before evaluation.

        Args:
            params (array): (len(_state_params),1) array of parameters specifying "state" of system
            continuation_param (float): the continuation parameter value

        Returns:
            of_eval (array): (1,) array containing objective function minus 'contour_val' :math:`\alpha`, given by :math:`C(k_{inv}, k_1, k_2, S_t, E_t) = \sum_i^N (P(t_i; k_{inv}, k_1, k_2, S_t, E_t) - \hat{P_i})^2 - \alpha` where :math:`\hat{P_i}` is the :math:`i^{th}` data point to be fit
        """
        self._variable_param_dict[self._continuation_param] = continuation_param
        for i, param in enumerate(self._state_params):
            self._variable_param_dict[param] = params[i]
        for param in self._const_params:
            self._variable_param_dict[param] = self._true_param_dict[param]
        of_eval = np.array((self.of(self._variable_param_dict.values()) - self._contour_val,))
        return of_eval

    def _f(self, params):
        """Performs the same operation as 'of', but takes only one argument that combines the state parameters and continuation parameter into one array. This enables '_f' to be used with the 'hessian' function

        Args:
            params (array): (len(_state_params) + 1,1) array of parameters. The first 'nstate_params' specify the  "state" of system, the last param[-1] is the continuation parameter

        Returns:
            of_eval (array): (1,) array containing objective function evaluated at 'params' minus 'contour_val'
        """
        of_eval = self.f(params[:-1], params[-1])
        return of_eval

    def f_gradient(self, params, continuation_param):
        """Calculates the gradient of specialized objective function with respect to '_state_params' and 'continuation_param', returning a vector

        Args:
            params (array): (len(_state_params),1) array of parameters specifying "state" of system
            continuation_param (float): the continuation parameter value

        Returns:
            of_gradient (array): (_state_params + 1, 1) vector containing the gradient evaluated at 'params' and 'continuation_param'
        """
        combined_params = np.empty(params.shape[0] + 1)
        combined_params[:-1] = params
        combined_params[-1] = continuation_param
        of_gradient = gradient(self._f, combined_params)
        return np.array(((of_gradient),))
                                                                                                                
            



