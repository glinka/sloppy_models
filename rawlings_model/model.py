"""Implementation of the chemical reaction system A -> <- B, B -> C as detailed on pg. 218 of Chemical Reactor Analysis and Design Fundamentals by Rawlings and Ekerdt (2nd printing)

Contact: holiday@alexanderholiday.com
"""

import dill
from os.path import isfile
import numpy as np
import scipy.integrate as spint
import sympy
from algorithms.ObjectiveFunction import ObjectiveFunction

class Rawlings_Model:
    """Models mechanism A -> <- B, B -> C, with rate constants k1, kinv, and k2 respectively. Includes various functions to help in analysis of parameter space


    Attributes:
        _Cs (array): timecourse of concentration of species C over time, assumed to be the only measured feature of the system
    """

    def __init__(self, times, A0, k1, kinv, k2, using_sympy=False):
        """Generates baseline/true dataset of C concentration over time based on the provided parameters

        Args:
            times (array): times at which to evaluate C throughout calculations (should never be changed after being set)
            A0 (float): initial value of concentratin of A (should never be changed after being set)
            k1 (float): true value of parameter k1
            kinv (float): true value of parameter kinv
            k2 (float): true value of parameter k2
            using_sympy (bool): whether or not to generate gradient and hessian functions of the ob. fn. from symbolic representations
        """
        self._A0 = A0
        self._TIMES = times
        self._Cs = self.gen_timecourse(k1, kinv, k2)

        self.sympy_lsq_of_f = None
        self.sympy_lsq_of_gradient = None
        self.sympy_lsq_of_hessian = None
        if using_sympy:
            # set up sympy objective function
            k1, kinv, k2 = sympy.symbols('k1,kinv,k2')
            alpha = (k1+kinv+k2 + ((k1+kinv+k2)**2 - 4*k1*k2)**0.5)/2
            beta =  (k1+kinv+k2 - ((k1+kinv+k2)**2 - 4*k1*k2)**0.5)/2
            sympy_times = sympy.Matrix(times).transpose()
            Cs_true = sympy.Matrix(self._Cs).transpose()
            # Cs = sympy.zeros(times.shape[0], 1)
            ones = sympy.ones(1, times.shape[0]) # for adding the time-independent term at each time
            Cs = self._A0*(ones*k1*k2/(alpha*beta) + (-alpha*sympy_times).applyfunc(sympy.exp)*k1*k2/(alpha*(alpha-beta)) - (-beta*sympy_times).applyfunc(sympy.exp)*k1*k2/(beta*(alpha-beta)))
            # # if we've already constructed and compiled the functions from sympy, just reload
            # # jk doesn't work
            # symbolic_lsq_of = None
            # if isfile("./data/symbolic-of.dill"):
            #     symbolic_lsq_of = dill.load(open("./data/symbolic-of.dill", 'r'))
            # else:
                # symbolic_lsq_of = ObjectiveFunction(sum((Cs - Cs_true).applyfunc(lambda x:x*x)), [k1, kinv, k2])
                # dill.dump(symbolic_lsq_of, open("./data/symbolic-of.dill", 'w'))
            symbolic_lsq_of = ObjectiveFunction(sum((Cs - Cs_true).applyfunc(lambda x:x*x)), [k1, kinv, k2])
            self.sympy_lsq_of_f = symbolic_lsq_of.f
            self.sympy_lsq_of_gradient = symbolic_lsq_of.gradient
            self.sympy_lsq_of_hessian = symbolic_lsq_of.hessian
            

    def gen_full_timecourse(self, k1, kinv, k2):
        """Generates timecourse of concentrations of A, B, and C at 'times' based on the initial conditions A0 = 'A0', B0 = 0, C0 = 0 and parameters given in self._constants

        Args:
            k1 (float): value of parameter k1
            kinv (float): value of parameter kinv
            k2 (float): value of parameter k2

        Returns:
            concs (array): (times.shape[0], 3) shape array of concentrations of A, B and C in the columns
        """
        alpha = 0.5*(k1+kinv+k2 + np.sqrt(np.power(k1+kinv+k2, 2) - 4*k1*k2))
        beta  = 0.5*(k1+kinv+k2 - np.sqrt(np.power(k1+kinv+k2, 2) - 4*k1*k2))
        As = self._A0*(k1*(alpha-k2)*np.exp(-alpha*self._TIMES)/(alpha*(alpha-beta)) + k1*(k2-beta)*np.exp(-beta*self._TIMES)/(beta*(alpha-beta)))
        Bs = self._A0*(-k1*np.exp(-alpha*self._TIMES)/(alpha-beta) + k1*np.exp(-beta*self._TIMES)/(alpha-beta))
        Cs = self._A0*(k1*k2/(alpha*beta) + np.exp(-alpha*self._TIMES)*k1*k2/(alpha*(alpha-beta)) - np.exp(-beta*self._TIMES)*k1*k2/(beta*(alpha-beta)))
        return np.array((As,Bs,Cs)).T


    def gen_timecourse(self, k1, kinv, k2):
        """Generates timecourse of concentrations of C at 'times' based on the initial conditions A0 = 'A0', B0 = 0, C0 = 0 and parameters given in self._constants

        Args:
            k1 (float): value of parameter k1
            kinv (float): value of parameter kinv
            k2 (float): value of parameter k2

        Returns:
            Cs (array): value of C at 'times'
        """
        alpha = 0.5*(k1+kinv+k2 + np.sqrt(np.power(k1+kinv+k2, 2) - 4*k1*k2))
        beta  = 0.5*(k1+kinv+k2 - np.sqrt(np.power(k1+kinv+k2, 2) - 4*k1*k2))
        Cs = self._A0*(k1*k2/(alpha*beta) + np.exp(-alpha*self._TIMES)*k1*k2/(alpha*(alpha-beta)) - np.exp(-beta*self._TIMES)*k1*k2/(beta*(alpha-beta)))
        return Cs

    def lsq_of(self, k1, kinv, k2):
        """Returns least squares obj. fn. evaluated at the test parameters 'k1', 'kinv', and 'k2', i.e. returns :math:`\| \hat{C} - C \|^2` where :math:`\hat{C}` is the true value of C, a vector containing its timecourse.

        Args:
            k1 (float): value of parameter k1
            kinv (float): value of parameter kinv
            k2 (float): value of parameter k2

        Returns:
            lsq_of_eval: least squares obj. fn. evaluated at the test parameters 'k1', 'kinv', and 'k2', i.e. returns :math:`\| \hat{C} - C \|^2` where :math:`\hat{C}` is the true value of C, a vector containing its timecourse.
        """
        lsq_of_eval = np.power(np.linalg.norm(self.gen_timecourse(k1, kinv, k2) - self._Cs), 2)
        return lsq_of_eval

    def lsq_f(self, ks):
        """Returns vector of differences between predicted and true concentrations of C, i.e. returns :math:`f(k_1, k_{inv}, k_2)` where the objective function is given by $\| f(k_1, k_{inv}, k_2) \|$.

        Args:
            ks (array): (3,1) array of (k1, kinv, k2)

        Returns:
            f_eval (array): (ntimes, 1) array of difference between predicted and actual C concentrations with time
        """
        return self.gen_timecourse(*ks) - self._Cs
                                                                                                      



