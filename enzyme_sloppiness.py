import numpy as np
import scipy.integrate as spint
from sympy import Function, dsolve, Eq, Derivative, symbols

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EnzymeOF:
    """Represents the objective function created by the reactions described by Prof. Zagaris"""

    def __init__(self, data, times):
        """Completely specifies the objective fn. value"""
        self._data = data
        self._times = times

    def of_t(self, params):
        """Returns value of *transformed* least-squares objective fn. used to fit concentration of P"""
        # as these initial values seems set by eqns (8) and (9), do not leave as params but set here within fn.
        # initial concentrations used in integration
        # params = (K, V, sigma, epsilon, kappa)
        # S0 = K*sigma gives
        S0 = params[0]*params[2]
        C0 = 0
        P0 = 0
        Cs0 = np.array((S0, C0, P0))
        # pack up enzyme parameters to pass
n        enzyme_profile = spint.odeint(_enzyme_rhs, Cs0, self._times, args=tuple(params))
        return np.sum(np.power(enzyme_profile[:,2] - self._data[:,2], 2))

    def of(self, params):
        """Returns value of *original* least-squares objective fn. used to fit concentration of P""" 
        # transform params from (kinv, k1, k2, St, Et) -> (K, V, sigma, epsilon, kappa)
        transformed_params = transform_params(params)
        return self.of_t(transformed_params)

############################################################
############################################################
############################################################
############################################################
############################################################

def symbolic_enzyme_integrationparams(K, V, sigma, epsilon, kappa):
    """Attempts but ultimately fails to symbolically integrate the enzyme kinetic odes"""
    S, C, P = symbols('S,C,P', function=True)
    t = symbols('t')
    odes = (
        Eq(S(t).diff(t), (kappa + 1)*V*(-S(t) + C(t)*S(t)/(epsilon*(sigma + 1)*K) + C(t)*kappa/(epsilon*(sigma + 1)*(kappa + 1)))/K),
        Eq(C(t).diff(t), (kappa + 1)*V*(S(t) - C(t)*S(t)/(epsilon*(sigma + 1)*K) - C(t)/(epsilon*(sigma + 1)))/K),
        Eq(P(t).diff(t), V*C(t)/(epsilon*(sigma + 1)*K))
    )
    print odes
    print dsolve(odes)

def transform_params(params):
    """Transforms params from (kinv, k1, k2, St, Et) -> (K, V, sigma, epsilon, kappa)"""
    kinv = params[0]
    k1 = params[1]
    k2 = params[2]
    St = params[3]
    Et = params[4]
    K = (kinv + k2)/k1
    V = k2*Et
    sigma = St/K
    kappa = kinv/k2
    epsilon = Et/(St + K)
    return np.array((K, V, sigma, epsilon, kappa))

def _enzyme_rhs(Cs, t, K, V, sigma, epsilon, kappa):
        """Function passed to scipy integrate routine to find concentration profile

        Args:
        Cs (array): shape (3,) array containing concentrations of (S, C, P)
        t (float): time
        K, V, sigma, epsilon, kappa (floats): model params
        """
        S = Cs[0]
        C = Cs[1]
        P = Cs[2]
        Sprime = (kappa + 1)*V*(-S + C*S/(epsilon*(sigma + 1)*K) + C*kappa/(epsilon*(sigma + 1)*(kappa + 1)))/K
        Cprime = (kappa + 1)*V*(S - C*S/(epsilon*(sigma + 1)*K) - C/(epsilon*(sigma + 1)))/K 
        Pprime = V*C/(epsilon*(sigma + 1)*K)
        return np.array((Sprime, Cprime, Pprime))

def hessian(f, x, h=1e-4):
    """Evaluates a centered finite-difference approximation of the Hessian of 'f' at 'x' using stepsize 'h'"""
    n = x.shape[0]
    hessian = np.empty((n,n))
    ioffset = np.zeros(n)
    for i in range(n):
        # set offset in proper position of a unit vector
        ioffset[i-1] = 0
        ioffset[i] = h
        # centered finite diff approx to df/dxdx
        hessian[i,i] = (f(x+ioffset) - 2*f(x) + f(x-ioffset))/(h*h)
        for j in range(i+1,n):
            # set j offset
            joffset = np.zeros(n)
            joffset[j] = h
            # centered finite diff approx to df/dxdy
            hessian[i,j] = (f(x + ioffset + joffset) - f(x + ioffset - joffset) - f(x - ioffset + joffset) + f(x - ioffset - joffset))/(4*h*h)
            hessian[j,i] = hessian[i,j]
    return hessian

def check_sloppiness():
    """Checks for sloppiness in the model by printing the Hessian's eigenvalues when evaluated at the minimum of least-squares objective fn."""
    # params = (K, V, sigma, epsilon, kappa)
    # as per suggestions in paper
    params = np.array((1.0, 1.0, 1.0, 1e-2, 10.0))
    # S0 = K*sigma gives
    S0 = params[0]*params[2]
    C0 = 0.0
    P0 = 0.0
    Cs0 = np.array((S0, C0, P0))
    # tscale = (sigma + 1)*K/V
    tscale = (params[2] + 1)*params[0]/params[1]
    npts = 20
    times = tscale*np.linspace(1,npts,npts)/5.0
    data = spint.odeint(_enzyme_rhs, Cs0, times, args=tuple(params))
    # visualize data
    # plt.hold(True)
    # for i in range(3):
    #     plt.plot(times, data[:,i])
    # plt.show()
    enzyme_of = EnzymeOF(data, times)

    # testing stepsize's effect on hessian approx
    # # hvals = np.logspace(-6,-3, nhvals)
    # hvals = np.linspace(1e-4, 7e-3, nhvals)
    # eigs = np.empty((nhvals, 5))
    # for i in range(nhvals):
    #     hessian_eval = hessian(enzyme_of.of_t, params, h=hvals[i])
    #     eigs[i] = np.sort(np.linalg.eigvals(hessian_eval))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for i in range(5):
    #     ax.cla()
    #     ax.plot(hvals, eigs[:,i])
    #     ax.set_xscale('log')
    #     plt.savefig('./figs/hessian/eig' + str(i) + '.png')

    # use transformed ob. fn. here, of_t
    hessian_eval = hessian(enzyme_of.of_t, params, h=0.007)
    print 'Results for transformed problem:\n'
    print 'eigs of 3x3 hessian, hopefully not sloppy:\n', np.linalg.eigvals(hessian_eval[:3,:3])

    # plot the 3d eigenvectors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    eigvals, eigvects = np.linalg.eig(hessian_eval[:3,:3])
    for i in range(3):
        xyz = np.vstack((np.zeros(3), eigvects[:,i]/np.linalg.norm(eigvects[:,i])))
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2])
    plt.show()
        
    print 'eigs of full 5x5 hessian, hopefully has two sloppy directions:\n', np.linalg.eigvals(hessian_eval)
    # use original hessian
    hessian_eval = hessian(enzyme_of.of, transform_params(params), h=0.007)
    print '\nResults for original problem:\n'
    print 'eigs of 3x3 hessian, hopefully not sloppy:\n', np.linalg.eigvals(hessian_eval[:3,:3])
    print 'eigs of full 5x5 hessian, hopefully has two sloppy directions:\n', np.linalg.eigvals(hessian_eval)

if __name__=='__main__':
    # test_linear_2eq_order2()
    # params = np.array((1.0, 1.0, 1.0, 1e-2, 10.0))
    # symbolic_enzyme_integrationparams(*params)
    check_sloppiness()
