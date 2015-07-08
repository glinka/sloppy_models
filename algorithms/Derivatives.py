"""Numerical calculation of the Hessian of an objective function through centered finite differences"""

import CustomErrors
import numpy as np

def hessian(f, x, h=1e-8):
    """Evaluates a centered finite-difference approximation of the *Hessian* of 'f' at 'x' using stepsize 'h'

    Args:
        f (function): function :math:`R^n \\rightarrow R` for which we approximate the Hessian
        x (array): shape (n,) vector at which to approximate the Hessian
        h (float): stepsize to be used in approximation

    >>> f = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[0]*x[1] + x[1]*x[2] # x**2 + y**2 + z**2 + x*y + y*z
    >>> x = np.zeros(3)
    >>> print hessian(f, x)
    """
    n = x.shape[0]
    hessian = np.empty((n,n))
    ioffset = np.zeros(n)
    for i in range(n):
        # set offset in proper position of a unit vector but first unset previous changes
        ioffset[i-1] = 0
        ioffset[i] = h
        hessian[i,i] = (f(x+ioffset) - 2*f(x) + f(x-ioffset))/(h*h) # centered finite diff approx to df/dxdx
        for j in range(i+1,n):
            # set j offset
            joffset = np.zeros(n)
            joffset[j] = h
            hessian[i,j] = (f(x + ioffset + joffset) - f(x + ioffset - joffset) - f(x - ioffset + joffset) + f(x - ioffset - joffset))/(4*h*h) # centered finite diff approx to df/dxdy
            hessian[j,i] = hessian[i,j]
    return hessian

def gradient(f, x, h=1e-4):
    """Evaluates a centered finite-difference approximation of the *gradient* of 'f' at 'x' using stepsize 'h'

    Args:
        f (function): function :math:`R^n \\rightarrow R` for which we approximate the gradient
        x (array): shape (n,) vector at which to approximate the gradient
        h (float): stepsize to be used in approximation

    .. note:
        This method is **terribly sensitive to 'h'** in the case of the Michaelis Menten system

    >>> f = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[0]*x[1] + x[1]*x[2] # x**2 + y**2 + z**2 + x*y + y*z
    >>> x = np.ones(3)
    >>> print gradient(f, x)
    """
    n = x.shape[0]
    gradient = np.empty(n)
    ioffset = np.zeros(n)
    for i in range(n):
        # set offset in proper position of a unit vector but first unset previous changes
        ioffset[i-1] = 0
        ioffset[i] = h
        try:
            gradient[i] = (f(x+ioffset) - f(x-ioffset))/(2*h) # centered finite diff approx to df/dx
        except CustomErrors.EvalError:
            raise
    return gradient
