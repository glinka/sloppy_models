"""
.. module:: Bisection_Method
   :platform: Unix, Windows, Mac
   :synopsis: Basic implementation of one-dimensional bisection method
"""
import CustomErrors
import numpy as np
from Derivatives import gradient

def find_zero(f, x0, x1=None, abstol=1e-9, find_bounds=False, maxiters=1000):
    """Searches for zero of 'f' in 'bounding_interval' using the bisection method, exiting when :math:`\\|f(x_k)\\| \\leq abstol`

    Args:
        f (function): function taking one, one-dimensional argument 'x', callable as 'f(x)'
        x0 (float): some bound of f, either upper or lower
        x1 (float): some bound of f, 'x0' and 'x1' should define an interval bounding the zero of 'f'. If x1 is left unset, find_bounds will be assumed 'True' and a bounding interval will be searched for
            .. note::
            Ideally :math:`\textrm{sign}(f(x_l)) = - \textrm{sign}(f(x_u))`, i.e. the sign of 'f' flips on either side of the interval. If not, the user can set 'find_bounds' to 'True' and an automated effort will be made to locate this region.
        abstol (float): absolute error tolerance. Upon finding an 'x' value such that :math:`\\|f(x)\\| \\leq abstol`, the function exits
        find_bounds (bool): whether or not to search for a bounding interval
        maxiters (int): maximum number of bisection iterations to try

    Returns:
        x (float): the converged-to value of 'x', a zero of 'f'
    """
    if ((x1 is not None) and (np.sign(f(x0)) == np.sign(f(x1)))) or (x1 is None):
        # do not have bounding interval, try to with one
        x1 = _find_bounding_interval(f, x0)

    # have interval, identify lower and upper bounds
    if f(x0) < 0:
        x_lower = x0; x_upper = x1
    else:
        x_lower = x1; x_upper = x0
    # determine initial error
    if np.abs(f(x0)) < np.abs(f(x1)):
        error = np.abs(f(x0))
        x = x0
    else:
        error = np.abs(f(x1))
        x = x1
    # bisect away
    iters = 0
    while iters < maxiters and error > abstol:
        # create new point in center of interval and evaluate 'f' at it
        x_mid = x_lower + (x_upper - x_lower)/2.0
        feval = f(x_mid)
        # determine if point is new lower or upper bound
        if np.sign(feval) == 1:
            x_upper = x_mid
        else:
            x_lower = x_mid
        # check if new point reduces error
        if np.abs(feval) < error:
            error = np.abs(feval)
            x = x_mid
        iters = iters + 1
    if iters < maxiters:
        # converged, return
        return x
    else:
        # exit with error
        print 'bisection failed :('
        raise CustomErrors.ConvergenceError

def _find_bounding_interval(f, x, dx=1.0, maxiters=1000):
    """Attempts to find interval of real line on which 'f' changes sign, and thus in which a zero of 'f' is contained"""
    #TODO use derivative information in combination with the error to intelligently search for the interval, dis just stupid right now
    feval = f(x)
    init_sign = np.sign(feval)
    search_direction = -np.sign(gradient(f, np.array((x,))))*init_sign
    iters = 0
    x_new = np.copy(x)
    while np.sign(feval) == init_sign and iters < maxiters:
        x_new = x_new + search_direction*dx
        feval = f(x_new)
        iters = iters + 1
    if iters < maxiters:
        # converged, return
        return x_new
    else:
        # exit with error
        print 'bounding failed :('
        raise CustomErrors.ConvergenceError
        
if __name__=='__main__':
    f = lambda x: x*x - 1
    print find_zero(f, -1.5), find_zero(f, -0.5), find_zero(f, 0.5), find_zero(f, 1.5)
