"""
.. module:: PseudoArclengthContinuation
   :platform: Unix, Windows, Mac
   :synopsis: Basic implementation of pseudo-arclength continuation algorithm
"""

import util_fns as uf
import numpy as np
import scipy.sparse.linalg as spla
import Newton

class PSA:
    """Basic pseudo-arclength continuation which requires an analytic expression for Jacobian
    
    Attributes:
        _f (function): function :math:`f(x, k)` from :math:`R^{n+1} \\rightarrow R^n` on which continuation will be performed, and in which :math:`k` is the continuation parameter
        _Df (array): the Jacobian of '_f', a function :math:`Df(x, k)` from :math:`R^{n+1} \\rightarrow R^{n(n+1)}`

    .. note::

    _Df is not a square matrix due to the continuation parameter

    >>> insert example code here
    """

    def __init__(self, f, Df):
        """Assigns function and Jacobian to private member variables

        Args:
            f (function): function along which to continue, accepting two arguments: the x-vector and scalar continuation parameter at which to evaluate the function
            Df (function): Jacobian of 'f', called in the same manner as 'f'
        """

        # set private variables
        self._f = f
        self._Df = Df

    def _Df_arclength(self, x, xprime):
        """Evaluates extended Jacobian which incorprates arclength parameter in bottom row
        
        .. note::

        The variable 'x' contains both the length-n vector specifying the system's state and the scalar 'k' specifying the parameter value. This is the true "arclength x" value used our continuation scheme. **It has shape (n+1,).**

        Args:
            x (array): the shape (n+1,) vector containing arclength continuation's x value
            xprime (array): value of arclength :math:`\\frac{dx}{ds}`

        Returns:
            Df_arclength (array): shape (n+1, n+1) array containing the evaluation of the extended, square Jacobian
        """
        # again, the number of state variables 'n' is one less than the length of x because we have appended the scalar parameter value
        # TODO: it would be clearer to define locally "k = x[n]...", but is there a speed difference?
        n = x.shape[0] - 1
        Df_arclength = np.empty((n+1, n+1))
        try:
            Df_arclength[:n,:] = self._Df(x[:n],x[n])
        except EvalError:
            raise
        Df_arclength[n,:] = xprime
        return Df_arclength

    def _f_arclength(self, x, xprev, xprime, ds):
        """Evaluates extended :math:`f_{ext} \\in R^{n+1}` which incorprates arclength parameters in final entry
        
        .. note::

        The variable 'x' contains both the length-n vector specifying the system's state and the scalar 'k' specifying the parameter value. This is the true "arclength x" value used our continuation scheme. **It has shape (n+1,).**
        
        Args:
            x (array): the shape (n+1,) vector containing arclength continuation's x value
            xprev (array): value of arclength x from previous point on branch
            xprime (array): value of arclength :math:`\\frac{dx}{ds}`
            ds (float): length of arclength step
        
        Returns:
            f (array): shape (n+1,) array containing the evaluation of :math:`f_{ext}`
        """
        
        # again, the number of state variables 'n' is one less than the length of x because we have appended the scalar parameter value
        n = x.shape[0] - 1
        f = np.empty(n+1)
        try:
            f[:n] = self._f(x[:n], x[n])
        except EvalError:
            raise
        f[n] = np.dot(x[:n] - xprev[:n], xprime[:n]) + (x[n] - xprev[n])*xprime[n]  - ds
        return f


    def find_branch(self, x0, k0, ds, nsteps, progress_bar=False, **kwargs):
        """Continues along the branch of values for which :math:`f(x,k)=0` via pseudo-arclength continuation

        Args:
            x0 (array): the initial x vector
            k0 (float): the initial parameter value
            ds (float): the arclength step size
            nsteps (int): the total number of arclength steps to take
            progress_bar (bool): whether or not to show a progress bar as iterations proceed

        Returns:
            Numpy array of dimension (nsteps, n+1), each row of which contains first the length-n value of x and then the scalar parameter value at which a point on the branch was found

        .. note::

        If :math:`f(x_0,k_0) \\neq 0`, this method automatically searches for an appropriate starting point via a Newton iteration at :math:`k=k_0`
        """
        # TODO: faster method than defining lambda fn?
        n = x0.shape[0]
        f_init = lambda x: self._f(x, k0)[:n]
        Df_init = lambda x: self._Df(x, k0)[:n,:n]
        # find initial point on branch
        newton_solver = Newton.Newton(f_init, Df_init)
        try:
            xstart = newton_solver.find_zero(x0, **kwargs)
        except EvalError:
            raise
        # append parameter value
        xstart = np.hstack((xstart, k0))
        # find initial slopes
        # pretend initial slopes are 0 in x, 1 in k
        tempslopes = np.zeros(n+1)
        tempslopes[n] = 1
        # note that tempslopes is also rhs of initial slope calc (see Auto notes)
        try:
            xprime = spla.gmres(self._Df_arclength(xstart, tempslopes), tempslopes)[0]
        except EvalError:
            raise
        # normalize
        xprime_start = xprime/np.linalg.norm(xprime)
        # update newton to new functions
        newton_solver = Newton.Newton(self._f_arclength, self._Df_arclength)
        halfnsteps = nsteps/2
        branch_pts = np.empty((2*halfnsteps + 1, n+1))
        branch_pts[-1] = np.copy(xstart)
        # take nsteps/2 forward and backward from the initial point
        # in case the inner loop over 'i' exits prematurely, keep track of how many were successfully obtained
        ncompleted_pts = halfnsteps
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # The error handling in this double loop is as follows:
        # The inner loop might raise an EvalError from the 'find_zero' function
        # which will be caught in the outer loop. If this happens, we must adjust
        # 'ncompleted_pts' accordingly, to reflect how many iterations were successful
        # before the error. Then, we continue in the opposite direction (k==1). Again,
        # an error could be raised, in which case we return whatever was
        # found to that point. Otherwise, return the partial branch from (k==0)
        # and the full branch from (k==1).
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for k in range(2):
            # flip ds to continue in both directions
            ds = -ds
            # move x back to "center" of branch
            x = np.copy(xstart)
            xprime = np.copy(xprime_start)
            try:
                for i in range(halfnsteps):
                    if progress_bar:
                        uf.progress_bar(k*halfnsteps + i + 1, nsteps)
                    # initial guess for next point on branch
                    x0 = x + xprime*ds
                    # save previous value for arclength eqn
                    xprev = np.copy(x)
                    # update parameter values for f and Df in newton solver
                    newton_solver.change_parameters([xprev, xprime, ds], [xprime])
                    x = newton_solver.find_zero(x0, **kwargs)
                    # use finite diff approx for xprime
                    xprime = (x - xprev)/ds
                    # normalize
                    xprime = xprime/np.linalg.norm(xprime)
                    branch_pts[k*ncompleted_pts + i] = np.copy(x)
            except EvalError:
                # continue from ncompleted_pts
                if k == 0:
                    ncompleted_pts = i
                    continue
                # k == 1, copy initial point from end of 'branch' and return whatever was successfully found
                else:
                    branch[ncompleted_pts + i] = branch[-1]
                    return branch[:ncompleted_pts + i + 1]
            else:
                ncompleted_pts = halfnsteps
        # could have encountered error when k==0, no error when k==1: adjust accordingly
        branch[ncompleted_pts + halfnsteps] = branch[-1]
        return branch[:ncompleted_pts + halfnsteps + 1]

