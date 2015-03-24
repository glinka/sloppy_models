import numpy as np
import scipy.sparse.linalg as spla

class Newton:
    """Solves :math:`f(x) = 0` using Newton's method with an analytical Jacobian and scipy`s GMRES linear solver

    Attributes:
        _f: the function to find the zero of, :math:`f: R^n \\rightarrow R^n`
        _Df: the Jacobian of '_f', :math:`Df: R^n \\rightarrow R^{n*n}`

    >>> example code here
    """

    def __init__(self, f, Df, fargs=[], Dfargs=[]):
        """Assigns function and Jacobian to private member variables

        Args:
            f (function): function which accepts and returns a length-n numpy array
            Df (function): Jacobian function which accepts a length-n numpy array, returns an array of dimension (n, n)
            fargs (list): optional arguments that will be passed to f
            Dfargs (list): optional arguments that will be passed to Df
        """
        self._f = f
        self._Df = Df
        self._fargs = fargs
        self._Dfargs = Dfargs

    def find_zero(self, x0, abstol=1e-7, reltol=1e-7, maxiters=100000):
        """Attempts to find a zero of 'f' through Newton-GMRES, terminating when :math:`\\|F(x_k)\\| \\leq reltol*\\|F(x_0)\\| + abstol`

        Args:
            x0 (array): initial guess for solution
        
        Kwargs:
            abstol (float): absolute tolerance for Newton iteration 
            reltol (float): relative tolerance **both for Newton iteration and inner GMRES iteration**
            maxiters (int): maximum number of iterations to attempt

        .. note::

        scipy's implementation of GMRES exits after 20 iterations by default, and when either the absolute **or** relative errors are less than the input kwarg 'tol'

        """
        # copy x0 to avoid bizarre behavior
        x = np.copy(x0)
        iters = 0
        # need init_error for reltol
        init_error = np.linalg.norm(self._f(x0, *self._fargs))
        error = init_error
        totaltol = init_error*reltol + abstol

        while error > totaltol and iters < maxiters:
            # update with output of linear solver
            x = x + spla.gmres(self._Df(x, *self._Dfargs), -self._f(x, *self._fargs), tol=reltol)[0]
            error = np.linalg.norm(self._f(x, *self._fargs))
            iters = iters + 1
        if iters < maxiters:
            # converged, return
            return x
        else:
            # TODO: probably should raise some error
            print '******************************'
            print 'failed to converge within total tolerance:', totaltol
            print 'output error:', error
            print '******************************'
            return False

    def change_parameters(self, fargs, Dfargs):
        """Updates the optional arguments to both f and Df

        Args:
            fargs (list): new optional arguments to pass to f
            Dfargs (list): new optional arguments to pass to f
        """
        self._fargs = fargs
        self._Dfargs = Dfargs

    def change_fparameters(self, fargs):
        """Updates the optional arguments to f

        Args:
            fargs (list): new optional arguments to pass to f
        """
        self._fargs = fargs

    def change_Dfparameters(self, Dfargs):
        """Updates the optional arguments to Df

        Args:
            Dfargs (list): new optional arguments to pass to f
        """
        self._Dfargs = Dfargs
