import CustomErrors
import numpy as np
import scipy.sparse.linalg as spla
import BisectionMethod

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

    def find_zero(self, x0, abstol=1e-11, reltol=1e-11, maxiters=10, damping=0, bisection_on_maxiter=False):
        """Attempts to find a zero of 'f' through Newton-GMRES, terminating when :math:`\\|F(x_k)\\| \\leq reltol*\\|F(x_0)\\| + abstol`

        Args:
            x0 (array): initial guess for solution
        
        Kwargs:
            abstol (float): absolute tolerance for Newton iteration 
            reltol (float): relative tolerance **both for Newton iteration and inner GMRES iteration**
            maxiters (int): maximum number of iterations to attempt
            damping (float): damping factor for newton update, representing the scaling of the update, i.e. :math:`x^{k+1} = x^k + (1 - \kappa) dx^k`, where :math:`\kappa` is the damping factor. Thus 'damping=0' corresponds to no damping, while 'damping=1' corresponds to no update.                                                                                                        

        .. note::
            scipy's implementation of GMRES exits after 20 iterations by default, and when either the absolute **or** relative errors are less than the input kwarg 'tol'

        """
        # copy x0 to avoid bizarre behavior due to lack of understanding of numpy's assignment rules
        x = np.copy(x0)
        iters = 0
        # need init_error for reltol
        try:
            init_error = np.linalg.norm(self._f(x0, *self._fargs))
        except CustomErrors.EvalError:
            raise CustomErrors.EvalError('Could not evaluate initial error')
        error = init_error
        totaltol = init_error*reltol + abstol

        while error > totaltol and iters < maxiters:
            # update with output of linear solver
            try:
                dx, convergence_info = spla.gmres(self._Df(x, *self._Dfargs), -self._f(x, *self._fargs), tol=reltol)
            except CustomErrors.EvalError:
                raise CustomErrors.EvalError('Could not evaluate f or Df at newton iteration ' + str(iters + 1))
            if np.any(np.isinf(dx)) or np.any(np.isnan(dx)):
                raise CustomErrors.EvalError('Solution at newton iteration ' + str(iters + 1) + ' had NaN or inf')
            else:
                x = x + (1 - damping)*dx
                error = np.linalg.norm(self._f(x, *self._fargs))

            iters = iters + 1
        if iters < maxiters:
            # converged, return
            return x
        elif bisection_on_maxiter and x.shape[0] is 1:
            # go into a bisection method for the one-dimensional problem
            x = Bisection_Method.find_zero(self._f, x, abstol=abstol)
            if np.linalg.norm(self._f(x, *self._fargs)) < totaltol:
                return x
            else:
                raise CustomErrors.ConvergenceError
        else:
            # exit with error
            raise CustomErrors.ConvergenceError

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
