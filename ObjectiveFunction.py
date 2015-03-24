import numpy as np
import sympy
import sympy.utilities.autowrap
import dmaps

class ObjectiveFunction:
    """A class representing an objective function and its corresponding gradient and Hessian

    Attributes:
    _f: underlying objective function being represented by this class as a binary lambda function
    _x: arguments to function _f represented by sympy symbols
    _gradient: gradient of _f  represented as a binary lambda function
    _hessian: hessian of _f represented as a binary lambda function

    >>> x,y = symbols('x,y')
    >>> of = ObjectiveFunction(x*x+y, [x,y])
    >>> print of.f([1,1]), of.gradient([1,1]), of.hessian([1,1])
    """

    def __init__(self, f, x):
        """Converts sympy input objective function to a compiled binary lambda function and creates corresponding gradient and Hessian

        Args:
        f (sympy function): symbolic representation of objectice function
        x (sympy symbols): list of symbols which constitute arguments to f
        .. note::
        'f' must be a scalar-valued function, but may certainly accept an arbitrary number of arguments
        """

        # set object's private members
        self._x = x
        # after symbolically differentiating, convert functions to compiled lambda functions with 'ufuncify' for increased performance with function evaluation
        self._f = sympy.utilities.autowrap.ufuncify(self._x, f)
        self._gradient = [sympy.diff(f, i) for i in self._x]
        self._hessian = [[sympy.diff(i, j) for j in self._x] for i in self._gradient]
        self._gradient = [sympy.utilities.autowrap.ufuncify(self._x, i) for i in self._gradient]
        self._hessian = [[sympy.utilities.autowrap.ufuncify(self._x, i) for i in j] for j in self._hessian]

    def f(self, x):
        """Evaluate objective function
        Returns:
        Objective function '_f' evaluated at 'x'
        """
        return np.array(self._f(*x))

    def gradient(self, x):
        """Evaluate objective function's gradient
        Returns:
        Gradient '_gradient' evaluated at 'x', a vector
        """
        return np.array([f(*x) for f in self._gradient])

    def hessian(self, x):
        """Evaluate objective function's Hessian
        Returns:
        Gradient '_hessian' evaluated at 'x', a matrix
        """
        return np.array([[f(*x) for f in row] for row in self._hessian])


