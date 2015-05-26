import numpy as np

class custom_kernel:
    """A single-function class used to evaluate the modified DMAPS kernel between two points as motivated by Lafone's thesis. That is :math:`W_{ij}=exp(\\frac{\|x_i - x_j\|^2}{\epsilon} - \\frac{(of(x_i) - of(x_j))^2}{\epsilon^2})`

    Attributes:
        _epsilon (float): the DMAPS parameter :math:`\epsilon` to be used in kernel evaluations
    """

    def __init__(self, epsilon):
        # set epsilon
        self._epsilon = epsilon

    def dmaps_of_kernel(self, pt1, pt2):
        """The function used to evaluate :math:`W_{ij}` between 'pt1' and 'pt2' with the prespecified value of :math:`\epsilon`

        Note:
            pt[:-1] contains the parameter vector, while pt[-1] contains the objective function evaluation at that parameter set
        """
        return np.exp(-np.power(np.linalg.norm(pt1[:-1] - pt2[:-1]), 2)/self._epsilon - np.power(pt1[-1] - pt2[-2], 2)/np.power(self._epsilon, 2))
