import algorithms.CustomErrors as CustomErrors
import numpy as np
import scipy.integrate as spint

class Integrator:
    """A class for quick integration of an ODE using the specified algorithm. Functionality resembles scipy.odeint() but allows the user to tailor the integration algorithm to their specific system. Useful when many integrations must be performed

    Attributes:
        _f (function): right-hand-side of the ODE, **callable as f(t, x)** where 't' is a scalar, and 'x' is a length-N state variable
        _integrator (scipy integrator): internal integrator: the return value of scipy.ode()
    """

    def __init__(f, algorithm='lsoda'):
        """Sets up integrator
        Args:
            f (function): right-hand-side of the ODE, **callable as f(t, x)** where 't' is a scalar, and 'x' is a length-N state variable
            algorithm (str): any of 'vode', 'zvode', 'lsoda' (default), 'dopri5' or 'dopri853', i.e. the options accepted by scipy.ode()
        """
        self._f = f
        self._integrator = spint.integrator(algorithm)

    def integrate(self, x0, times):
        """Integrates 'f' over 'times', returning its value at each points in 'times'. If integration is unsuccessful, raises an IntegrationError from CustomErrors

        Args:
            x0 (array): initial value of system, assumed given for t=0
            times (array): times at which to save value of state variable

        Returns:
            trajectory (array): (times.shape[0], x0.shape[0]) array in which columns contain the trajectory of a given state variable
        """
        self._integrator.set_initial_value(x0)
        trajectory = np.empty((times.shape[0], x0.shape[0]))
        for i, t in enumerate(times):
            trajectory[i] = self._integrator.integrate(t)
            if not self._integrator.successful():
                raise CustomErrors.IntegrationError
        return trajectory

def integrate(f, x0, times, algorithm='lsoda'):
    """Integrates 'f' over 'times' with initial conditions of 'x0', returning its value at each points in 'times'. If integration is unsuccessful, raises an IntegrationError from CustomErrors. Useful for quickly integrating a function a couple times. Otherwise use the class 'Integrator' to avoid re-initializing the scipy integrator.

    Args:
        x0 (array): initial value of system, assumed given for t=0
        times (array): times at which to save value of state variable

    Returns:
        trajectory (array): (times.shape[0], x0.shape[0]) array in which columns contain the trajectory of a given state variable
    """
    integrator = spint.integrator(algorithm)
    integrator.set_initial_value(x0)
    trajectory = np.empty((times.shape[0], x0.shape[0]))
    for i, t in enumerate(times):
        trajectory[i] = integrator.integrate(t)
        if not integrator.successful():
            raise CustomErrors.IntegrationError
    return trajectory
    
