"""Header for Z_Model.pyx"""
cimport numpy as np
cimport cython


cdef class Z_Model:

    cdef double _a, _b, _lam, _eps
    cdef object _integrator

    # def __init__(self, np.ndarray[np.float64_t] params)
    cpdef np.ndarray[np.float64_t] _rhs(self, float t, np.ndarray[np.float64_t] x)

    cpdef void change_parameters(self, np.ndarray[np.float64_t] new_params)

    cpdef get_trajectory_quadratic(self, np.ndarray[np.float64_t] x0, np.ndarray[np.float64_t] times)
