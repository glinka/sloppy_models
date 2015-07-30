"""Defines customized errors for numerical schemes contained in this directory"""

class EvalError(ArithmeticError):
    pass

class IntegrationError(EvalError):
    pass

class InitialIntegrationError(IntegrationError):
    pass

class ConvergenceError(ArithmeticError):
    pass

class PSAError(ArithmeticError):
    pass

class LSODA_Warning:
    pass
