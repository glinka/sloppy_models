"""Defines customized errors for numerical schemes contained in this directory"""

class EvalError(ArithmeticError):
    pass

class IntegrationError(EvalError):
    pass

class ConvergenceError(ArithmeticError):
    pass

class PSAError(ArithmeticError):
    pass
