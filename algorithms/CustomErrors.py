"""Defines customized errors for numerical schemes contained in this directory"""

class EvalError(ArithmeticError):
    def __init__(self, msg):
        self.msg = msg

class IntegrationError(EvalError):
    def __init__(self, msg):
        EvalError.__init__(self, msg)

class InitialIntegrationError(IntegrationError):
    pass

class ConvergenceError(ArithmeticError):
    pass

class PSAError(ArithmeticError):
    def __init__(self, msg):
        self.msg = msg

class LSODA_Warning:
    pass
