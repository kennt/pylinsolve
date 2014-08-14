""" Contains the main model class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import collections

import sympy

from pylinsolve.equation import Equation
from pylinsolve.parameter import Parameter
from pylinsolve.variable import Variable


class DuplicateNameError(ValueError):
    """ Exception: Duplicate name detected, name already in use. """
    def __init__(self, text):
        super(DuplicateNameError, self).__init__()
        self.text = text

    def __str__(self):
        return self.text


class SolutionNotFoundError(Exception):
    """ Exception: The solver could not converge on a solution. """
    def __init__(self, text):
        super(SolutionNotFoundError, self).__init__()
        self.text = text

    def __str__(self):
        return self.text


class Model(object):
    """ This is the main Model class.  Variables, parameters, and
        equations are defined through this class.
    """

    def __init__(self):
        self._vars = collections.OrderedDict()
        self._params = collections.OrderedDict()
        self._equations = collections.OrderedDict()
        self._solutions = list()

    def var(self, name, desc=None, default=None, symbol=None):
        """ Creates a variable for use within the model.

            Arguments:
                name: The name of the variable.  This is the symbol
                    used in the equations.
                desc: A longer description of the variable.
                default: The default value of the variable, if the
                    value is not set.
                symbol: The sympy Symbol instance associated
                    with this variable. If not specifed, a
                    symbol with the name is created.

            Returns: a Variable

            Raises:
                DuplicateNameError:
        """
        if name in self._vars or name in self._params:
            raise DuplicateNameError('Name already in use: ' + name)
        var = Variable(name, desc=desc, default=default, symbol=symbol)
        self._vars[name] = var
        return var

    def vars(self, names):
        """ Creates multiple variables for use within the model.

            Arguments:
                names:

            Returns: a list of Variables that were created.

            Raises:
                DuplicateNameError
        """
        # Perform the symbol expansion and then add the names
        vars_created = []
        for sym in sympy.symbols(names):
            vars_created.append(self.var(sym.name, symbol=sym))
        return vars_created

    def param(self, name, desc=None, initial=None):
        """ Creates a parameter for use within the model.

            Returns: a Parameter
        """
        if name in self._vars or name in self._params:
            raise DuplicateNameError('Name already in use: ' + name)
        param = Parameter(name, desc, initial)
        self._params[name] = param
        return param

    def params(self, name):
        """ Creates multiple parameters for use within the model.

            Arguments:
                names:

            Returns: a list of parameters that were created

            Raises:
                DuplicateNameError
        """
        pass

    def add(self, equation, desc=None):
        """ Adds an equation to the model.

            Arguments:
                equation: A string containing the equation.  There
                    cannot be any free symbols in the equation, all
                    non-numeric symbols must have been defined as a
                    parameter or a variable.
                desc: A description of the equation

            Returns: an Equation
        """
        eqn = Equation(equation, desc=desc)
        eqn.model = self
        eqn.parse()

    def solve(self, iterations=10, until=None, threshold=0.001,
              relexation=None):
        """ Runs the solver.

            The solver will try to find a solution until one of the
            conditions is reached:
                (1) the iterations limit is reached
                (2) the until condition returns True

            Arguments:
                iterations: The number of iterations to run until
                    a solution is reached. (Default: 10)
                until: This is a function that determines whether or
                    not a solution is reached.  (Default: residual error)
                threshold: If using the default end condition, this is the
                    threshold that the residuals must be less than.
                relaxation: Set this to use SOR rather than pure
                    Gauss-Seidel iteration. (Default: 1.0, Gauss-Seidel)

            Raises:
                SolutionNotFoundError:
        """
        pass

    def solution(self):
        """ Returns the solution for the latest iteration.
        """
        pass

    def variables(self):
        """ Returns the dict of variables
        """
        return self._vars

    def parameters(self):
        """ Returns the dict of parameters
        """
        return self._params

    def equations(self):
        """ Returns the list of equations
        """
        return self._equations
