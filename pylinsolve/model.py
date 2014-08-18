""" Contains the main model class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import collections

import sympy
from sympy import Function
from sympy.utilities.lambdify import implemented_function

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


def _add_var_to_context(context, var):
    """ Adds the var and the variable's series accessor function
        to the list of known symbols.
    """
    name = var.name
    f_name = Variable.series_name(name)
    var_acc = implemented_function(Function(f_name),
                                   lambda(t): var.model.get_at(name, t))

    context[name] = var.symbol
    context[f_name] = var_acc


def _add_param_to_context(context, param):
    """ Adds the paramter to the list of known symbols.
    """
    context[param.name] = param.symbol


class Model(object):
    """ This is the main Model class.  Variables, parameters, and
        equations are defined through this class.
    """

    def __init__(self):
        self._vars = collections.OrderedDict()
        self._params = collections.OrderedDict()
        self._equations = collections.OrderedDict()
        self._solutions = list()
        self._local_context = dict()
        self._var_default = None
        self._param_initial = None

    def set_var_default(self, default):
        """ Sets the general default value for all variables. """
        self._var_default = default

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
        default = default or self._var_default
        if name in self._vars or name in self._params:
            raise DuplicateNameError('Name already in use: ' + name)
        var = Variable(name, desc=desc, default=default, symbol=symbol)
        self._vars[name] = var

        _add_var_to_context(self._local_context, var)
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

    def set_param_initial(self, initial):
        """ Sets the default initial parameter value for all Parameters """
        self._param_initial = initial

    def param(self, name, desc=None, initial=None):
        """ Creates a parameter for use within the model.

            Returns: a Parameter
        """
        initial = initial or self._param_initial
        if name in self._vars or name in self._params:
            raise DuplicateNameError('Name already in use: ' + name)
        param = Parameter(name, desc, initial)
        self._params[name] = param
        _add_param_to_context(self._local_context, param)
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
        eqn.parse(self._local_context)

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
        return self._solutions[-1]

    def get_at(self, name, iteration):
        """ Returns the value for a variable at an iteration.
            The value for iter may be positive or negative:
                If >= 0, absolute position
                    0 is the very first iteration.  Note that
                    iterations are available only AFTER the
                    solver has returned success.
                If < 0, relative position
                    -1 is the iteration previous to the current
                    iteration.

            Arguments:
                name:
                iteration:

            Returns:
                The value of the variable for that iteration.
        """
        return self._solutions[iteration][name]

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
