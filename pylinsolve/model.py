""" Contains the main model class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import collections

from sympy import Symbol
from sympy import Function

from pylinsolve.equation import Equation, EquationError
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
    context[var.name] = var


def _add_param_to_context(context, param):
    """ Adds the paramter to the list of known symbols.
    """
    context[param.name] = param


class _SeriesAccessor(Function):
    """ Implements a sympy function to access the variable's values
        from previous iterations.
    """
    nargs = 2

    @classmethod
    def eval(cls, *arg):
        if not isinstance(arg[0], Variable):
            raise EquationError('not-a-variable',
                                str(arg[0]),
                                'Cannot access a series for a non-variable')

        if arg[0].model is None:
            raise EquationError('no-model',
                                arg[0].name,
                                'Variable must belong to a model')
        return arg[0].model.get_at(arg[0], arg[1])


def _add_series_accessor(context):
    """ Adds the function to access values from the previous iteration.
    """
    context['_series_acc'] = _SeriesAccessor
    context['_iter'] = Symbol('_iter')


class Model(object):
    """ This is the main Model class.  Variables, parameters, and
        equations are defined through this class.

        Attributes:
            variables:
            parameters:
            equations:
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.variables = collections.OrderedDict()
        self.parameters = collections.OrderedDict()
        self.equations = list()

        self._private_parameters = collections.OrderedDict()
        self._solutions = list()
        self._local_context = dict()
        self._var_default = None
        self._param_initial = None

        _add_series_accessor(self._local_context)

    def set_var_default(self, default):
        """ Sets the general default value for all variables. """
        self._var_default = default

    def var(self, name, desc=None, default=None):
        """ Creates a variable for use within the model.

            Arguments:
                name: The name of the variable.  This is the symbol
                    used in the equations.
                desc: A longer description of the variable.
                default: The default value of the variable, if the
                    value is not set.

            Returns: a Variable

            Raises:
                DuplicateNameError:
        """
        default = default or self._var_default
        if name in self.variables or name in self.parameters:
            raise DuplicateNameError('Name already in use: ' + name)
        var = Variable(name, desc=desc, default=default)
        self.variables[name] = var
        var.model = self

        _add_var_to_context(self._local_context, var)
        return var

    def set_param_initial(self, initial):
        """ Sets the default initial parameter value for all Parameters """
        self._param_initial = initial

    def param(self, name, desc=None, initial=None):
        """ Creates a parameter for use within the model.

            Returns: a Parameter
        """
        initial = initial or self._param_initial
        if name in self.variables or name in self.parameters:
            raise DuplicateNameError('Name already in use: ' + name)
        param = Parameter(name, desc=desc, initial=initial)
        self.parameters[name] = param
        _add_param_to_context(self._local_context, param)
        return param

    def add(self, equation, desc=None):
        """ Adds an equation to the model.

            Arguments:
                equation: A string containing the equation.  There
                    cannot be any free symbols in the equation, all
                    non-numeric symbols must have been defined as a
                    parameter, variable, or a sympy built-in
                    variable (like pi or E).
                desc: A description of the equation

            Returns: an Equation
        """
        eqn = Equation(equation, desc=desc)
        eqn.model = self
        eqn.parse(self._local_context)
        self.equations.append(eqn)

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

    def get_at(self, variable, iteration):
        """ Returns the value for a variable at an iteration.
            The value for iter may be positive or negative:
                If >= 0, absolute position
                    0 is the very first iteration.  Note that
                    iterations are available only AFTER the
                    solver has returned success.  Otherwise the
                    variable's default value is returned.
                If < 0, relative position
                    -1 is the iteration previous to the current
                    iteration.

            Arguments:
                variable:
                iteration:

            Returns:
                The value of the variable for that iteration.
        """
        # Need to replace the iteration needed with a variable
        # that represents the accessed variable
        #   for example, if are processing iteration 10
        #     then x[-1] -> _x__1
        #   _x__1 then gets added as a parameter
        #
        #   For positive iterations, that's a static iteration
        #   x[5] -> _x_5
        #   _x_5 will get added as a parameter.
        #   If iteration 5 has not happened yet, the default value
        #   value will be returned.
        #
        # Before solving, the appropriate values will be set in the
        # parameters.
        #

        if not iteration.is_number or not iteration.is_Number:
            raise EquationError('iteration-not-a-number',
                                str(iteration),
                                'iteration value must be a number')
        iter_value = int(iteration)
        if iter_value < 0:
            iter_name = "_{0}__{1}".format(str(variable), -iter_value)
        else:
            iter_name = "_{0}_{1}".format(str(variable), iter_value)

        param = Parameter(iter_name, initial=variable.default)
        self._private_parameters[iter_name] = param
        _add_param_to_context(self._local_context, param)
        return param
