""" Contains the main model class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import collections

import numpy
from sympy import sympify
from sympy import Function
from sympy.core.cache import clear_cache

from pylinsolve.equation import Equation, EquationError
from pylinsolve.parameter import Parameter, SeriesParameter
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
    def __init__(self, text=None):
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
        """ Called from sympy to evaluate the function """
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


def _run_solver(A, x, b,
                max_iterations=10, relax=1.0,
                until=None, threshold=0.001,
                decimals=None, debuglist=None):
    """ Runs the main solver loop

        Returns:
            A numpy vector containing the solution of the
            equations.

        Raises:
            SolutionNotFoundError
    """
    # pylint: disable=invalid-name,too-many-locals
    testf = (until or
             (lambda x1, x2: numpy.allclose(x1, x2, rtol=threshold)))

    curr = numpy.copy(x)
    if debuglist is not None:
        debuglist.append(curr)

    soln = None

    for _ in xrange(max_iterations):
        nextx = numpy.copy(curr)

        for i in xrange(A.shape[0]):
            sub1 = A[i, :i].dot(nextx[:i])
            sub2 = A[i, i+1:].dot(curr[i+1:])
            nextx[i] = (nextx[i] +
                        relax * (((b[i] - sub1 - sub2) / A[i, i]) - nextx[i]))

        if debuglist is not None:
            debuglist.append(nextx)

        if testf(curr, nextx):
            soln = nextx
            break
        curr = nextx

    if soln is None:
        raise SolutionNotFoundError()
    if decimals is not None:
        soln = numpy.around(soln, decimals=decimals)
    return soln


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

        # Upon creating a new model, clear the cache
        # Otherwise creating multiple models creates
        # problems because sympy() will not reevaluate
        # functions and the series accessor will not
        # get created.  Because sympy keeps this cache
        # around, will have to be careful if using these
        # models in a multi-threaded context.
        clear_cache()

        self.variables = collections.OrderedDict()
        self.parameters = collections.OrderedDict()
        self.equations = list()
        self.solutions = list()

        self._private_parameters = collections.OrderedDict()
        self._local_context = dict()
        self._var_default = None
        self._param_initial = None

        # mapping of variable -> index
        # used for the matrix algebra solvers
        self._var_map = collections.OrderedDict()

        # mapping of variable -> equation
        self._var_eqn = dict()

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

        self._var_map[var] = len(self._var_map)
        return var

    def vars(self, *args):
        """ Creates multiple variables at the same time.

            Arguments:
                *args: the names of the variables to create

            Returns:
                a list of the variables created
        """
        varlist = list()
        for arg in args:
            varlist.append(self.var(arg))
        return varlist

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

    def _validate_equations(self):
        """ Does some validation """
        # Make sure that each variable has an equation
        for variable in self.variables.values():
            if variable.equation is None:
                raise EquationError('under-specified',
                                    variable.name,
                                    'variable does not have equation')

    def _latest_solution_vector(self):
        """ Returns the latest solution vector.  If there is
            none, returns a vector with the default values.

            Returns:
                A numpy array of dimension 1
        """
        latest = numpy.zeros((len(self.variables),))
        for variable in self.variables.values():
            index = self._var_map[variable]
            latest[index] = variable.value
        return latest

    def _prepare_solver(self, sparse=False):
        """ Prepares the solver for running.
            This will run through the equations, evaluating the
            parameters/variables and placing the cofficients into
            matrices for evaluation.

            Returns:
                A tuple of matrices, (A, b)
        """
        # pylint: disable=invalid-name

        # prepare the local context in order to eval the
        # variables
        context = dict()
        for param in self.parameters.values():
            context[param] = param.value
        for param in self._private_parameters.values():
            context[param] = param.value

        nvars = len(self.variables)
        b = numpy.zeros((nvars,))
        if sparse:
            from scipy.sparse import dok_matrix
            A = dok_matrix((nvars, nvars))
        else:
            A = numpy.zeros((nvars, nvars))
        row = 0
        for variable in self.variables.values():
            for varname, term in variable.equation.variable_terms().items():
                index = self._var_map[self.variables[varname]]
                A[row, index] = sympify(term).evalf(subs=context)

            b[row] = -sympify(
                variable.equation.constant_term()).evalf(subs=context)
            row += 1

        if sparse:
            A = A.tocsr()
        return (A, b)

    def _update_solutions(self, solution):
        """ Unpack the solutions from the numpy vector into a
            dict() and update the solutions array.
        """
        new_soln = collections.OrderedDict()
        for variable in self.variables.values():
            index = self._var_map[variable]
            variable.value = solution[index]
            new_soln[variable.name] = float(variable.value)
        for param in self.parameters.values():
            new_soln[param.name] = float(param.value)
        self.solutions.append(new_soln)

    def solve(self, iterations=10, until=None, threshold=0.001,
              relaxation=1.0, decimals=None, sparse=False):
        """ Runs the solver.

            The solver will try to find a solution until one of the
            conditions is reached:
                (1) the iterations limit is reached
                    In this case a SolutionNotFoundError will be raised.
                (2) the until condition returns True

            Arguments:
                iterations: The number of iterations to run until
                    a solution is reached. (Default: 10)
                until: This is a function that determines whether or
                    not a solution is reached.  (Default: residual error)
                    This takes two parameters, the previous solution
                    vector and the current solution vector.
                threshold: If using the default end condition, this is the
                    threshold that the residuals must be less than.
                relaxation: Set this to use SOR rather than pure
                    Gauss-Seidel iteration. (Default: 1.0, Gauss-Seidel)

            Raises:
                SolutionNotFoundError:
        """
        # pylint: disable=invalid-name
        self._validate_equations()
        x = self._latest_solution_vector()
        if len(self.solutions) == 0:
            self._update_solutions(x)
        A, b = self._prepare_solver(sparse=sparse)
        solution = _run_solver(A, x, b,
                               max_iterations=iterations,
                               until=until,
                               threshold=threshold,
                               relax=relaxation,
                               decimals=decimals)

        # unpack the solution vector into the variables and
        # solution dict()
        self._update_solutions(solution)

    def get_at(self, variable, iteration):
        """ Returns the variable for a previous iteration.
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
        iter_value = sympify(iteration)
        if not iter_value.is_number or not iter_value.is_Number:
            raise EquationError('iteration-not-a-number',
                                str(iter_value),
                                'iteration value must be a number')
        iter_value = int(iter_value)
        if iter_value < 0:
            iter_name = "_{0}__{1}".format(str(variable), -iter_value)
        else:
            iter_name = "_{0}_{1}".format(str(variable), iter_value)

        param = SeriesParameter(iter_name,
                                variable=variable,
                                iteration=iter_value,
                                initial=variable.default)
        self._private_parameters[iter_name] = param
        _add_param_to_context(self._local_context, param)
        return param

    def get_value(self, variable, iteration):
        """ Returns the value of the variable for the given
            iteration.  Iteration may be +/-.  If the iteration
            value is out of range, None is returned.

            Parameters:
                variable: The variable whose value we want.
                iteration: The iteration value that we want to
                    look at.  If this is positive, then it is the
                    absolute iteration position.  If negative, then
                    it is the relative iteration position (thus a
                    value of -1 means the previous iteration)

            Returns:
                The value of the variable at that iteration is
                returned. If the iteration is out of range, an
                IndexError exception will be raised.
        """
        return self.solutions[iteration][variable.name]
