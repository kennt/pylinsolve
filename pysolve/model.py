""" Contains the main model class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import collections

import numpy

from sympy import sympify
from sympy import Symbol, Function
from sympy.core.cache import clear_cache
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import factorial_notation, auto_number
from sympy.utilities import lambdify

from pysolve.equation import Equation, EquationError, _rewrite
from pysolve.parameter import Parameter, SeriesParameter
from pysolve.utils import is_aclose
from pysolve.variable import Variable


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


class CalculationError(Exception):
    """ Exception: An error occurred while evaluating an equation """
    def __init__(self, inner, equation, context):
        super(CalculationError, self).__init__()
        self.inner = inner
        self.equation = equation
        self.context = context

    def __str__(self):
        return str(self.inner) + ' : ' + str(self.equation.equation)


def _common_iterable(obj):
    """ Use this to iterate through the list or dict """
    if isinstance(obj, dict):
        for key in obj:
            yield key, obj[key]
    else:
        for _, value in enumerate(obj):
            yield value[0], value[1]


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
        if (not isinstance(arg[0], Variable) and
                not isinstance(arg[0], Parameter)):
            raise EquationError('not-a-variable',
                                str(arg[0]),
                                'Must be a var or param')

        if arg[0].model is None:
            raise EquationError('no-model',
                                arg[0].name,
                                'Variable must belong to a model')
        return arg[0].model.get_at(arg[0], arg[1])


class _deltaFunction(Function):
    """ Implements d(x) == x - x(-1) """
    nargs = 1

    def fdiff(self, argindex=1):
        """ This function returns a constant value,
            therefore the derivative is 0
        """
        return 0

    @classmethod
    def eval(cls, *arg):
        """ Called from sympy to evaluate the function """
        if (not isinstance(arg[0], Variable) and
                not isinstance(arg[0], Parameter)):
            raise EquationError('d-arg-not-a-variable',
                                str(arg[0]),
                                'The arg to d() must be a var or param')
        return arg[0] - arg[0].model.get_at(arg[0], -1)


class _IfTrueFunction(Function):
    """ Implements a sympy function to implement a step function.
        This will return 1 if the argument is true, 0 otherwise.
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """ This function implements a Heaviside step function,
            but determing where it occurs is problematic.
        """
        return 0

    @classmethod
    def eval(cls, *args):
        """ Called from sympy to evaluate the function """
        if args[0]:
            return 1
        else:
            return 0


class _IfTrueNoEvalFunction(Function):
    """ Stops the evaluation of the function in sympy. """

    def fdiff(self, argindex=1):
        """ This function implements a Heaviside step function,
            but determing where it occurs is problematic.
        """
        return 0

    @classmethod
    def eval(cls, *args):
        """ Called during evaluation, but this one does nothing """
        pass

# Functions defined and used at parse time
_PARSE_FUNCS = [('_series_acc', _SeriesAccessor),
                ('d', _deltaFunction),
                ('if_true', _IfTrueNoEvalFunction)]

# Functions used at runtime
_RT_FUNCS = [('if_true', _IfTrueFunction, _IfTrueNoEvalFunction), ]

# Built-in funcs, supported by sympy
from sympy import exp, log, Abs, Min, Max, sign, sqrt
_BUILTIN_FUNCS = [('exp', exp),
                  ('log', log),
                  ('abs', Abs),
                  ('Max', Max),
                  ('Min', Min),
                  ('sign', sign),
                  ('sqrt', sqrt),
                  ]


def _add_functions(context):
    """ Adds our builtin functions.
    """
    for func in _PARSE_FUNCS:
        context[func[0]] = func[1]


def _build_jacobian(model):
    """ Creates the jacobian function matrix """
    jacobian = []
    nvars = len(model.variables.values())
    for var_i in model.variables.values():
        row_i = [None] * nvars
        for atom in var_i.equation.expr.atoms():
            if atom.is_Symbol and atom.name in model.variables:
                expr = var_i - var_i.equation.expr
                row_i[atom._index] = model._lambdify(expr.diff(atom))

        # add the partial derivative with respect to its own equation
        # We store the equations as var_i = expr
        # But what we really want is f(...) = 0, so convert
        #   var_i = expr  -->  var_i - expr = 0
        # and
        #   f(...) = var_i - expr = 0
        # therefore the derivative, df/dvar_i = 1
        row_i[var_i._index] = lambda *x: 1

        jacobian.append(row_i)
    return jacobian


def _evaluate_equations_vector(model, context, current):
    """ Evaluates the vector of equations at current """
    # pylint: disable=invalid-name, star-args
    nvars = len(model.variables.values())
    F = numpy.zeros((nvars, ))
    for i, var in enumerate(model.variables.values()):
        try:
            F[i] = -(current[var._index] - var.equation.func(*current))
        except Exception as err:
            raise CalculationError(err, var.equation, context)
    return F


def _evaluate_jacobian(model, jacobian, current):
    """ Evaluates the jacobian at current """
    # pylint: disable=invalid-name, star-args
    nvars = len(model.variables.values())
    J = numpy.zeros((nvars, nvars, ))
    for i in range(nvars):
        for j in range(nvars):
            if jacobian[i][j] is not None:
                J[i, j] = jacobian[i][j](*current)
    return J


class BroydenSolver(object):
    """ Implements the Broyden method for solving nonlinear equations.
    """
    def __init__(self, model):
        self.model = model
        self.jacobian = None
        self.prev_d_inv = None
        self.prev_d = None

    def setup(self):
        """ Perform any prep work """
        self.prev_d_inv = None
        self.prev_d = None
        if self.jacobian is None:
            self.jacobian = _build_jacobian(self.model)

    def reset(self):
        """ Clear any prep work """
        self.jacobian = None
        self.prev_d_inv = None
        self.prev_d = None

    def solve(self, context, current, next_soln):
        """ Performs a single iteration of the solver, generating a solution

            Arguments:
                current: The current solution, should not be modified.
                next_soln: On entry, this contains a copy of current, the
                    values for the next iteration should be placed here.

            Raises:
                CalculationError
        """
        # pylint: disable=invalid-name, too-many-locals

        # Here's the algorithm
        #   http://www.math.usm.edu/lambers/mat419/lecture11.pdf
        # initial solution: x0
        # D[0] = J(x[0])
        # d[0] = - inv(D[0]) * g(x[0])
        # x[1] = x[0] + d[0]
        # k = 0
        # until converge
        #   u[k] = inv(D[k]) * g(x[k+1])
        #   c[k] = d[k] * (d[k] + u[k])
        #   inv(D[k+1]) = inv(D[k]) - (u[k] * d[k])*inv(D[k])/c[k]
        #   d[k+1] = - inv(D[k+1])*g(x[k+1])
        #   x[k+2] = x[k+1] + d[k+1]
        #   k = k + 1

        if self.prev_d is None:
            g0 = _evaluate_equations_vector(self.model, context, current)
            # D[0] = J(x[0])
            D0 = _evaluate_jacobian(self.model, self.jacobian, current)

            # d[0] = - inv(D[0]) * g(x[0])
            D0inv = numpy.linalg.inv(D0)    # D0**-1
            d0 = numpy.inner(D0inv.dot(g0), -1)  # - (D0**-1 * g0)

            self.prev_d_inv = D0inv
            self.prev_d = d0

            dx = d0
        else:
            D0inv = self.prev_d_inv
            d0 = self.prev_d
            g1 = _evaluate_equations_vector(self.model, context, current)

            #   u[k] = inv(D[k]) * g(x[k+1])
            u0 = numpy.dot(D0inv, g1)       # D0inv * g1

            #   c[k] = d[k] * (d[k] + u[k])
            c0 = d0.dot(numpy.add(d0, u0))  # d0 * (d0 + u0)

            #   inv(D[k+1]) = inv(D[k]) - (u[k] * d[k])*inv(D[k])/c[k]
            D1inv = D0inv - (u0*d0.transpose())*D0inv/c0
            term1 = numpy.dot(numpy.outer(u0, d0), D0inv)  # (u0*d0T)*D0inv
            term1 = numpy.divide(term1, c0)
            D1inv = numpy.subtract(D0inv, term1)

            #   d[k+1] = - inv(D[k+1])*g(x[k+1])
            d1 = numpy.inner(numpy.dot(D1inv, g1), -1)

            self.prev_d_inv = D1inv
            self.prev_d = d1

            dx = d1

        # dx now contains x(n+1) - x(n), to get x(n+1) add back in x(n)
        for i, var in enumerate(self.model.variables.values()):
            next_soln[var._index] += float(dx[i])


class NewtonRaphsonSolver(object):
    """ Implements the Newton-Raphson method for solving a system of
        non-linear equations.
    """
    def __init__(self, model):
        self.model = model
        self.jacobian = None

    def setup(self):
        """ Perform any prepatory work before solving """
        if self.jacobian is None:
            self.jacobian = _build_jacobian(self.model)

    def reset(self):
        """ Reset the solver """
        self.jacobian = None

    def solve(self, context, current, next_soln):
        """ Performs a single iteration of the solver, generating a solution

            Arguments:
                current: The current solution, should not be modified.
                next_soln: On entry, this contains a copy of current, the
                    values for the next iteration should be placed here.

            Raises:
                CalculationError
        """
        # pylint: disable=invalid-name

        # evaluate the jacobian, and solve the linear equations
        #   J(X)(x(n+1) - x(n)) = -F(xn)
        F = _evaluate_equations_vector(self.model, context, current)
        J = _evaluate_jacobian(self.model, self.jacobian, current)

        try:
            x = numpy.linalg.solve(J, F)
        except numpy.linalg.LinAlgError as err:
            raise CalculationError(err, None, context)

        # x now contains x(n+1) - x(n), to get x(n+1) add back in x(n)
        for i, var in enumerate(self.model.variables.values()):
            next_soln[var._index] += float(x[i])


class GaussSeidelSolver(object):
    """ Implements the Gauss-Seidel method for solving a system
        of non-linear equations.
    """
    def __init__(self, model):
        self.model = model

    def setup(self):
        """ Perform any prepatory work before solving """
        pass

    def reset(self):
        """ Reset the solver """
        pass

    def solve(self, context, current, next_soln):
        """ Performs a single iteration of the solver, generating a solution

            Arguments:
                current: The current solution, should not be modified.
                next_soln: On entry, this contains a copy of current, the
                    values for the next iteration should be placed here.

            Raises:
                CalculationError
        """
        # pylint: disable=star-args, unused-argument
        for equation in self.model.equations:
            variable = equation.variable
            value = None
            try:
                value = variable.equation.func(*next_soln)
                next_soln[variable._index] = float(value)
            except Exception as err:
                # check to see if any of the atoms have a None value
                curr_context = {v: next_soln[v._index] for v in context.keys()}
                curr_context['__equation_value__'] = value
                for atom in variable.equation.expr.atoms():
                    if atom.is_Symbol and atom not in context:
                        raise CalculationError(
                            str(atom)+' has no value, cannot solve equation',
                            variable.equation,
                            curr_context)
                raise CalculationError(
                    err,
                    variable.equation,
                    curr_context)


class Model(object):
    """ This is the main Model class.  Variables, parameters, and
        equations are defined through this class.

        Attributes:
            variables:
            parameters:
            solutions:
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
        self.solutions = list()
        self.equations = list()

        self._private_parameters = collections.OrderedDict()
        self._local_context = dict()
        self._var_default = None
        self._param_default = None

        self._need_function_update = True

        _add_functions(self._local_context)

        # Variables used to lambdify the expressions
        self._arg_list = None
        self._private_funcs = None

        self._solvers = dict()
        self._solvers['newton-raphson'] = NewtonRaphsonSolver(self)
        self._solvers['gauss-seidel'] = GaussSeidelSolver(self)
        self._solvers['broyden'] = BroydenSolver(self)

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
        if default is None:
            default = self._var_default
        if name in self.variables or name in self.parameters:
            raise DuplicateNameError('Name already in use: ' + name)
        var = Variable(name, desc=desc, default=default)
        self.variables[name] = var
        var.model = self

        _add_var_to_context(self._local_context, var)
        self._need_function_update = True
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

    def _evaluate(self, value):
        """ Returns the float value or evaluates it """
        try:
            val = float(value)
        except ValueError:
            val = self.evaluate(value)
        return val

    def set_param_default(self, default):
        """ Sets the default initial parameter value for all Parameters """
        self._param_default = default

    def param(self, name, desc=None, default=None):
        """ Creates a parameter for use within the model.

            Returns: a Parameter
        """
        if default is None:
            default = self._param_default
        if name in self.variables or name in self.parameters:
            raise DuplicateNameError('Name already in use: ' + name)
        param = Parameter(name, desc=desc, default=default)
        param.model = self
        self.parameters[name] = param
        _add_param_to_context(self._local_context, param)
        self._need_function_update = True
        return param

    def set_values(self, values, ignore_errors=False):
        """ Sets the values of variables or parameters """
        for name, value in _common_iterable(values):
            if name in self.variables:
                self.variables[name].value = self._evaluate(value)
            elif name in self.parameters:
                self.parameters[name].value = self._evaluate(value)
            elif not ignore_errors:
                raise ValueError(
                    "{0} is not a parameter/variable".format(name))

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
        eqn = Equation(self, equation, desc=desc)
        eqn.parse(self._local_context)
        self._need_function_update = True

        self.equations.append(eqn)
        return eqn

    def _validate_equations(self):
        """ Does some validation """
        # Make sure that each variable has an equation
        for variable in self.variables.values():
            if variable.equation is None:
                raise EquationError('under-specified',
                                    variable.name,
                                    'variable does not have equation')

    def _get_context(self):
        """ Prepares the context for evaluation """
        context = collections.OrderedDict()

        for variable in self.variables.values():
            if variable.value is not None:
                context[variable] = float(variable.value)
        for param in self.parameters.values():
            if param.value is not None:
                context[param] = float(param.value)
        for param in self._private_parameters.values():
            if param.value is not None:
                context[param] = float(param.value)
        return context

    def _update_solutions(self, solution):
        """ Given the solution, update the variables and the
            solutions list.
        """
        self.set_values(solution, ignore_errors=True)
        self.solutions.append(solution.copy())

    def _clear_lambda_args(self):
        """ Clears the list of lambda args, ensures that the
            variables will get rebuilt
        """
        self._arg_list = None
        self._private_funcs = None

    def _build_lambda_args(self, context):
        """ Creates the argument list for lambdify

            This should be the same argument list for all functions
            that are lambdified.
        """
        if self._arg_list is None:
            self._arg_list = [x for x in context.keys()]
            for i in range(len(self._arg_list)):
                if isinstance(self._arg_list[i], Symbol):
                    self._arg_list[i]._index = i

        if self._private_funcs is None:
            self._private_funcs = {x[2].__name__: x[1] for x in _RT_FUNCS}
            for func in _BUILTIN_FUNCS:
                self._private_funcs[func[0]] = func[1]

    def _lambdify(self, expr):
        """ Creates a lambdified expression with the appropriate args """
        return lambdify(self._arg_list, expr, self._private_funcs)

    def _run_solver(self,
                    solver,
                    context,
                    max_iterations=10,
                    until=None,
                    threshold=0.001,
                    debuglist=None):
        """ Runs the main solver loop

            Returns: a context with the values of the solution

            Raises:
                SolutionNotFoundError
        """
        # pylint: disable=too-many-locals

        testf = (until or (lambda x1, x2: is_aclose(x1, x2, rtol=threshold)))

        if debuglist is not None:
            debuglist.append(context)

        next_soln = [float(x) for x in context.values()]
        soln = None

        for _ in range(max_iterations):
            current = next_soln
            next_soln = list(current)

            solver.solve(context, current, next_soln)

            if debuglist is not None:
                debuglist.append({v: next_soln[v._index]
                                 for v in context.keys()})

            if testf(current, next_soln):
                soln = {v: next_soln[v._index] for v in context.keys()}
                break

        if soln is None:
            # determine the variables that have not converged
            problem_vars = []
            for variable in self.variables.values():
                if not testf([current[variable._index], ],
                             [next_soln[variable._index], ]):
                    problem_vars.append(variable.name)
            raise SolutionNotFoundError(', '.join(problem_vars) +
                                        ' have not converged')
        return soln

    def solve(self, iterations=10, until=None, threshold=0.001,
              debuglist=None, method='gauss-seidel'):
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
                debuglist: If this is set, then the current context will
                    be appended every iteration when solving.
                method: The method that should be used for solving the
                    system of equations:
                    available methods:
                        gauss-seidel
                        newton-raphson

            Raises:
                SolutionNotFoundError:
        """
        # pylint: disable=invalid-name
        self._validate_equations()

        current = self._get_context()
        if len(self.solutions) == 0:
            self._update_solutions({k.name: v for k, v in current.items()})

        # do we need to update the function lambdas?  This is needed
        # if the number of variables/parameters/equations change.
        if self._need_function_update:
            self._clear_lambda_args()
            self._build_lambda_args(current)

            for var in self.variables.values():
                var.equation.func = self._lambdify(var.equation.expr)

            for solver in self._solvers.values():
                solver.reset()

            self._need_function_update = False

        if method not in self._solvers:
            raise ValueError(
                '{0} is not a valid solver method type'.format(method))
        solver = self._solvers[method]

        solver.setup()
        solution = self._run_solver(solver,
                                    current,
                                    max_iterations=iterations,
                                    until=until,
                                    threshold=threshold,
                                    debuglist=debuglist)

        soln = {k.name: v for k, v in solution.items()}
        self._update_solutions(soln)

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

        if iter_name not in self._private_parameters:
            param = SeriesParameter(iter_name,
                                    variable=variable,
                                    iteration=iter_value,
                                    default=variable.default)
            self._private_parameters[iter_name] = param
            _add_param_to_context(self._local_context, param)
            self._need_function_update = True
        return self._private_parameters[iter_name]

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

    def evaluate(self, equation):
        """ Evaluates an arbitrary function using the current values
            of the variables.

            Parmeters:
                eqn:

            Returns:
                The value of the expression.
        """
        equation = _rewrite(self.variables, self.parameters, equation)
        expr = parse_expr(equation,
                          self._local_context,
                          transformations=(factorial_notation, auto_number))
        expr = sympify(expr).subs(self._get_context())
        for func in _RT_FUNCS:
            expr = expr.replace(func[2], func[1])
            if sympify(expr).is_number:
                break

        return float(expr)
