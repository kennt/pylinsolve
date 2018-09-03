""" equation unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import collections
import unittest

import sympy
from sympy import Symbol

from pysolve.equation import Equation, _rewrite, EquationError
from pysolve.model import _add_var_to_context, _add_param_to_context
from pysolve.model import _add_functions
from pysolve.parameter import Parameter
from pysolve.variable import Variable


class TestEquation(unittest.TestCase):
    """ Tests the Equation class """
    # pylint: disable=missing-docstring

    class MockModel(object):
        """ Mock model class used for equation testing. """
        def __init__(self):
            self.variables = collections.OrderedDict()
            self.parameters = collections.OrderedDict()
            self._local_context = {}

        def get_at(self, name, iteration):
            """ Implements the model get_at() """
            # pylint: disable=no-self-use
            if not iteration.is_number or not iteration.is_Number:
                raise EquationError('test-not-a-number', '', '')
            if iteration < 0:
                return Symbol("_{0}__{1}".format(name, -iteration))
            else:
                return Symbol("_{0}_{1}".format(name, iteration))

    def setUp(self):
        # pylint: disable=invalid-name

        self.model = TestEquation.MockModel()
        self.model.variables['x'] = Variable('x')
        self.model.variables['y'] = Variable('y')
        self.model.variables['z'] = Variable('z')
        self.x = self.model.variables['x']
        self.x.model = self.model
        self.y = self.model.variables['y']
        self.y.model = self.model
        self.z = self.model.variables['z']
        self.z.model = self.model

        self.model.parameters['a'] = Parameter('a')
        self.model.parameters['b'] = Parameter('b')
        self.a = self.model.parameters['a']
        self.a.model = self.model
        self.b = self.model.parameters['b']
        self.b.model = self.model

        for var in self.model.variables.values():
            _add_var_to_context(self.model._local_context, var)
        for param in self.model.parameters.values():
            _add_param_to_context(self.model._local_context, param)
        _add_functions(self.model._local_context)

    def test_equation_init(self):
        """ Test if we can construct an instance """
        eqn = Equation(self.model, 'x = y')
        self.assertIsNotNone(eqn)

    def test_equation_rewrite(self):
        """ Test the equation rewriting function """
        variables = dict()
        variables['x'] = Variable('x')
        variables['y'] = Variable('y')
        self.assertEquals('x - y', _rewrite(variables, {}, 'x - y'))
        self.assertEquals('xx - y', _rewrite(variables, {}, 'xx - y'))
        self.assertEquals('xx - yx', _rewrite(variables, {}, 'xx - yx'))
        self.assertEquals('xx(0) - yx', _rewrite(variables, {}, 'xx(0) - yx'))
        self.assertEquals('_series_acc(x,-1)',
                          _rewrite(variables, {}, 'x(-1)'))
        self.assertEquals('_series_acc(x,-t)',
                          _rewrite(variables, {}, 'x(-t)'))

        parameters = dict()
        parameters['a'] = Parameter('a')
        parameters['b'] = Parameter('b')
        self.assertEquals('_series_acc(a,-1)',
                          _rewrite({}, parameters, 'a(-1)'))

    def test_simple_parse(self):
        """ Test very simple equation parsing """
        eqn = Equation(self.model, 'x = y')
        self.assertIsNotNone(eqn)

        self.assertEquals('x = y', eqn.equation)
        eqn.parse(self.model._local_context)

        self.assertEquals('y', str(eqn.expr))
        self.assertEquals(self.y, eqn.expr)
        self.assertEquals(eqn, self.x.equation)
        self.assertEquals(self.x, eqn.variable)

    def test_parse_equals_sign_error(self):
        """ Test error handling for wrong number of "=" signs """
        eqn = Equation(self.model, 'x == y')
        with self.assertRaises(EquationError) as context:
            eqn.parse(self.model._local_context)
        self.assertEquals('equals-sign', context.exception.errorid)

        eqn = Equation(self.model, 'x - y')
        with self.assertRaises(EquationError) as context:
            eqn.parse(self.model._local_context)
        self.assertEquals('equals-sign', context.exception.errorid)

        eqn = Equation(self.model, 'x = y =')
        with self.assertRaises(EquationError) as context:
            eqn.parse(self.model._local_context)
        self.assertEquals('equals-sign', context.exception.errorid)

    def test_missing_left_hand_side(self):
        """ Test that the left-hand side contains a variable """
        eqn = Equation(self.model, 'a = x')
        with self.assertRaises(EquationError) as context:
            eqn.parse(self.model._local_context)
        self.assertEquals('lhs-variables', context.exception.errorid)

        eqn = Equation(self.model, '13 = x')
        with self.assertRaises(EquationError) as context:
            eqn.parse(self.model._local_context)
        self.assertEquals('lhs-variables', context.exception.errorid)

    def test_multi_vars_left_hand_side(self):
        """ Test for more than one variable on left hand side """
        eqn = Equation(self.model, 'x + y = 32')
        with self.assertRaises(EquationError) as context:
            eqn.parse(self.model._local_context)
        self.assertEquals('lhs-variables', context.exception.errorid)

    def test_non_linear_left_hand_side(self):
        """ Test for non-linearity in definition """
        with self.assertRaises(EquationError) as context:
            eqn = Equation(self.model, 'x**2 = y')
            eqn.parse(self.model._local_context)
        self.assertEquals('non-linear', context.exception.errorid)

        with self.assertRaises(EquationError) as context:
            eqn = Equation(self.model, 'log(x) = y')
            eqn.parse(self.model._local_context)
        self.assertEquals('non-linear', context.exception.errorid)

    def test_variable_already_defined(self):
        """ Test to see if a variable has two equations """
        eqn = Equation(self.model, 'x = 32 + y')
        eqn.parse(self.model._local_context)

        with self.assertRaises(EquationError) as context:
            eqn2 = Equation(self.model, 'x = z')
            eqn2.parse(self.model._local_context)
        self.assertEquals('var-eqn-exists', context.exception.errorid)

    def test_constant_expressions(self):
        """ Test the basic handling of simple constant expressions
            on the left-hand side of the expression.
        """
        # simple constants
        eqn = Equation(self.model, 'x-32=0')
        eqn.parse(self.model._local_context)
        self.assertEquals(32, eqn.expr)
        self.assertIsNotNone(self.x.equation)
        self.assertIsNone(self.y.equation)
        self.assertEquals(self.x.equation, eqn)

        # constants that use parameters
        self.x.equation = None
        eqn = Equation(self.model, 'x + 22*a=13')
        eqn.parse(self.model._local_context)
        self.assertEquals(0, 13-22*self.a - eqn.expr)

        # constants that use sympy symbols (such as pi, E)
        self.x.equation = None
        eqn = Equation(self.model, 'x + 44*pi*E = 0')
        eqn.parse(self.model._local_context)
        self.assertIsNotNone(self.x.equation)
        self.assertEquals(eqn, self.x.equation)
        self.assertEquals(0, -44*sympy.pi*sympy.E - eqn.expr)

        # constant expressions that use functions
        self.x.equation = None
        eqn = Equation(self.model, 'x+99*log(10) = 43')
        eqn.parse(self.model._local_context)
        self.assertIsNotNone(self.x.equation)
        self.assertEquals(eqn, self.x.equation)
        self.assertEquals(0, 43 - 99*sympy.log(10) - eqn.expr)

        # multiple constant expressions
        self.x.equation = None
        eqn = Equation(self.model, 'x - 3*pi**2 + 99*log(10) = y')
        eqn.parse(self.model._local_context)
        self.assertIsNotNone(self.x.equation)
        self.assertEquals(eqn, self.x.equation)
        self.assertEquals(0,
                          (self.y + 3*sympy.pi*sympy.pi -
                           99*sympy.log(10) - eqn.expr))

    def test_coefficient_handling(self):
        """ Test for coefficients on the left-hand side """
        self.x.equation = None
        eqn = Equation(self.model, '22*x = y')
        eqn.parse(self.model._local_context)
        self.assertIsNotNone(self.x.equation)
        self.assertEquals(eqn, self.x.equation)
        self.assertEquals(0, eqn.expr - (self.y / 22.))

        self.x.equation = None
        eqn = Equation(self.model, '22*a*x = y')
        eqn.parse(self.model._local_context)
        self.assertIsNotNone(self.x.equation)
        self.assertEquals(eqn, self.x.equation)
        self.assertEquals(0, eqn.expr - (self.y / (22*self.a)))

    def test_missing_symbols(self):
        """ Unknown symbols in equation """
        with self.assertRaises(NameError):
            eqn = Equation(self.model, '14*x = 23*ww')
            eqn.parse(self.model._local_context)

    def test_series_accessor(self):
        """ Test to see that the series accessor is converted correctly. """
        # This should work for variables, but not work for parameters
        eqn = Equation(self.model, 'x = x(-1)')
        eqn.parse(self.model._local_context)
        self.assertEquals('_x__1', str(eqn.expr))

    def test_parameter_series_accessor(self):
        """ Test that the series accessor works with parameters """
        eqn = Equation(self.model, 'x = a(-1)')
        eqn.parse(self.model._local_context)
        self.assertEquals('_a__1', str(eqn.expr))
