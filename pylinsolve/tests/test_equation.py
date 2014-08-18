""" equation unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

import sympy

from pylinsolve.equation import Equation, _rewrite, EquationError
from pylinsolve.model import _add_var_to_context, _add_param_to_context
from pylinsolve.parameter import Parameter
from pylinsolve.variable import Variable


class TestEquation(unittest.TestCase):
    # pylint: disable=missing-docstring

    class MockModel(object):
        """ Mock model class used for equation testing. """
        def __init__(self):
            self._variables = {}
            self._parameters = {}
            self._local_context = {}

        def variables(self):
            return self._variables

        def parameters(self):
            return self._parameters

    def setUp(self):
        # pylint: disable=invalid-name

        self.model = TestEquation.MockModel()
        self.model.variables()['x'] = Variable('x')
        self.model.variables()['y'] = Variable('y')
        self.model.variables()['z'] = Variable('z')
        self.x = self.model.variables()['x'].symbol
        self.y = self.model.variables()['y'].symbol
        self.z = self.model.variables()['z'].symbol

        self.model.parameters()['a'] = Parameter('a')
        self.model.parameters()['b'] = Parameter('b')
        self.a = self.model.parameters()['a'].symbol
        self.b = self.model.parameters()['b'].symbol

        for var in self.model.variables().values():
            _add_var_to_context(self.model._local_context, var)
        for param in self.model.parameters().values():
            _add_param_to_context(self.model._local_context, param)

    def test_equation_init(self):
        """ Test if we can construct an instance """
        eqn = Equation('x = y')
        self.assertIsNotNone(eqn)

    def test_equation_rewrite(self):
        """ Test the equation rewriting function """
        variables = dict()
        variables['x'] = Variable('x')
        variables['y'] = Variable('y')
        x_series = Variable.series_name('x')
        self.assertEquals('x - y', _rewrite(variables, {}, 'x - y'))
        self.assertEquals('xx - y', _rewrite(variables, {}, 'xx - y'))
        self.assertEquals('xx - yx', _rewrite(variables, {}, 'xx - yx'))
        self.assertEquals('xx(0) - yx', _rewrite(variables, {}, 'xx(0) - yx'))
        self.assertEquals('x-(y)', _rewrite(variables, {}, 'x = y'))
        self.assertEquals('{0}(-1)'.format(x_series),
                          _rewrite(variables, {}, 'x(-1)'))
        self.assertEquals('{0}(-t)'.format(x_series),
                          _rewrite(variables, {}, 'x(-t)'))

        self.assertEquals('z-({0}(10))'.format(x_series),
                          _rewrite(variables, {}, 'z=x(10)'))

        with self.assertRaises(EquationError) as context:
            _rewrite(variables, {'a': Parameter('a')}, 'a(-1)')
        self.assertEquals('parameter-function', context.exception.errorid)

    def test_parse_one_variable(self):
        """ Test one-variable equation. """
        eqn = Equation('z')
        eqn.model = self.model
        self.assertIsNotNone(eqn)

        self.assertEquals('z', eqn.equation)

        # need to generate the local context from the model
        eqn.parse(self.model._local_context)

        terms = eqn.variable_terms()
        self.assertEquals(1, len(terms))
        self.assertTrue('z' in terms)
        self.assertEquals(1, terms['z'])
        self.assertTrue('z' in self.model.variables())

        self.assertTrue(terms.keys()[0] in self.model.variables())

        term = eqn.constant_term()
        self.assertIsNotNone(term)
        self.assertEquals(0, term)

    def test_parse_one_var_with_coeff(self):
        eqn = Equation('-2*z')
        eqn.model = self.model
        self.assertIsNotNone(eqn)

        self.assertEquals('-2*z', eqn.equation)

        eqn.parse(self.model._local_context)

        terms = eqn.variable_terms()
        self.assertEquals(1, len(terms))
        self.assertTrue('z' in terms)
        self.assertTrue('z' in self.model.variables())

        self.assertTrue(terms.keys()[0] in self.model.variables())

        self.assertEquals(-2, terms['z'])

        term = eqn.constant_term()
        self.assertIsNotNone(term)
        self.assertEquals(0, term)

    def test_simple_parse(self):
        """ Test very simple equation parsing """
        eqn = Equation('x = y')
        eqn.model = self.model
        self.assertIsNotNone(eqn)

        self.assertEquals('x = y', eqn.equation)
        eqn.parse(self.model._local_context)

        terms = eqn.variable_terms()
        self.assertEquals(2, len(terms))
        self.assertTrue(terms.keys()[0] in ['x', 'y'])
        self.assertTrue(terms.keys()[1] in ['x', 'y'])

        self.assertEquals(1, terms['x'])
        self.assertEquals(-1, terms['y'])

        self.assertTrue('x' in self.model.variables())
        self.assertTrue('y' in self.model.variables())

        term = eqn.constant_term()
        self.assertEquals(0, term)

    def test_constant_expressions(self):
        """ Test the basic handling of simple constant expressions.
        """
        # simple constants
        eqn = Equation('32')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(0, len(eqn.variable_terms()))
        self.assertEquals(32, eqn.constant_term())

        # constants that use parameters
        eqn = Equation('22*a')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(0, len(eqn.variable_terms()))
        self.assertEquals('22*a', str(eqn.constant_term()))

        # constants that use sympy symbols (such as pi, E)
        eqn = Equation('44*pi*E')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(0, len(eqn.variable_terms()))
        self.assertEquals(0, 44*sympy.pi*sympy.E - eqn.constant_term())

        # constant expressions that use functions
        eqn = Equation('99*log(10)')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(0, len(eqn.variable_terms()))
        self.assertEquals(0, 99*sympy.log(10) - eqn.constant_term())

        # multiple constant expressions
        eqn = Equation('3*pi**2 + 99*log(10)')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(0, len(eqn.variable_terms()))
        self.assertEquals(
            0, (3*sympy.pi*sympy.pi + 99*sympy.log(10)) - eqn.constant_term())

        # constants on the other side of =
        eqn = Equation('4*pi**2 = 101*log(10)')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(0, len(eqn.variable_terms()))
        self.assertEquals(
            0,
            (4*sympy.pi*sympy.pi - 101*sympy.log(10)) - eqn.constant_term())

    def test_variable_expressions(self):
        """ Test more complicated variable expressions """
        eqn = Equation('4*x + 3*pi*x')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(1, len(eqn.variable_terms()))
        self.assertEquals(0, (4 + 3*sympy.pi) - eqn.variable_terms()['x'])
        self.assertEquals(0, eqn.constant_term())

        eqn = Equation('4*x*log(5) + 3*pi*x')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(1, len(eqn.variable_terms()))
        self.assertEquals(
            0,
            (4*sympy.log(5) + 3*sympy.pi) - eqn.variable_terms()['x'])
        self.assertEquals(0, eqn.constant_term())

        eqn = Equation('4*x*b + 3*a*x')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(1, len(eqn.variable_terms()))
        self.assertEquals(
            0,
            (4*self.b + 3*self.a) - eqn.variable_terms()['x'])
        self.assertEquals(0, eqn.constant_term())

    def test_multiple_variables(self):
        """ Multiple-variable equations """
        eqn = Equation('14*x + 3.6*pi*z')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(2, len(eqn.variable_terms()))
        self.assertEquals(0, 14 - eqn.variable_terms()['x'])
        self.assertEquals(0, (3.6*sympy.pi) - eqn.variable_terms()['z'])
        self.assertEquals(0, eqn.constant_term())

    def test_nonlinear_expressions(self):
        """ Non-linear expressions """
        with self.assertRaises(EquationError) as context:
            eqn = Equation('14*x**2 + 3.6*pi*z')
            eqn.model = self.model
            eqn.parse(self.model._local_context)
        self.assertEquals('unexpected-term', context.exception.errorid)

        with self.assertRaises(EquationError) as context:
            eqn = Equation('14*x*y + 3.6*pi*z')
            eqn.model = self.model
            eqn.parse(self.model._local_context)
        self.assertEquals('not-independent', context.exception.errorid)

        with self.assertRaises(EquationError) as context:
            eqn = Equation('14*a*log(x) + 3.6*pi*z')
            eqn.model = self.model
            eqn.parse(self.model._local_context)
        self.assertEquals('non-linear', context.exception.errorid)

    def test_missing_symbols(self):
        """ Unknown symbols in equation """
        with self.assertRaises(NameError):
            eqn = Equation('14*x + 23*ww')
            eqn.model = self.model
            eqn.parse(self.model._local_context)

    def test_series_accessor(self):
        """ Test to see that the series accessor is converted correctly. """
        # This should work for variables, but not work for parameters
        eqn = Equation('x - x(-1)')
        eqn.model = self.model
        eqn.parse(self.model._local_context)

        self.assertEquals(1, len(eqn.variable_terms()))
        self.assertEquals(1, eqn.variable_terms()['x'])
        self.assertEquals('-{0}(-1)'.format(Variable.series_name('x')),
                          str(eqn.constant_term()))

        with self.assertRaises(EquationError) as context:
            eqn = Equation('x - a(-1)')
            eqn.model = self.model
            eqn.parse(self.model._local_context)
        self.assertEquals('parameter-function', context.exception.errorid)
