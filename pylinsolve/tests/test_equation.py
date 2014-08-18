""" equation unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

from pylinsolve.equation import Equation, _rewrite
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
        self.model = TestEquation.MockModel()
        self.model.variables()['x'] = Variable('x')
        self.model.variables()['y'] = Variable('y')
        self.model.variables()['z'] = Variable('z')

        self.model.parameters()['a'] = Parameter('a')
        self.model.parameters()['b'] = Parameter('b')

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
        self.assertEquals('x - y', _rewrite(variables, 'x - y'))
        self.assertEquals('xx - y', _rewrite(variables, 'xx - y'))
        self.assertEquals('xx - yx', _rewrite(variables, 'xx - yx'))
        self.assertEquals('xx(0) - yx', _rewrite(variables, 'xx(0) - yx'))
        self.assertEquals('x-(y)', _rewrite(variables, 'x = y'))
        self.assertEquals('{0}(-1)'.format(x_series),
                          _rewrite(variables, 'x(-1)'))
        self.assertEquals('{0}(-t)'.format(x_series),
                          _rewrite(variables, 'x(-t)'))

        self.assertEquals('z-({0}(10))'.format(x_series),
                          _rewrite(variables, 'z=x(10)'))

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

        print terms

        for term in eqn.variable_terms():
            self.assertTrue(term[1].name in self.model.variables())

            if term[1].name == 'x':
                self.assertEquals(0, len(term[0]))
            else:
                self.assertEquals(1, len(term[0]))
                self.assertEquals('-', term[0][0])

        self.assertTrue('x' in self.model.variables())
        self.assertTrue('y' in self.model.variables())

        terms = eqn.constant_term()
        self.assertIsNotNone(eqn)
        self.assertEquals(0, len(terms))

    # test handling of constant expressions
    # test handling of non-linear expressions
    # test handling of variable expressions
    # test multiple variables
    # test handling of numeric constants (e, pi)
    # test handling of functions (cos, sin, log)
    # error, no variables in equation
    # error, non-linear expressions
    # error, missing symbols in model
    # test rewriting, series accessor
    # test rewriting, = sign
    # test logic parts, what if vars and parameters mixed together
    # test to see if terms add up if in multiple parts 3*a*x + 4*x
    # test to see if constant terms sum up

