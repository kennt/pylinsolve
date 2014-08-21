""" model unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

from pylinsolve.model import Model, DuplicateNameError


class TestModel(unittest.TestCase):
    # pylint: disable=missing-docstring

    def setUp(self):
        pass

    def test_model(self):
        """ Create an empty model class """
        model = Model()
        self.assertIsNotNone(model)

    def test_var(self):
        """ Test variable creation """
        model = Model()
        var_x = model.var('x')
        self.assertIsNotNone(var_x)
        with self.assertRaises(DuplicateNameError):
            model.var('x')

        with self.assertRaises(DuplicateNameError):
            model.param('x')

        self.assertTrue('x' in model.variables)
        self.assertTrue('x' not in model.parameters)

    def test_param(self):
        """ Test parameter creation """
        model = Model()
        param_x = model.param('x')
        self.assertIsNotNone(param_x)
        with self.assertRaises(DuplicateNameError):
            model.var('x')

        with self.assertRaises(DuplicateNameError):
            model.param('x')

        self.assertTrue('x' not in model.variables)
        self.assertTrue('x' in model.parameters)

    # test rule creation
    # test a simple rule
    # test for multiple name generation
    # test that the matrix is created correctly
    # test the solver
    # test the iterations
    # test the end condition
    # test for access to the solution
    # test for access to solutions array
    # test for access to time series data
    # test sparse matrix support
    # test for changes to params while running
    # some numerical tests to check for accuracy

    def test_set_var_default(self):
        """ Test the set_var_default """
        model = Model()
        var = model.var('test')
        self.assertEquals(None, var.default)
        model.set_var_default(12)
        var = model.var('test2')
        self.assertEquals(12, var.default)

    def test_param_initial(self):
        """ Test the set_param_default """
        model = Model()
        param = model.param('test')
        self.assertEquals(None, param.initial)
        model.set_param_initial(122)
        param = model.param('test2')
        self.assertEquals(122, param.initial)

    def test_rule(self):
        """ Test creating rules """
        model = Model()
        model.var('x')
        model.var('y')
        model.add('x = y')

        self.assertEquals(2, len(model.variables))
        self.assertEquals(0, len(model.parameters))
        self.assertEquals(1, len(model.equations))

        eqn = model.equations[0]
        self.assertEquals(2, len(eqn.variable_terms()))
        self.assertTrue('x' in eqn.variable_terms())
        self.assertEquals(1, eqn.variable_terms()['x'])
        self.assertTrue('y' in eqn.variable_terms())
        self.assertEquals(-1, eqn.variable_terms()['y'])

    def test_rule_with_coefficients(self):
        """ Test creating rules with simple coefficents """
        model = Model()
        model.var('x')
        model.var('y')
        model.add('2*x + 3*y')

        self.assertEquals(2, len(model.variables))
        self.assertEquals(0, len(model.parameters))
        self.assertEquals(1, len(model.equations))

        eqn = model.equations[0]
        self.assertEquals(2, len(eqn.variable_terms()))
        self.assertTrue('x' in eqn.variable_terms())
        self.assertEquals(2, eqn.variable_terms()['x'])
        self.assertTrue('y' in eqn.variable_terms())
        self.assertEquals(3, eqn.variable_terms()['y'])

    # test mixed variable/parameter equations
    # test series accessor
    # error: series accessor with non-bound variable
    # test variable/parameter evaluation
    # test series accessor evaluation
