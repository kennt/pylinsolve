""" equation unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

from pylinsolve.equation import Equation


class TestEquation(unittest.TestCase):
    # pylint: disable=missing-docstring

    class MockModel(object):
        """ Mock model class used for equation testing. """
        def __init__(self):
            self._variables = {}
            self._parameters = {}

        def variables(self):
            return self._variables

        def parameters(self):
            return self._parameters

    def setUp(self):
        self.model = MockModel()

    def test_equation_init(self):
        """ Test if we can construct an instance """
        eqn = Equation('x = y')
        self.assertIsNotNone(eqn)

    def test_parse_one_variable(self):
        """ Test one-variable equation. """
        eqn = Equation('z')
        eqn.model = self.model
        self.assertIsNotNone(eqn)

        self.assertEquals('z', eqn.equation)

        terms = eqn.variable_terms()
        # list of (coefficient, variable) tuples
        # expect terms to be [([], Variable('x')), (['-'], Variable('y'))]
        self.assertEquals(1, len(terms))
        self.assertEquals('z', terms[0][1].name)
        self.assertTrue('z' in self.model.variables())

        self.assertTrue(terms[0][1].name in self.model.variables())

        self.assertEquals(0, len(terms[0][0]))

        terms = eqn.constant_terms()
        self.assertIsNotNone(eqn)
        self.assertEquals(0, len(terms))

    def test_parse_one_var_with_coeff(self):
        eqn = Equation('-2*z')
        eqn.model = self.model
        self.assertIsNotNone(eqn)

        self.assertEquals('-2*z', eqn.equation)

        terms = eqn.variable_terms()
        # list of (coefficient, variable) tuples
        # expect terms to be [([], Variable('x')), (['-'], Variable('y'))]
        self.assertEquals(1, len(terms))
        self.assertEquals('z', terms[0][1].name)
        self.assertTrue('z' in self.model.variables())

        self.assertTrue(terms[0][1].name in self.model.variables())

        self.assertEquals(1, len(terms[0][0]))
        self.assertEquals('-2', terms[0][0])

        terms = eqn.constant_terms()
        self.assertIsNotNone(eqn)
        self.assertEquals(0, len(terms))


    def test_simple_parse(self):
        """ Test very simple equation parsing """
        eqn = Equation('x = y')
        eqn.model = self.model
        self.assertIsNotNone(eqn)

        self.assertEquals('x = y', eqn.equation)

        terms = eqn.variable_terms()
        # list of (coefficient, variable) tuples
        # expect terms to be [([], Variable('x')), (['-'], Variable('y'))]
        self.assertEquals(2, len(terms))
        self.assertNotEqual(terms[0][1].name, terms[1][1].name)
        self.assertTrue(terms[0][1].name in ['x', 'y'])
        self.assertTrue(terms[1][1].name in ['x', 'y'])

        for term in eqn.variable_terms():
            self.assertTrue(term[1].name in self.model.variables())

            if term[1].name == 'x':
                self.assertEquals(0, len(term[0]))
            else:
                self.assertEquals(1, len(term[0]))
                self.assertEquals('-', term[0][0])

        self.assertTrue('x' in self.model.variables())
        self.assertTrue('y' in self.model.variables())

        terms = eqn.constant_terms()
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
