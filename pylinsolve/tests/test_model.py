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

        self.assertTrue('x' in model.variables())
        self.assertTrue('x' not in model.parameters())

    def test_param(self):
        """ Test parameter creation """
        model = Model()
        param_x = model.param('x')
        self.assertIsNotNone(param_x)
        with self.assertRaises(DuplicateNameError):
            model.var('x')

        with self.assertRaises(DuplicateNameError):
            model.param('x')

        self.assertTrue('x' not in model.variables())
        self.assertTrue('x' in model.parameters())

    # test rule creation
    # test a simple rule
    # test for duplicate names
    # test for multiple name generation
    # test that the matrix is created correctly
    # test the solver
    # test the iterations
    # test the end condition
    # test for access to the solution
    # test for access to solutions array
    # test sparse matrix support
    # test for changes to params while running
