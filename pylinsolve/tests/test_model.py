""" model unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

from pylinsolve.model import Model


class TestModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_model(self):
        model = Model()
        self.assertNotNone(model)

    def test_var(self):
        pass

    def test_param(self):
        pass

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
