""" variable unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

from pysolve import InvalidNameError
from pysolve.variable import Variable


class TestVariable(unittest.TestCase):
    """ Testcases for the Variable class
    """

    def setUp(self):
        pass

    def test_variable_create(self):
        """ Test simple variable creation """
        pass

    def test_illegal_names(self):
        """ Test for illeagl name handling """
        for name in Variable.ILLEGAL_NAMES:
            with self.assertRaises(InvalidNameError) as context:
                Variable(name)
            self.assertEquals(name, context.exception.name)
