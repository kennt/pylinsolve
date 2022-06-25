""" variable unit tests

    Copyright (c) 2014-2022 Kenn Takara
    See LICENSE for details

"""

# pylint: disable=duplicate-code

import unittest

from pysolve import InvalidNameError
from pysolve.variable import PysolveVariable


class TestPysolveVariable(unittest.TestCase):
    """ Testcases for the PysolveVariable class
    """

    def setUp(self):
        # pylint: disable=unnecessary-pass
        pass

    def test_illegal_names(self):
        """ Test for illeagl name handling """
        for name in PysolveVariable.ILLEGAL_NAMES:
            with self.assertRaises(InvalidNameError) as context:
                PysolveVariable(name)
            self.assertEqual(name, context.exception.name)
