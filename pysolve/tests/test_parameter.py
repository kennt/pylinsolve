""" parameter unit tests

    Copyright (c) 2014-2022 Kenn Takara
    See LICENSE for details

"""

import unittest

from pysolve import InvalidNameError
from pysolve.parameter import Parameter, SeriesParameter
from pysolve.variable import PysolveVariable


class TestParameter(unittest.TestCase):
    """ Testcases for the Parameter class
    """

    def test_parameter_create(self):
        """ Test simple variable creation """
        param = Parameter('a')
        self.assertIsNotNone(param)

    def test_illegal_names(self):
        """ Test for illeagl name handling """
        for name in PysolveVariable.ILLEGAL_NAMES:
            with self.assertRaises(InvalidNameError) as context:
                Parameter(name)
            self.assertEqual(name, context.exception.name)

    def test_parameter_access(self):
        """ Test access to get/set of value attribute """
        param = Parameter('a')
        param.value = 1.2
        self.assertEqual(1.2, param.value)


class TestSeriesParameter(unittest.TestCase):
    """ Testcases for the SeriesParameter class """

    class MockModel:
        """ Mockup of the Model class """
        # pylint: disable=too-few-public-methods
        def __init__(self):
            self.last_variable = None
            self.last_iteration = None

        def get_value(self, variable, iteration):
            """ Mock call, records the parameters for the last call """
            self.last_variable = variable
            self.last_iteration = iteration
            return iteration

    def test_seriesparameter_create(self):
        """ Simple SeriesParameter create """
        variable = PysolveVariable('x')
        param = SeriesParameter('a', variable=variable, iteration=0)
        self.assertIsNotNone(param)

    def test_seriesparameter_access(self):
        """ Test get/set access of value attribute """
        model = TestSeriesParameter.MockModel()

        varx = PysolveVariable('x', default=-1)
        varx.model = model

        param = SeriesParameter('a', variable=varx, iteration=-1)
        self.assertEqual(-1, param.value)
        self.assertEqual(model.last_variable, varx)
        self.assertEqual(model.last_iteration, -1)

        param = SeriesParameter('a', variable=varx, iteration=2)
        self.assertEqual(2, param.value)
        self.assertEqual(model.last_variable, varx)
        self.assertEqual(model.last_iteration, 2)

        with self.assertRaises(AttributeError):
            param.value = 4
