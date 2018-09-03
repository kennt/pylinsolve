""" parameter unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

from pysolve import InvalidNameError
from pysolve.parameter import Parameter, SeriesParameter
from pysolve.variable import Variable


class TestParameter(unittest.TestCase):
    """ Testcases for the Parameter class
    """

    def test_parameter_create(self):
        """ Test simple variable creation """
        param = Parameter('a')
        self.assertIsNotNone(param)

    def test_illegal_names(self):
        """ Test for illeagl name handling """
        for name in Variable.ILLEGAL_NAMES:
            with self.assertRaises(InvalidNameError) as context:
                Parameter(name)
            self.assertEquals(name, context.exception.name)

    def test_parameter_access(self):
        """ Test access to get/set of value attribute """
        param = Parameter('a')
        param.value = 1.2
        self.assertEquals(1.2, param.value)


class TestSeriesParameter(unittest.TestCase):
    """ Testcases for the SeriesParameter class """

    class MockModel(object):
        """ Mockup of the Model class """
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
        variable = Variable('x')
        param = SeriesParameter('a', variable=variable, iteration=0)
        self.assertIsNotNone(param)

    def test_seriesparameter_access(self):
        """ Test get/set access of value attribute """
        model = TestSeriesParameter.MockModel()

        varx = Variable('x', default=-1)
        varx.model = model

        param = SeriesParameter('a', variable=varx, iteration=-1)
        self.assertEquals(-1, param.value)
        self.assertEquals(model.last_variable, varx)
        self.assertEquals(model.last_iteration, -1)

        param = SeriesParameter('a', variable=varx, iteration=2)
        self.assertEquals(2, param.value)
        self.assertEquals(model.last_variable, varx)
        self.assertEquals(model.last_iteration, 2)

        with self.assertRaises(AttributeError):
            param.value = 4
