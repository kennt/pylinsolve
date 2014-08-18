""" Contains the Variable class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import sympy

from pylinsolve import InvalidNameError


class Variable(object):
    """ This class contains a 'variable'.  This is a value that
        is being solved here, thus it can change during solving.
        (This is the opposite of a parameter, which is not changed
        by the solver during the solving of a problem).

        Unallowed names are the constants used by sympy:
            I, oo, nan, pi, E

        Attributes:
            symbol:
            name:
            desc:
            default:
            value:
    """

    ILLEGAL_NAMES = ['I', 'oo', 'nan', 'pi', 'E']

    def __init__(self, name, desc=None, default=None, symbol=None):
        if name in Variable.ILLEGAL_NAMES:
            raise InvalidNameError(name, 'Name already used by sympy')

        self.name = name
        self.desc = desc
        self.default = default

        self.symbol = symbol or sympy.Symbol(name)
        self.value = default

    @classmethod
    def series_name(cls, name):
        """ Returns the internal method to access the series data for
            this variable name.
        """
        return "__{0}_".format(name)
