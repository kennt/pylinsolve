""" Contains the Parameter class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import sympy

from pylinsolve import InvalidNameError
from pylinsolve.variable import Variable


class Parameter(object):
    """ This class contains a 'parameter'.  This is an exogenous
        variable.  The solver is not allowed to change this value
        when solving a set of equations.

        Attributes:
            symbol:
            name:
            desc:
            initial:
            value:
    """
    def __init__(self, name, desc=None, initial=None, symbol=None):
        if name in Variable.ILLEGAL_NAMES:
            raise InvalidNameError(name, 'Name already used by sympy')
        self.name = name
        self.desc = desc
        self.initial = initial

        self.symbol = symbol or sympy.Symbol(name)
        self.value = initial
