""" Contains the Parameter class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import sympy


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
        self.name = name
        self.desc = desc
        self.initial = initial

        self.symbol = symbol or sympy.Symbol(name)
        self.value = initial
