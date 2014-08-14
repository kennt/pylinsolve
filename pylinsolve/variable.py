""" Contains the Variable class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import sympy


class Variable(object):
    """ This class contains a 'variable'.  This is a value that
        is being solved here, thus it can change during solving.
        (This is the opposite of a parameter, which is not changed
        by the solver during the solving of a problem).

        Attributes:
            symbol:
            name:
            desc:
            default:
            value:
    """
    def __init__(self, name, desc=None, default=None, symbol=None):
        self.name = name
        self.desc = desc
        self.default = default

        self.symbol = symbol or sympy.Symbol(name)
        self.value = default
