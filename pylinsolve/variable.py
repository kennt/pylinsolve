""" Contains the Variable class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import sympy

from ..value import Value

class Variable(Value):
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
        self._value = default
        self._values = list()

    @property
    def value(self):
        return self.value

    @value.setter
    def set_value(self, value):
        self._value = value
        self._values.append(value)

    def values(self):
        return self._values
