""" Contains the Parameter class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

from sympy import Symbol

from pylinsolve import InvalidNameError
from pylinsolve.variable import Variable


class Parameter(Symbol):
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
    # pylint: disable=too-many-ancestors

    def __init__(self, name, desc=None, initial=None):
        if name in Variable.ILLEGAL_NAMES:
            raise InvalidNameError(name, 'Name already used by sympy')

        super(Parameter, self).__init__(name)
        self.name = name
        self.desc = desc
        self.initial = initial

        self.value = initial
