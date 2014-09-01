""" Contains the Variable class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

from sympy import Symbol

from pylinsolve import InvalidNameError


class Variable(Symbol):
    """ This class contains a 'variable'.  This is a value that
        is being solved here, thus it can change during solving.
        (This is the opposite of a parameter, which is not changed
        by the solver during the solving of a problem).

        Unallowed names are the constants used by sympy:
            I, oo, nan, pi, E

        Attributes:
            name:
            desc:
            default:
            model:
            value:
            equation: This is the equation that is used to specify
                this variable.  In Linear Algebra this is the row
                used to specify the A(i,i) term.
    """
    # pylint: disable=too-many-ancestors

    ILLEGAL_NAMES = ['I', 'oo', 'nan', 'pi', 'E']

    def __init__(self, name, desc=None, default=None):
        if name in Variable.ILLEGAL_NAMES:
            raise InvalidNameError(name, 'Name already used by sympy')

        super(Variable, self).__init__(name)

        self.name = name
        self.desc = desc
        self.default = default
        self.model = None
        self.equation = None

        self.value = default
