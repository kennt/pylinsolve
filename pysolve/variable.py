""" Contains the PysolveVariable class.

    Copyright (c) 2014-2022 Kenn Takara
    See LICENSE for details

"""

# pylint: disable=duplicate-code

from sympy import Symbol

from pysolve import InvalidNameError


class PysolveVariable(Symbol):
    """ This class contains a 'variable'.  This is a value that
        is being solved here, thus it can change during solving.
        (This is the opposite of a parameter, which is not changed
        by the solver during the solving of a problem).

        Unallowed names are the constants used by sympy:
            I, oo, nan, pi, E

        Attributes:
            name: The symbolic name of the variable.
            desc: A long form description of the variable.
            default: The default value (the initial value of the var)
            model: The model that this variable belongs to.
            value: The actual (current) value of the var.
            equation: This is the equation used to evaluate the
                variable.
    """
    # pylint: disable=too-many-ancestors

    ILLEGAL_NAMES = ['I', 'oo', 'nan', 'pi', 'E']

    def __init__(self, name, desc=None, default=None):
        if name in PysolveVariable.ILLEGAL_NAMES:
            raise InvalidNameError(name, 'Name already used by sympy')

        super().__init__()

        self.name = name
        self.desc = desc
        self.default = default
        self.model = None
        self.equation = None
        self.value = default

        self._index = None
