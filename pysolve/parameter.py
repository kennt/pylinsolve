""" Contains the Parameter class.

    Copyright (c) 2014-2022 Kenn Takara
    See LICENSE for details

"""

# pylint: disable=duplicate-code

from sympy import Symbol

from pysolve import InvalidNameError
from pysolve.variable import PysolveVariable


class Parameter(Symbol):
    """ This class contains a 'parameter'.  This is an exogenous
        variable.  The solver is not allowed to change this value
        when solving a set of equations.

        Attributes:
            symbol:
            name:
            desc:
            default:
            value:
    """
    # pylint: disable=too-many-ancestors

    def __init__(self, name, desc=None, default=None):
        if name in PysolveVariable.ILLEGAL_NAMES:
            raise InvalidNameError(name, 'Name already used by sympy')

        super().__init__()
        self.name = name
        self.desc = desc
        self.default = default
        self.model = None

        self._index = None
        self._value = default

    @property
    def value(self):
        """ Getter accessor for parameter value """
        return self._value

    @value.setter
    def value(self, val):
        """ Setter accessor for parameter value """
        self._value = val


class SeriesParameter(Parameter):
    """ A parameter that can access the previous solution values.

        Attributes:
            name:
            variable:
            iteration:
            default:
    """
    # pylint: disable=too-many-ancestors

    def __init__(self, name, variable=None, iteration=None, default=None):
        super().__init__(name, default=default)
        if variable is None or iteration is None:
            raise ValueError('variable and iteration cannot be none')
        self.variable = variable
        self.iteration = iteration

    @property
    def value(self):
        """ Returns the value of a variable at a another iteration.

            If the iteration value is out-of-range, the variable's
            default value is returned.
        """
        try:
            return self.variable.model.get_value(
                self.variable, self.iteration)
        except IndexError:
            return self.variable.value or self.variable.default
