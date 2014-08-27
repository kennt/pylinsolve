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

        self._value = initial

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
            initial:
    """
    # pylint: disable=too-many-ancestors

    def __init__(self, name, variable=None, iteration=None, initial=None):
        super(SeriesParameter, self).__init__(name, initial=initial)
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
            return self.variable.default
