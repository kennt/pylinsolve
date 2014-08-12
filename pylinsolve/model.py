""" Contains the main model class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import collections

from ..variable import Variable


class DuplicateNameError(Exception):
    """ Exception: Duplicate name detedted, name already in use. """
    def __init__(self, text):
        super(DuplicateNameError, self).__init__()
        self.text = text

    def __str__(self):
        return self.text


class Model(object):
    def __init__(self):
        self._vars = collections.OrderedDict()
        self._params = collections.OrderedDict()
        self._rules = collections.OrderedDict()
        self._solutions = list()

    def var(self, name, desc=None, initial=None, symbol=None):
        """ Creates a variable for use within the model.

            Returns: a Variable
        """
        if name in self._vars or name in self._params:
            raise DuplicateNameError('Name already in use: ' + name)
        var = Variable(name, desc=desc, default=default, symbol=symbol)
        self._vars[name] = var
        return var

    def vars(self, names):
        # Perform the symbol expansion and then add the names
        for sym in sympy.symbols(names):
            self.var(sym.name, symbol=sym)
        pass

    def param(self, name, desc=None, initial=None):
        """ Creates a parameter for use within the model.

            Returns: a Parameter
        """
        if name in self._vars or name in self._params:
            raise DuplicateNameError('Name already in use: ' + name)
        param = Parameter(name, desc, initial)
        self._params[name] = param
        return param

    def params(self, name):
        pass

    def rule(self, equation):
        """ Creates a rule for use within the model.

            Returns: a Rule
        """
        pass

    def solve(self, iterations=1, until=end_condition):
        """ Runs the solver.

            Parameters:
                iterations:
                until:
        """
        pass

    def solution(self):
        """ Returns the solution for the latest iteration.
        """
        pass

    def variables(self):
        pass

    def parameters(self):
        pass

    def solutions(self):
        """ Returns a list of solutions (for each iteration).
        """
        pass
