""" Contains the Equation class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import sympy



class EquationError(ValueError):
    """ Exception: An error in the equation specification was found 

        Arguments:
            id: An id to identify the particular type of error
            equation: The equation input string
            text: A description of the error
    """
    def __init__(self, id, equation, text):
        super(EquationError, self).__init__()
        self.text = 'Error in the equation:{0} : {1} : {2}'.format(equation,
                                                                   id,
                                                                   text)

    def __str__(self):
        return self.text



class Equation(object):
    """ This class contains an 'equation'.

        Attributes:
            equation: The original equation string
            desc:
    """
    def __init__(self, equation, desc=None):
        self.equation = equation
        self.desc = desc
        self.model = None

    def _validate(self):
        # validate the current expression
        # can only have one variable per term
        # all symbols must be predefined, variables, or parameters
        # variables must be linear
        pass

    def parse(self, context):
        # parse the equation
        # validate
        # gather constant terms
        # separate variable terms from constant terms
        # for variable terms determine the coefficients
        pass

    def variable_terms(self):
        """ Returns a list of the variable terms in the equation.

            The terms are seperated into tuples (coefficient, variable)
        """
        pass

    def constant_terms(self):
        """ Returns a list of the constant terms in the equation.
            These are terms that do not contain a variable.
        """
        pass
