""" Contains the Equation class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import re

from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import factorial_notation, auto_number

from pylinsolve.variable import Variable


class EquationError(ValueError):
    """ Exception: An error in the equation specification was found

        Arguments:
            errorid: An id to identify the particular type of error
            equation: The equation input string
            text: A description of the error
    """
    def __init__(self, errorid, equation, text):
        super(EquationError, self).__init__()
        self.text = 'Error in the equation:{0} : {1} : {2}'.format(equation,
                                                                   errorid,
                                                                   text)
        self.errorid = errorid

    def __str__(self):
        return self.text


def _rewrite(variables, parameters, equation):
    """ Internal function that will do some preprocessing of the equation
        expression.
        This will convert:
            This is to allow easier access to the solution series data.
            'x(-t)' -> '__x_(-t)'

            Put the equation into our "canonical" form, 'f(x)=0'
            This only occurs if an '=' appears in the expression
            'x = y' -> 'x -(y)'
    """
    new_equation = equation

    # If variables are being called like functions, treat them
    # as if we are trying to access the time series data.
    for var in variables.keys():
        new_equation = re.sub(r"\b{0}\(".format(var),
                              "{0}(".format(Variable.series_name(var)),
                              new_equation)

    # Check for parameters that are being used like the variable
    # series accessor functions.
    for param in parameters.keys():
        if re.search(r"\b{0}\(".format(param), new_equation):
            raise EquationError('parameter-function',
                                equation,
                                'Parameters cannot access previous values: ' +
                                param)

    if '=' in new_equation:
        parts = new_equation.split('=')
        new_equation = '-('.join([x.strip() for x in parts]) + ')'
    return new_equation


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
        self._var_terms = dict()
        self._const_term = 0

    def parse(self, context):
        """ Parses the string with sympy.

            Arguments:
                context: This is a dictionary of name-symbol
                    pairs.  This is used by sympy to parse the
                    input text.

            Raises:
                EquationError
        """
        variables = self.model.variables()
        parameters = self.model.parameters()

        # Rewrite the equation into canonical form
        equation = _rewrite(variables, parameters, self.equation)

        # parse the equation
        expr = parse_expr(equation,
                          context,
                          transformations=(factorial_notation, auto_number))
        expr = expr.expand()
        self._separate_terms(expr)

    def _separate_terms(self, expr):
        """ Separate the terms into constant terms and variable terms.
            (with the coefficients broken out separately).
        """
        # pylint: disable=too-many-branches
        variables = self.model.variables()

        coeffs = expr.as_coefficients_dict()
        for key in coeffs.keys():
            if key.is_number:
                # this evaluates to a number, so no variables/parameters
                # are involved in the expression.
                self._const_term += coeffs[key]*key
            elif key.is_Symbol:
                if key.name in variables:
                    self._var_terms.setdefault(key.name, 0)
                    self._var_terms[key.name] += coeffs[key]
                elif key.name in self.model.parameters():
                    # This is not a variable, may be a constant or parameter
                    # add to the constant list
                    self._const_term += coeffs[key]*key
                else:
                    # may be a sympy constant, but not one we supplied
                    self._const_term += coeffs[key]*key
            elif key.is_Mul:
                # Check the atoms to see if there are any variables,
                # there should only be one
                atoms = [k for k in key.atoms()
                         if not k.is_number and k.name in variables]
                if len(atoms) > 1:
                    raise EquationError('not-independent',
                                        self.equation,
                                        'equations are not independent: ' +
                                        str(key))
                elif len(atoms) == 0:
                    # This is a constant term
                    self._const_term += coeffs[key]*key
                else:
                    # This is a single variable.  Need to make sure that
                    # this is not in a function.
                    coeff_mul_parts = key.as_coeff_mul(atoms[0])
                    if (len(coeff_mul_parts[1]) == 1 and
                            atoms[0] == coeff_mul_parts[1][0]):
                        var = coeff_mul_parts[1][0]
                        self._var_terms.setdefault(var.name, 0)
                        self._var_terms[var.name] += \
                            coeffs[key] * coeff_mul_parts[0]
                    else:
                        raise EquationError('non-linear',
                                            self.equation,
                                            'linear expressions only: ' +
                                            str(coeffs[key]*key))
            else:
                # This is most likely a function or operator of
                # some kind that is unexpected
                raise EquationError('unexpected-term',
                                    self.equation,
                                    'unexpected term : ' + str(key))

    def variable_terms(self):
        """ Returns a dict of the variable terms in the equation.

            The terms are seperated into tuples (coefficient:variable)
        """
        return self._var_terms

    def constant_term(self):
        """ Returns a list of the constant terms in the equation.
            These are terms that do not contain a variable.
        """
        return self._const_term
