""" Contains the Equation class.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import re

from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import factorial_notation, auto_number


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
            'x(-t)' -> '_series_acc(x, _iter-t)'
            We translate this into a function so that we can evaluate
            the parameter symbolically before the call.

            Put the equation into our "canonical" form, 'f(x)=0'
            This only occurs if an '=' appears in the expression
            'x = y' -> 'x -(y)'
    """
    new_equation = equation

    # If variables are being called like functions, treat them
    # as if we are trying to access the time series data.
    for var in variables.keys():
        new_equation = re.sub(r"\b{0}\(".format(var),
                              "_series_acc({0},".format(var),
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
        variables = self.model.variables
        parameters = self.model.parameters

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
        variables = self.model.variables

        # need to search the equation string manually
        # sympy may reorder the terms
        for term in expr.as_ordered_terms():
            if term.is_number:
                self._const_term += term
            elif term.is_Symbol:
                if term.name in variables:
                    self._var_terms.setdefault(term.name, 0)
                    self._var_terms[term.name] += 1
                else:
                    # may be a parameter or sympy-supplied symbol
                    self._const_term += term
            elif term.is_Mul:
                atoms = [k for k in term.atoms()
                         if not k.is_number and k.name in variables]
                if len(atoms) > 1:
                    raise EquationError('not-independent',
                                        self.equation,
                                        'equations are not independent: ' +
                                        str(term))
                elif len(atoms) == 0:
                    # This is a constant term
                    self._const_term += term
                else:
                    # There is a single variable in here.
                    var = None
                    coeff = 1
                    for arg in term.args:
                        if arg.is_Symbol and arg.name in variables:
                            var = arg
                        else:
                            coeff *= arg

                    if var is None:
                        # could not find a single variable, it may
                        # be non-linear
                        raise EquationError('non-linear',
                                            self.equation,
                                            'linear expressions only: ' +
                                            str(term))
                    self._var_terms.setdefault(var.name, 0)
                    self._var_terms[var.name] += coeff
            else:
                # This is most likely a function or operator of
                # some kind that is unexpected
                raise EquationError('unexpected-term',
                                    self.equation,
                                    'unexpected term : ' + str(term))

        first_var = None
        for var in variables.values():
            if var.equation is None and var.name in self._var_terms:
                var.equation = self
                first_var = var
                break
        if first_var is None:
            raise EquationError('no-variable',
                                self.equation,
                                'Equation may not contain a variable ' +
                                'or the system may be overspecified')

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
