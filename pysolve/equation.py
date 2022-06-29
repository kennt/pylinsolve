""" Contains the Equation class.

    Copyright (c) 2014-2022 Kenn Takara
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
        super().__init__()
        self.text = f'Error in the equation:{equation} : {errorid} : {text}'
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
    """
    new_equation = equation

    # If variables are being called like functions, treat them
    # as if we are trying to access the time series data.
    for var in variables.keys():
        new_equation = re.sub(rf"\b{var}\(",
                              f"_series_acc({var},",
                              new_equation)

    for param in parameters.keys():
        new_equation = re.sub(rf"\b{param}\(",
                              f"_series_acc({param},",
                              new_equation)

    return new_equation


def _is_linear(expr, var):
    """ Returns true if the equation is linear in the variable.
        False otherwise
    """
    poly = expr.as_poly(var)
    if poly:
        return poly.degree() == 1
    return False


class Equation:
    """ This class contains an 'equation'.

        An equation is of the form "x = f(...)".  The left-hand side
        contains the variable that is being solved for.  The right-hand
        side will be evaluated to give a value for that variable.

        The actual form of the equation is such that there can only be
        a single variable on the left hand side (although constants and
        parameters may also be used).

        Examples:
            x - x(-1) = .....
            10*alpha1*x = ......

        The equation will then be parsed and transformed into the
        canonical form
            x = f(....)

        Attributes:
            equation: The original equation string
            desc: A description of the equation.  Used for docs.
            model: A reference back to the model that contains the
                proper context, that is the equation will be evaluated
                within the context of this model.
            expr: This is the "value" of the equation. This is a sympy
                expression that will need to be evaluated within the
                proper context to return a value.
            func: The 'lambdified' version of expr (for perf)
            variable: The variable that this equation defines.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, model, equation, desc=None):
        self.equation = equation
        self.desc = desc
        self.model = model
        self.expr = None
        self.func = None
        self.variable = None

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

        # Find the location of the equal sign, also need to ignore
        # things like <=, >=, ==, !=, etc...
        sides = re.split(r"(?<![<>=!])=(?!=)", equation)
        if len(sides) != 2:
            raise EquationError('equals-sign',
                                self.equation,
                                "Equation must be of the form f(...) = g(...)")

        transforms = (factorial_notation, auto_number)

        # need to examine the left-hand side
        rhs = parse_expr(sides[1], context, transformations=transforms)

        lhs = parse_expr(sides[0], context, transformations=transforms)
        lhs = lhs.expand()

        # Determine how to isolate the variable by addition/division
        variable, add_terms, mul_terms = self._isolate_variable(lhs)

        if not _is_linear(lhs, variable):
            raise EquationError('non-linear',
                                self.equation,
                                'the main variable is not linear')

        if variable is not None and variable.equation is not None:
            raise EquationError('var-eqn-exists',
                                self.equation,
                                'equation for variable already defined : ' +
                                variable.name)
        if add_terms is not None:
            rhs = rhs - add_terms
        if mul_terms is not None:
            rhs = rhs / mul_terms

        self.variable = variable
        self.expr = rhs
        variable.equation = self

    def _isolate_variable(self, expr):
        """ This will isolate the expr so that it only contains
            a single variable.

            original: 34*x + 99 = y
            thus: 34*x + 99
            will return (99, 34)
            and we will then modify the rhs to do (y-99)/34

            Returns: a tuple containing the additive factor and the
                multiplicative factor for the left-hand side.
        """
        variables = self.model.variables

        # first, there must only be one variable
        atoms = [k for k in expr.atoms()
                 if not k.is_number and k.name in variables]
        if len(atoms) != 1:
            raise EquationError('lhs-variables',
                                self.equation,
                                'The left-hand side must have one variable')
        variable = atoms[0]
        mul_term = expr.coeff(variable, 1)
        add_term = 0
        for term in expr.as_ordered_terms():
            if variable not in term.atoms():
                add_term += term
        return (variable, add_term, mul_term)
