""" Contains utility functions

    Copyright (c) 2014-2022 Kenn Takara
    See LICENSE for details

"""

import numpy


def round_solution(soln, decimals=4):
    """ Runs through a dict() and rounds the values.

        Arguments:
            soln: The values in this dict() will be rounded using
                numpy.round().
            decimals: The number of decimals places to the right to
                be rounded.

        Returns: A new dict() that contains the rounded values.
    """
    new_soln = {}
    for key in soln.keys():
        new_soln[key] = numpy.round(soln[key], decimals=decimals)
    return new_soln


def is_aclose(prev, curr, atol=1e-4, rtol=1e-4):
    """ Determines if the values within two dicts() are
        close.  Uses numpy.isclose()

        Arguments:
            prev: previous iteration dict()
            curr: current iteration dict()
            atol: absolute tolerance
            rtol: relative tolerance

        Returns: True if the values of the dict() are within
            the tolerances.  False otherwise.
    """
    aprev = numpy.array(prev)
    acurr = numpy.array(curr)
    return numpy.allclose(aprev, acurr, atol=atol, rtol=rtol)


def is_close(prev, curr, atol=1e-4, rtol=1e-4):
    """ Determines if the values within two dicts() are
        close.  Uses numpy.isclose()

        Arguments:
            prev: previous iteration dict()
            curr: current iteration dict()
            atol: absolute tolerance
            rtol: relative tolerance

        Returns: True if the values of the dict() are within
            the tolerances.  False otherwise.
    """
    for k in prev.keys():
        if not numpy.isclose(prev[k], curr[k], atol=atol, rtol=rtol):
            return False
    return True


def generate_html_table(header, adata):
    """ Generates an html table for use within iPython """

    def _add_row(rowtype, rowdata):
        """ Adds HTML for a single row """
        sdata = "<tr>"
        for data in rowdata:
            sdata += f"<{rowtype}>{data}</{rowtype}>"
        sdata += "</tr>"
        return sdata

    shtml = "<table>"
    if header is not None:
        shtml += _add_row("th", header)
    for row in adata:
        shtml += _add_row("td", row)
    shtml += "</table>"
    return shtml
