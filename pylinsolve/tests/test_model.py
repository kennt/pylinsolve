""" model unit tests

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""

import unittest

import numpy

from pylinsolve.equation import EquationError
from pylinsolve.model import Model, DuplicateNameError, SolutionNotFoundError
from pylinsolve.model import _run_solver
from pylinsolve.utils import round_solution, is_close


class TestModel(unittest.TestCase):
    """ Testcases for the model """
    # pylint: disable=missing-docstring,invalid-name

    def test_model(self):
        """ Create an empty model class """
        model = Model()
        self.assertIsNotNone(model)

    def test_var(self):
        """ Test variable creation """
        model = Model()
        var_x = model.var('x')
        self.assertIsNotNone(var_x)
        with self.assertRaises(DuplicateNameError):
            model.var('x')

        with self.assertRaises(DuplicateNameError):
            model.param('x')

        self.assertTrue('x' in model.variables)
        self.assertTrue('x' not in model.parameters)

    def test_set_variables(self):
        """ Test set_variables() """
        model = Model()
        var_x = model.var('x', default=-1)
        self.assertEquals(-1, var_x.value)
        model.set_variables({'x': 22})
        self.assertEquals(22, var_x.value)

    def test_param(self):
        """ Test parameter creation """
        model = Model()
        param_x = model.param('x')
        self.assertIsNotNone(param_x)
        with self.assertRaises(DuplicateNameError):
            model.var('x')

        with self.assertRaises(DuplicateNameError):
            model.param('x')

        self.assertTrue('x' not in model.variables)
        self.assertTrue('x' in model.parameters)

    def test_set_var_default(self):
        """ Test the set_var_default """
        model = Model()
        var = model.var('test')
        self.assertEquals(None, var.default)
        model.set_var_default(12)
        var = model.var('test2')
        self.assertEquals(12, var.default)

    def test_param_default(self):
        """ Test the set_param_default """
        model = Model()
        param = model.param('test')
        self.assertEquals(None, param.default)
        model.set_param_default(122)
        param = model.param('test2')
        self.assertEquals(122, param.default)

    def test_rule(self):
        """ Test creating rules """
        model = Model()
        model.var('x')
        model.var('y')
        model.add('x = y')

        self.assertEquals(2, len(model.variables))
        self.assertEquals(0, len(model.parameters))
        self.assertEquals(1, len(model.equations))

        eqn = model.equations[0]
        self.assertEquals(2, len(eqn.variable_terms()))
        self.assertTrue('x' in eqn.variable_terms())
        self.assertEquals(1, eqn.variable_terms()['x'])
        self.assertTrue('y' in eqn.variable_terms())
        self.assertEquals(-1, eqn.variable_terms()['y'])

    def test_rule_with_coefficients(self):
        """ Test creating rules with simple coefficents """
        model = Model()
        model.var('x')
        model.var('y')
        model.add('2*x + 3*y')

        self.assertEquals(2, len(model.variables))
        self.assertEquals(0, len(model.parameters))
        self.assertEquals(1, len(model.equations))

        eqn = model.equations[0]
        self.assertEquals(2, len(eqn.variable_terms()))
        self.assertTrue('x' in eqn.variable_terms())
        self.assertEquals(2, eqn.variable_terms()['x'])
        self.assertTrue('y' in eqn.variable_terms())
        self.assertEquals(3, eqn.variable_terms()['y'])

    def test_series_get_at(self):
        """ Test the series accessor creation, get_at() """
        model = Model()
        varx = model.var('x')
        param = model.get_at(varx, 0)
        self.assertIsNotNone(param)
        self.assertEquals('_x_0', param.name)
        self.assertEquals(0, param.iteration)
        self.assertEquals(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

        param = model.get_at(varx, 1)
        self.assertIsNotNone(param)
        self.assertEquals('_x_1', param.name)
        self.assertEquals(1, param.iteration)
        self.assertEquals(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

        param = model.get_at(varx, -1)
        self.assertIsNotNone(param)
        self.assertEquals('_x__1', param.name)
        self.assertEquals(-1, param.iteration)
        self.assertEquals(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

        param = model.get_at(varx, 10000)
        self.assertIsNotNone(param)
        self.assertEquals('_x_10000', param.name)
        self.assertEquals(10000, param.iteration)
        self.assertEquals(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

    def test_series_get_at_errors(self):
        """ Test bad parameters to get_at """
        model = Model()
        varx = model.var('x')
        vary = model.var('y')

        with self.assertRaises(EquationError):
            model.get_at(varx, vary)

    def test_run_solver(self):
        """ Simple tests to see if the solver works """
        A = numpy.array([[10., -1., 2., 0.],
                         [-1., 11., -1., 3.],
                         [2., -1., 10., -1.],
                         [0., 3., -1., 8.]])
        b = numpy.array([6., 25., -11., 15.])
        x = numpy.array([1., 1., 1., 1.])

        debuglist = list()
        soln = _run_solver(A, x, b, threshold=1e-4, debuglist=debuglist)
        self.assertIsNotNone(soln)
        self.assertTrue(numpy.isclose(1., soln[0]))
        self.assertTrue(numpy.isclose(2., soln[1]))
        self.assertTrue(numpy.isclose(-1., soln[2]))
        self.assertTrue(numpy.isclose(1., soln[3]))

        A = numpy.array([[16., 3.],
                         [7., -11.]])
        b = numpy.array([11., 13.])
        x = numpy.array([1., 1.])
        debuglist = list()
        soln = _run_solver(A, x, b, threshold=1e-5, debuglist=debuglist)

        # Need to round the values up to 4 decimal places
        soln = numpy.around(soln, decimals=4)

        self.assertIsNotNone(soln)
        self.assertTrue(numpy.isclose(0.8122, soln[0]))
        self.assertTrue(numpy.isclose(-0.6650, soln[1]))

    def test_run_solver_failure(self):
        """ Test a case where the algorithm diverges """
        A = numpy.array([[2., 3.],
                         [5., 7.]])
        b = numpy.array([11., 13.])
        x = numpy.array([1.1, 2.3])
        debuglist = list()
        with self.assertRaises(SolutionNotFoundError):
            _run_solver(A, x, b, threshold=1e-4, debuglist=debuglist)

    def test_equation_validate(self):
        """ Test the error checking within the solve() function """
        model = Model()
        model.var('x')
        with self.assertRaises(EquationError) as context:
            model.solve()
        self.assertEquals('under-specified', context.exception.errorid)

    def test_latest_solution_vector(self):
        """ Test that we grab the latest solution vector correctly """
        model = Model()
        varx = model.var('x', default=22)
        vary = model.var('y', default=33)
        varz = model.var('z', default=1024)
        self.assertEquals(22, varx.value)
        self.assertEquals(33, vary.value)
        self.assertEquals(1024, varz.value)
        soln = model._latest_solution_vector()

        # the solution vector follows the order the variables
        # were added
        self.assertEquals(22, soln[0])
        self.assertEquals(33, soln[1])
        self.assertEquals(1024, soln[2])

    def test_prepare_solver(self):
        """ Test the _prepare_solver() function """
        model = Model()
        model.var('x')
        model.var('y')
        model.add('x - 3*y = 5')
        model.add('2*x + y = 22')
        A, b = model._prepare_solver()
        self.assertEquals(1, A[0, 0])
        self.assertEquals(-3, A[0, 1])
        self.assertEquals(2, A[1, 0])
        self.assertEquals(1, A[1, 1])
        self.assertEquals(5, b[0])
        self.assertEquals(22, b[1])

        model = Model()
        model.var('x')
        model.var('y')
        model.add('x = 3*y + 5')
        model.add('y = 2*x - 22')
        A, b = model._prepare_solver()
        self.assertEquals(1, A[0, 0])
        self.assertEquals(-3, A[0, 1])
        self.assertEquals(-2, A[1, 0])
        self.assertEquals(1, A[1, 1])
        self.assertEquals(5, b[0])
        self.assertEquals(-22, b[1])

    def test_update_solutions(self):
        """ Test _update_solutions function """
        model = Model()
        varx = model.var('x')
        vary = model.var('y')

        model._update_solutions(numpy.array([1.1, 2.2]))
        self.assertEquals(1, len(model.solutions))
        self.assertEquals(1.1, varx.value)
        self.assertTrue('x' in model.solutions[0])
        self.assertEquals(1.1, model.solutions[0]['x'])
        self.assertEquals(2.2, vary.value)
        self.assertTrue('y' in model.solutions[0])
        self.assertEquals(2.2, model.solutions[0]['y'])

        model._update_solutions(numpy.array([3.3, 4.4]))
        self.assertEquals(2, len(model.solutions))
        self.assertEquals(3.3, varx.value)
        self.assertTrue('x' in model.solutions[1])
        self.assertEquals(3.3, model.solutions[1]['x'])
        self.assertEquals(4.4, vary.value)
        self.assertTrue('y' in model.solutions[1])
        self.assertEquals(4.4, model.solutions[1]['y'])

    def test_get_value(self):
        """ Test the get_value function """
        model = Model()
        varx = model.var('x', default=-1)
        vary = model.var('y')

        model._update_solutions(numpy.array([1.1, 2.2]))
        model._update_solutions(numpy.array([3.3, 4.4]))

        self.assertEquals(1.1, model.get_value(varx, 0))
        self.assertEquals(1.1, model.get_value(varx, -2))
        self.assertEquals(4.4, model.get_value(vary, 1))
        self.assertEquals(4.4, model.get_value(vary, -1))

        with self.assertRaises(IndexError):
            self.assertEquals(-1, model.get_value(varx, -1000))

        with self.assertRaises(IndexError):
            self.assertEquals(-1, model.get_value(varx, 1000))

    def test_parameter_eval(self):
        """ Test evaluation of parameters before solving """
        model = Model()
        model.var('x')
        model.var('y')
        model.param('a', default=1.2)
        model.param('b', default=-2.3)
        model.add('x = a*y + 5')
        model.add('y = b*x + 10')
        A, b = model._prepare_solver()
        self.assertEquals(1, A[0, 0])
        self.assertEquals(-1.2, A[0, 1])
        self.assertEquals(2.3, A[1, 0])
        self.assertEquals(1, A[1, 1])
        self.assertEquals(5, b[0])
        self.assertEquals(10, b[1])

    def test_series_parameter_eval(self):
        """ Test evaluation of series parameter values before solving """
        model = Model()
        model.var('x')
        model.var('y')
        model.param('a', default=1.2)
        model.param('b', default=-2.3)
        model.add('x - x(-1) = a*y + 5')
        model.add('y = b*x + 10')

        # add some dummy solutions
        model._update_solutions(numpy.array([1.1, 2.2]))
        model._update_solutions(numpy.array([3.3, 4.4]))

        A, b = model._prepare_solver()
        self.assertEquals(1, A[0, 0])
        self.assertEquals(-1.2, A[0, 1])
        self.assertEquals(2.3, A[1, 0])
        self.assertEquals(1, A[1, 1])
        self.assertEquals(8.3, b[0])
        self.assertEquals(10, b[1])

        model._update_solutions(numpy.array([1., 2]))

        A, b = model._prepare_solver()
        self.assertEquals(1, A[0, 0])
        self.assertEquals(-1.2, A[0, 1])
        self.assertEquals(2.3, A[1, 0])
        self.assertEquals(1, A[1, 1])
        self.assertEquals(6., b[0])
        self.assertEquals(10, b[1])

    # test the end condition
    # error: series accessor with non-bound variable

    def test_evaluate(self):
        """ Test arbitrary function evaluation """
        model = Model()
        model.var('x', default=1)
        model.var('y', default=10)
        model.param('a', default=.5)

        self.assertEquals(11, model.evaluate('x+y'))

    def test_evaluate_series_access(self):
        """ Test evaluation with series accessor """
        model = Model()
        model.var('x', default=1)
        model.var('y', default=10)
        model.param('a', default=.5)
        model.solutions = [{'x': 2, 'y': 11, 'a': 60}]

        self.assertEquals(11, model.evaluate('y(-1)'))
        self.assertEquals(73, model.evaluate('x(-1) + y(-1) + a(-1)'))

    def test_full_model(self):
        """ Test by implementing a model

            This model is taken from the book
                Monetary Economics 2ed, Godley and Lavoie, 2012
            Chapter 3, The Simplest Model wtih Government Money
            Model SIM
        """
        # pylint: disable=too-many-statements
        model = Model()
        model.set_var_default(0)
        model.vars('Y', 'YD', 'Ts', 'Td', 'Hs', 'Hh', 'Gs', 'Cs',
                   'Cd', 'Ns', 'Nd')
        model.set_param_default(0)
        Gd = model.param('Gd')
        W = model.param('W')
        alpha1 = model.param('alpha1')
        alpha2 = model.param('alpha2')
        theta = model.param('theta')

        model.add('Cs = Cd')
        model.add('Gs = Gd')
        model.add('Ts = Td')
        model.add('Ns = Nd')
        model.add('YD = (W*Ns) - Ts')
        model.add('Td = theta * W * Ns')
        model.add('Cd = alpha1*YD + alpha2*Hh(-1)')
        model.add('Hs - Hs(-1) =  Gd - Td')
        model.add('Hh - Hh(-1) = YD - Cd')
        model.add('Y = Cs + Gs')
        model.add('Nd = Y/W')

        # setup default parameter values
        Gd.value = 20.
        W.value = 1.0
        alpha1.value = 0.6
        alpha2.value = 0.4
        theta.value = 0.2

        model.solve(iterations=100, threshold=1e-3)
        soln = round_solution(model.solutions[-1], decimals=1)
        self.assertTrue(numpy.isclose(38.5, soln['Y']))
        self.assertTrue(numpy.isclose(7.7, soln['Ts']))
        self.assertTrue(numpy.isclose(30.8, soln['YD']))
        self.assertTrue(numpy.isclose(18.5, soln['Cs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hh']))

        model.solve(iterations=100, threshold=1e-3)
        soln = round_solution(model.solutions[-1], decimals=1)
        self.assertTrue(numpy.isclose(47.9, soln['Y']))
        self.assertTrue(numpy.isclose(9.6, soln['Ts']))
        self.assertTrue(numpy.isclose(38.3, soln['YD']))
        self.assertTrue(numpy.isclose(27.9, soln['Cs']))
        self.assertTrue(numpy.isclose(22.7, soln['Hs']))
        self.assertTrue(numpy.isclose(22.7, soln['Hh']))

        # Now run until the solutions themselves converge
        prev_soln = model.solutions[-1]
        converges = False
        for _ in xrange(100):
            model.solve(iterations=100, threshold=1e-3)

            # run until we converge
            soln = model.solutions[-1]
            if is_close(prev_soln, soln, atol=1e-3):
                converges = True
                break
            prev_soln = soln

        self.assertTrue(converges)
        prev = round_solution(model.solutions[-2], decimals=1)
        soln = round_solution(model.solutions[-1], decimals=1)
        self.assertTrue(numpy.isclose(100, soln['Y']))
        self.assertTrue(numpy.isclose(20, soln['Ts']))
        self.assertTrue(numpy.isclose(80, soln['YD']))
        self.assertTrue(numpy.isclose(80, soln['Cs']))
        self.assertTrue(numpy.isclose(0, soln['Hs'] - prev['Hs']))
        self.assertTrue(numpy.isclose(0, soln['Hh'] - prev['Hh']))
