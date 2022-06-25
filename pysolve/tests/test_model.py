""" model unit tests

    Copyright (c) 2014-2022 Kenn Takara
    See LICENSE for details

"""

# pylint: disable=duplicate-code

import unittest

import numpy

from pysolve.equation import EquationError
from pysolve.model import Model, DuplicateNameError, SolutionNotFoundError
from pysolve.model import CalculationError
from pysolve.utils import round_solution, is_close


class TestModel(unittest.TestCase):
    """ Testcases for the model """
    # pylint: disable=missing-docstring,invalid-name,too-many-public-methods

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
        """ Test set_values of variables """
        model = Model()
        var_x = model.var('x', default=-1)
        self.assertEqual(-1, var_x.value)
        model.set_values({'x': 22})
        self.assertEqual(22, var_x.value)

        with self.assertRaises(ValueError):
            model.set_values({'zz': -1})

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

    def test_set_parameters(self):
        """ Test set_values of parameters """
        model = Model()
        param_a = model.param('a', default=-1)
        self.assertEqual(-1, param_a.value)
        model.set_values({'a': 22})
        self.assertEqual(22, param_a.value)

        with self.assertRaises(ValueError):
            model.set_values({'zz': -1})

    def test_set_variables_equation(self):
        """ Test set_values of variables using equations """
        model = Model()
        model.param('a', default=-1)
        model.var('x', default=0)

        model.set_values({'x': 'a+12'})
        self.assertEqual(11, model.variables['x'].value)

    def test_set_parameters_equation(self):
        """ Test set_values of parameters using equations """
        model = Model()
        model.param('a', default=-1)
        model.var('x', default=0)

        model.set_values({'a': 'x+12'})
        self.assertEqual(12, model.parameters['a'].value)

    def test_setting_multiple_variables(self):
        """ Test set_values() with multiple variables """
        model = Model()
        model.param('a', default=3)
        model.param('b', default=4)
        model.var('x', default=-1)
        model.var('y', default=-2)

        model.set_values({'a': 'a+11', 'b': '4*b'})
        self.assertEqual(14, model.parameters['a'].value)
        self.assertEqual(16, model.parameters['b'].value)

    def test_set_var_default(self):
        """ Test the set_var_default """
        model = Model()
        var = model.var('test')
        self.assertEqual(None, var.default)
        model.set_var_default(12)
        var = model.var('test2')
        self.assertEqual(12, var.default)

    def test_param_default(self):
        """ Test the set_param_default """
        model = Model()
        param = model.param('test')
        self.assertEqual(None, param.default)
        model.set_param_default(122)
        param = model.param('test2')
        self.assertEqual(122, param.default)

    def test_rule(self):
        """ Test creating rules """
        model = Model()
        model.var('x')
        model.var('y')
        eqn = model.add('x = y')

        self.assertEqual(2, len(model.variables))
        self.assertEqual(0, len(model.parameters))
        self.assertIsNotNone(model.variables['x'].equation)
        self.assertIsNone(model.variables['y'].equation)

        self.assertEqual(1, len(model.equations))
        self.assertEqual(eqn, model.equations[0])

    def test_rule_with_coefficients(self):
        """ Test creating rules with simple coefficents """
        model = Model()
        model.var('x')
        model.var('y')
        model.add('2*x = 3*y')

        self.assertEqual(2, len(model.variables))
        self.assertEqual(0, len(model.parameters))

    def test_series_get_at(self):
        """ Test the series accessor creation, get_at() """
        # pylint: disable=protected-access
        model = Model()
        varx = model.var('x')
        param = model.get_at(varx, 0)
        self.assertIsNotNone(param)
        self.assertEqual('_x_0', param.name)
        self.assertEqual(0, param.iteration)
        self.assertEqual(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

        param = model.get_at(varx, 1)
        self.assertIsNotNone(param)
        self.assertEqual('_x_1', param.name)
        self.assertEqual(1, param.iteration)
        self.assertEqual(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

        param = model.get_at(varx, -1)
        self.assertIsNotNone(param)
        self.assertEqual('_x__1', param.name)
        self.assertEqual(-1, param.iteration)
        self.assertEqual(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

        param = model.get_at(varx, 10000)
        self.assertIsNotNone(param)
        self.assertEqual('_x_10000', param.name)
        self.assertEqual(10000, param.iteration)
        self.assertEqual(varx, param.variable)
        self.assertTrue(param.name in model._private_parameters)

    def test_series_get_at_errors(self):
        """ Test bad parameters to get_at """
        model = Model()
        varx = model.var('x')
        vary = model.var('y')

        with self.assertRaises(EquationError):
            model.get_at(varx, vary)

    def test_series_derivative(self):
        model = Model()
        varx = model.var('x')
        vary = model.var('y')
        equation = model.add('x = y + x(-1)')
        df = equation.expr.diff(varx)
        self.assertEqual(0, df)
        df = equation.expr.diff(vary)
        self.assertEqual(1, df)

    def test_equation_validate(self):
        """ Test the error checking within the solve() function """
        model = Model()
        model.var('x')
        with self.assertRaises(EquationError) as context:
            model.solve()
        self.assertEqual('under-specified', context.exception.errorid)

    def test_update_solutions(self):
        """ Test _update_solutions function """
        # pylint: disable=protected-access
        model = Model()
        varx = model.var('x')
        vary = model.var('y')

        model._update_solutions({'x': 1.1, 'y': 2.2})
        self.assertEqual(1, len(model.solutions))
        self.assertEqual(1.1, varx.value)
        self.assertTrue('x' in model.solutions[0])
        self.assertEqual(1.1, model.solutions[0]['x'])
        self.assertEqual(2.2, vary.value)
        self.assertTrue('y' in model.solutions[0])
        self.assertEqual(2.2, model.solutions[0]['y'])

        model._update_solutions({'x': 3.3, 'y': 4.4})
        self.assertEqual(2, len(model.solutions))
        self.assertEqual(3.3, varx.value)
        self.assertTrue('x' in model.solutions[1])
        self.assertEqual(3.3, model.solutions[1]['x'])
        self.assertEqual(4.4, vary.value)
        self.assertTrue('y' in model.solutions[1])
        self.assertEqual(4.4, model.solutions[1]['y'])

    def test_get_value(self):
        """ Test the get_value function """
        # pylint: disable=protected-access
        model = Model()
        varx = model.var('x', default=-1)
        vary = model.var('y')

        model._update_solutions({'x': 1.1, 'y': 2.2})
        model._update_solutions({'x': 3.3, 'y': 4.4})

        self.assertEqual(1.1, model.get_value(varx, 0))
        self.assertEqual(1.1, model.get_value(varx, -2))
        self.assertEqual(4.4, model.get_value(vary, 1))
        self.assertEqual(4.4, model.get_value(vary, -1))

        with self.assertRaises(IndexError):
            self.assertEqual(-1, model.get_value(varx, -1000))

        with self.assertRaises(IndexError):
            self.assertEqual(-1, model.get_value(varx, 1000))

    # test the end condition
    # error: series accessor with non-bound variable

    def test_evaluate(self):
        """ Test arbitrary function evaluation """
        model = Model()
        model.var('x', default=1)
        model.var('y', default=10)
        model.param('a', default=.5)

        self.assertEqual(11, model.evaluate('x+y'))

    def test_evaluate_series_access(self):
        """ Test evaluation with series accessor """
        model = Model()
        model.var('x', default=1)
        model.var('y', default=10)
        model.param('a', default=.5)
        model.solutions = [{'x': 2, 'y': 11, 'a': 60}]

        self.assertEqual(11, model.evaluate('y(-1)'))
        self.assertEqual(73, model.evaluate('x(-1) + y(-1) + a(-1)'))

    def test_delta(self):
        """ test the d() function """
        model = Model()
        model.var('x', default=-1)
        model.var('y', default=10)
        model.param('a', default=.5)
        model.solutions = [{'x': 2, 'y': 11, 'a': 60}]

        model.variables['x'].value = 5
        self.assertEqual(3, model.evaluate('d(x)'))

    def test_delta_error(self):
        model = Model()
        model.var('x', default=-1)
        model.var('y', default=10)
        model.param('a', default=.5)
        model.solutions = [{'x': 2, 'y': 11, 'a': 60}]
        model.variables['x'].value = 5

        with self.assertRaises(EquationError) as context:
            model.evaluate('d(-1)')
        self.assertEqual('d-arg-not-a-variable', context.exception.errorid)

    def test_if_true(self):
        """ Test the if_true builtin function """
        model = Model()
        model.var('x', default=12)
        model.var('y', default=3131)
        self.assertEqual(0, model.evaluate('if_true(x > 1000)'))
        self.assertEqual(1, model.evaluate('if_true(y > 1000)'))

    def test_model_failure(self):
        """ Test for divergence """
        model = Model()
        model.var('x', default=1.1)
        model.var('y', default=2.3)
        model.add('2*x = 11 - 3*y')
        model.add('7*y = 13 - 5*x')

        with self.assertRaises(SolutionNotFoundError):
            model.solve(iterations=100, threshold=1e-4)

    def test_calculation_error(self):
        """ Test an error while calculating """
        model = Model()
        model.var('y', default=0)
        model.var('x', default=0)
        model.add('y = 2/x')
        model.add('x = 12')

        with self.assertRaises(CalculationError) as context:
            model.solve(iterations=10, threshold=1e-4)
        self.assertTrue(isinstance(context.exception.inner, ZeroDivisionError))

    def test_model_with_function(self):
        """ Test model with builtin function call test """
        model = Model()
        model.var('x', default=0)
        model.var('y', default=0)
        model.add('2*x = 12 - y')
        model.add('y = if_true(x > 10) + 5')

        model.solve(iterations=10, threshold=1e-4)

        self.assertEqual(2, len(model.solutions))
        self.assertEqual(0, model.solutions[0]['x'])
        self.assertEqual(0, model.solutions[0]['y'])
        self.assertEqual(3.5, model.solutions[1]['x'])
        self.assertEqual(5, model.solutions[1]['y'])

        model = Model()
        model.var('x', default=0)
        model.var('y', default=0)
        model.add('2*x = 12 + y')
        model.add('y = if_true(x > 5)')

        model.solve(iterations=10, threshold=1e-4)

        self.assertEqual(2, len(model.solutions))
        self.assertEqual(0, model.solutions[0]['x'])
        self.assertEqual(0, model.solutions[0]['y'])
        self.assertEqual(6.5, model.solutions[1]['x'])
        self.assertEqual(1, model.solutions[1]['y'])

    def test_newton_raphson(self):
        """ Test solving with Newton-Raphson, instead of the
            default Gauss-Seidel
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

        debuglist = []
        model.solve(iterations=100,
                    threshold=1e-4,
                    debuglist=debuglist,
                    method='newton-raphson')
        soln = round_solution(model.solutions[-1], decimals=1)
        print(soln)
        self.assertTrue(numpy.isclose(38.5, soln['Y']))
        self.assertTrue(numpy.isclose(7.7, soln['Ts']))
        self.assertTrue(numpy.isclose(30.8, soln['YD']))
        self.assertTrue(numpy.isclose(18.5, soln['Cs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hh']))
        self.assertTrue(numpy.isclose(0, soln['_Hs__1']))
        self.assertTrue(numpy.isclose(0, soln['_Hh__1']))

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

        model.solve(iterations=200, threshold=1e-3)
        soln = round_solution(model.solutions[-1], decimals=1)
        self.assertTrue(numpy.isclose(38.5, soln['Y']))
        self.assertTrue(numpy.isclose(7.7, soln['Ts']))
        self.assertTrue(numpy.isclose(30.8, soln['YD']))
        self.assertTrue(numpy.isclose(18.5, soln['Cs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hh']))
        self.assertTrue(numpy.isclose(0, soln['_Hs__1']))
        self.assertTrue(numpy.isclose(0, soln['_Hh__1']))

        model.solve(iterations=200, threshold=1e-3)
        soln = round_solution(model.solutions[-1], decimals=1)
        self.assertTrue(numpy.isclose(47.9, soln['Y']))
        self.assertTrue(numpy.isclose(9.6, soln['Ts']))
        self.assertTrue(numpy.isclose(38.3, soln['YD']))
        self.assertTrue(numpy.isclose(27.9, soln['Cs']))
        self.assertTrue(numpy.isclose(22.7, soln['Hs']))
        self.assertTrue(numpy.isclose(22.7, soln['Hh']))
        self.assertTrue(numpy.isclose(12.3, soln['_Hs__1']))
        self.assertTrue(numpy.isclose(12.3, soln['_Hh__1']))

        # Now run until the solutions themselves converge
        prev_soln = model.solutions[-1]
        converges = False
        for _ in range(100):
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

    def test_calling_sympy_functions(self):
        """ Test the calling of sympy functions """
        model = Model()
        model.var('x')
        model.param('a')
        model.set_values({'x': 12, 'a': 5})
        self.assertEqual(1.1, model.evaluate('Max(1, 1.1)'))

    def test_broyden(self):
        """ Test solving with Broyden's method, instead of the
            default Gauss-Seidel
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

        debuglist = []
        model.solve(iterations=100,
                    threshold=1e-4,
                    debuglist=debuglist,
                    method='broyden')
        soln = round_solution(model.solutions[-1], decimals=1)
        print(soln)
        self.assertTrue(numpy.isclose(38.5, soln['Y']))
        self.assertTrue(numpy.isclose(7.7, soln['Ts']))
        self.assertTrue(numpy.isclose(30.8, soln['YD']))
        self.assertTrue(numpy.isclose(18.5, soln['Cs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hs']))
        self.assertTrue(numpy.isclose(12.3, soln['Hh']))
        self.assertTrue(numpy.isclose(0, soln['_Hs__1']))
        self.assertTrue(numpy.isclose(0, soln['_Hh__1']))
