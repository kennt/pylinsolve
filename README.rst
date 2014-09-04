==========
pysolve
==========
Solving systems of linear equations
-----------------------------------

The purpose of this code is to aid in expressing and solving
sets of equations using Python.

This tool will take a textual description of the equations 
and then run the solver iteratively until it converges to a solution.

The solver uses Gauss-Seidel/SOR to iterate to a solution.  
It also uses parts of sympy to aid in parsing the equations.

The initial motivation for this tool was to solve economic
models based on Stock Flow Consistent (SFC) models.

Example usage
-------------

.. code::
	from pysolve.model import Model
	from pysolve.utils import round_solution,is_close

	model = Model()

	model.set_var_default(0)
	model.var('Cd', desc='Consumption goods demand by households')
	model.var('Cs', desc='Consumption goods supply')
	model.var('Gs', desc='Government goods, supply')
	model.var('Hh', desc='Cash money held by households')
	model.var('Hs', desc='Cash money supplied by the government')
	model.var('Nd', desc='Demand for labor')
	model.var('Ns', desc='Supply of labor')
	model.var('Td', desc='Taxes, demand')
	model.var('Ts', desc='Taxes, supply')
	model.var('Y', desc='Income = GDP')
	model.var('YD', desc='Disposable income of households')	

	# This is a shorter way to declare multiple variables
	# model.vars('Y', 'YD', 'Ts', 'Td', 'Hs', 'Hh', 'Gs', 'Cs',
	#            'Cd', 'Ns', 'Nd')
	model.param('Gd', desc='Government goods, demand', initial=20)
	model.param('W', desc='Wage rate', initial=1)
	model.param('alpha1', desc='Propensity to consume out of income', initial=0.6)
	model.param('alpha2', desc='Propensity to consume o of wealth', initial=0.4)
	model.param('theta', desc='Tax rate', initial=0.2)

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

	# solve until convergence
	for _ in xrange(100):
	    model.solve(iterations=100, threshold=1e-3)

	    prev_soln = model.solutions[-2]
	    soln = model.solutions[-1]
	    if is_close(prev_soln, soln, atol=1e-3):
	        break

	print round_solution(model.solutions[-1], decimals=1)

For additional examples, view the iPython notebooks at



