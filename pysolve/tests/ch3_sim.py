""" Code for the SIM model
"""

# pylint: disable=invalid-name, duplicate-code

import time

from pysolve.model import Model
from pysolve.utils import round_solution, is_close


def create_model():
    """ Creates model SIM """
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
    model.param('Gd', desc='Government goods, demand', default=20)
    model.param('W', desc='Wage rate', default=1)
    model.param('alpha1',
                desc='Propensity to consume out of income',
                default=0.6)
    model.param('alpha2',
                desc='Propensity to consume o of wealth',
                default=0.4)
    model.param('theta', desc='Tax rate', default=0.2)

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

    return model


start = time.monotonic()

sim = create_model()

for _ in range(100):
    sim.solve(iterations=100, threshold=1e-5)

    prev_soln = sim.solutions[-2]
    soln = sim.solutions[-1]
    if is_close(prev_soln, soln, atol=1e-3):
        break

end = time.monotonic()
print("elapsed time = " + str(end-start))

print(round_solution(sim.solutions[-1], decimals=1))
