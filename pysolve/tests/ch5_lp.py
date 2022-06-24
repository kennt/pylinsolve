""" Code for the LP model
"""
# pylint: disable=invalid-name
import time

from pysolve.model import Model
from pysolve.utils import round_solution, is_close


def create_lp_model():
    """ Creates model LP """
    # pylint: disable=too-many-statements
    model = Model()

    model.set_var_default(0)
    model.var('Bcb', desc='Government bills held by the Central Bank')
    model.var('Bd', desc='Demand for government bills')
    model.var('Bh', desc='Government bills held by households')
    model.var('Bs', desc='Government bills supplied by government')
    model.var('BLd', desc='Demand for government bonds')
    model.var('BLh', desc='Government bonds held by households')
    model.var('BLs', desc='Supply of government bonds')
    model.var('CG', desc='Capital gains on bonds')
    model.var('CGe', desc='Expected capital gains on bonds')
    model.var('C', desc='Consumption')
    model.var('ERrbl', desc='Expected rate of return on bonds')
    model.var('Hd', desc='Demand for cash')
    model.var('Hh', desc='Cash held by households')
    model.var('Hs', desc='Cash supplied by the central bank')
    model.var('Pbl', desc='Price of bonds')
    model.var('Pble', desc='Expected price of bonds')
    model.var('Rb', desc='Interest rate on government bills')
    model.var('Rbl', desc='Interest rate on government bonds')
    model.var('T', desc='Taxes')
    model.var('V', desc='Household wealth')
    model.var('Ve', desc='Expected household wealth')
    model.var('Y', desc='Income = GDP')
    model.var('YDr', desc='Regular disposable income of households')
    model.var('YDre', desc='Expected regular disposable income of households')

    model.set_param_default(0)
    model.param('alpha1', desc='Propensity to consume out of income')
    model.param('alpha2', desc='Propensit to consume out of wealth')
    model.param('chi', desc='Weight of conviction in expected bond price')
    model.param('lambda10', desc='Parameter in asset demand function')
    model.param('lambda12', desc='Parameter in asset demand function')
    model.param('lambda13', desc='Parameter in asset demand function')
    model.param('lambda14', desc='Parameter in asset demand function')
    model.param('lambda20', desc='Parameter in asset demand function')
    model.param('lambda22', desc='Parameter in asset demand function')
    model.param('lambda23', desc='Parameter in asset demand function')
    model.param('lambda24', desc='Parameter in asset demand function')
    model.param('lambda30', desc='Parameter in asset demand function')
    model.param('lambda32', desc='Parameter in asset demand function')
    model.param('lambda33', desc='Parameter in asset demand function')
    model.param('lambda34', desc='Parameter in asset demand function')
    model.param('theta', desc='Tax rate')

    model.param('G', desc='Government goods')
    model.param('Rbar', desc='Exogenously set interest rate on govt bills')
    model.param('Pblbar', desc='Exogenously set price of bonds')

    model.add('Y = C + G')                                  # 5.1
    model.add('YDr = Y - T + Rb(-1)*Bh(-1) + BLh(-1)')      # 5.2
    model.add('T = theta *(Y + Rb(-1)*Bh(-1) + BLh(-1))')    # 5.3
    model.add('V - V(-1) = (YDr - C) + CG')                 # 5.4
    model.add('CG = (Pbl - Pbl(-1))*BLh(-1)')
    model.add('C = alpha1*YDre + alpha2*V(-1)')
    model.add('Ve = V(-1) + (YDre - C) + CG')
    model.add('Hh = V - Bh - Pbl*BLh')
    model.add('Hd = Ve - Bd - Pbl*BLd')
    model.add('Bd = Ve*lambda20 + Ve*lambda22*Rb - ' +
              'Ve*lambda23*ERrbl - lambda24*YDre')
    model.add('BLd = (Ve*lambda30 - Ve*lambda32*Rb ' +
              ' + Ve*lambda33*ERrbl - lambda34*YDre)/Pbl')
    model.add('Bh = Bd')
    model.add('BLh = BLd')
    model.add('Bs - Bs(-1) = (G + Rb(-1)*Bs(-1) + ' +
              'BLs(-1)) - (T + Rb(-1)*Bcb(-1)) - (BLs - BLs(-1))*Pbl')
    model.add('Hs - Hs(-1) = Bcb - Bcb(-1)')
    model.add('Bcb = Bs - Bh')
    model.add('BLs = BLh')
    model.add('ERrbl = Rbl + chi * (Pble - Pbl) / Pbl')
    model.add('Rbl = 1./Pbl')
    model.add('Pble = Pbl')
    model.add('CGe = chi * (Pble - Pbl)*BLh')
    model.add('YDre = YDr(-1)')
    model.add('Rb = Rbar')
    model.add('Pbl = Pblbar')

    # if_true(x) returns 1 if x is true, else 0 is returned
    model.add('z1 = if_true(tp > top)')
    model.add('z2 = if_true(tp < bot)')
    return model


start = time.monotonic()

sim = create_lp_model()
sim.set_values({'V': 95.803,
                'Bh': 37.839,
                'Bs': 57.964,
                'Bcb': 57.964 - 37.839,
                'BLh': 1.892,
                'BLs': 1.892,
                'Hs': 20.125,
                'YDr': 95.803,
                'Rb': 0.03,
                'Pbl': 20})
sim.set_values({'alpha1': 0.8,
                'alpha2': 0.2,
                'chi': 0.1,
                'lambda20': 0.44196,
                'lambda22': 1.1,
                'lambda23': 1,
                'lambda24': 0.03,
                'lambda30': 0.3997,
                'lambda32': 1,
                'lambda33': 1.1,
                'lambda34': 0.03,
                'theta': 0.1938})
sim.set_values({'G': 20,
                'Rbar': 0.03,
                'Pblbar': 20})

for _ in range(100):
    sim.solve(iterations=100, threshold=1e-5)

    prev_soln = sim.solutions[-2]
    soln = sim.solutions[-1]
    if is_close(prev_soln, soln, atol=1e-3):
        break

end = time.monotonic()
print("elapsed time = " + str(end-start))

print(round_solution(sim.solutions[-1], decimals=1))
