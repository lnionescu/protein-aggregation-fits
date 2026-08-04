'''
Normal global fitting, but with different ko_minus values below vs. above m_star.
-> this requires using the numerical solution, since we have no closed-form for it
'''
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits')
from richards import load_data
from scipy.optimize import basinhopping, brentq
from dataclasses import dataclass

def qss_free_monomer(m0_val, m_star_val, nk_val, n_val):
    '''
    For a fairer comparison between analytical and numerical solutions, we solve the initial cond    ition on the free monomer explicitly (in the closed-form solution, this is hidden away inside    the combined chi).
    m_free + n * m_star^(1-nk) * m_free^nk = 0
    '''
    if m0_val <= m_star_val:
        return m0_val   # below CMC: all protein as free monomer, no oligomers
    prefactor = n_val * m_star_val ** (1 - nk_val)   # = n * ko_plus / ko_minus
    def f(m_free):
        return m_free + prefactor * m_free ** nk_val - m0_val
    # root is guaranteed in (0, m0); solution < m_star for large nk
    return brentq(f, 1e-14, m0_val * (1 - 1e-12))


@dataclass
class KineticParameters:
    k1: float
    k2: float
    kp: float
    ko_minus: float
    nc: float
    n2: float
    nk: float
    n: float
    m_star: float

def model_analytical(t, m0, p: KineticParameters):
    ko_plus = p.ko_minus * p.m_star ** (1 - p.nk)
    m_free_qss = qss_free_monomer(m0, p.m_star, p.nk, p.n)
    Omega = m_free_qss / m0   # exact QSS, not the nk->inf approximation m_star/m0

    kappa_val = np.sqrt(2 * p.kp * m0 * p.k2 * m0 ** p.n2)
    eps_val = p.k1 * m0 ** p.nc / (2 * m0 * p.k2 * m0 ** p.n2)
    lam = np.sqrt(Omega ** (p.n2 + 1))
    Gamma = p.ko_minus / kappa_val

    #c = Gamma / lam if Gamma < 1 else 3 / (2 * p.n2 + 1)
    c = Gamma / lam

    tau = kappa_val * t
    arg = np.clip(lam * tau, 0, 500)    # avoid overflow in analytical solution
    return 1 - (1 + 2 * eps_val * Omega ** (p.nc - p.n2) / c * (np.cosh(arg) - 1)) ** (-c)


def minimize_diff_ko_minuses(guess, x_actual, y_actual, m0vals, init_guesses, free_params, fixed_params):
    fitted = {name: np.exp(guess[i]) * init_guesses[i] for i, name in enumerate(free_params)}
    p = {**fixed_params, **fitted}
    m_star = p['m_star']

    residual = 0.0
    for i in range(len(x_actual)):
        m0 = m0vals[i]
        ko_minus_here = p['ko_above'] if m0 >= m_star else p['ko_below']
        params = KineticParameters(k1 = p['k1'], k2 = p['k2'], kp = p['kp'], ko_minus = ko_minus_here, nc = p['nc'], n2=p['n2'], nk = p['nk'], n = p['n'], m_star=m_star)
        y_model = model_analytical(x_actual[i], m0, params)
        residual += np.sum((y_actual[i] - y_model) ** 2)
    return residual


#======================
# DATA LOAD AND CLEANUP
#======================



# use for old pH 6 data:
data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_proper_norm.tsv'
case = 'pH_6_old'

# use for old pH 6.5 data:
#data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv'
#case = 'pH_6.5_old'

# use for new pH 6.5 data:
#data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6.5_june.tsv'
#case = 'pH_6.5_new'



x_data, y_data, m0vals = load_data(data_path)

if case == 'pH_6_old':
    mask = m0vals != 0.8
    x_data = [x for x, m0 in zip(x_data, m0vals) if m0 != 0.8]
    y_data = [y for y, m0 in zip(y_data, m0vals) if m0 != 0.8]
    m0vals = m0vals[mask]
    drop_idx = np.where(m0vals == 1.4)[0][2]
    keep = np.arange(len(m0vals)) != drop_idx
    x_data = [x for x, k in zip(x_data, keep) if k]
    y_data = [y for y, k in zip(y_data, keep) if k]
    m0vals = m0vals[keep]

elif case == 'pH_6.5_new':
    mask = m0vals != 1.25
    x_data = [x for x, m0 in zip(x_data, m0vals) if m0 != 1.25]
    y_data = [y for y, m0 in zip(y_data, m0vals) if m0 != 1.25]
    m0vals = m0vals[mask]
    drop_idx = np.where(m0vals == 1.75)[0][0]
    keep = np.arange(len(m0vals)) != drop_idx
    x_data = [x for x, k in zip(x_data, keep) if k]
    y_data = [y for y, k in zip(y_data, keep) if k]
    m0vals = m0vals[keep]
    drop_idx = np.where(m0vals == 2.5)[0][2]
    keep = np.arange(len(m0vals)) != drop_idx
    x_data = [x for x, k in zip(x_data, keep) if k]
    y_data = [y for y, k in zip(y_data, keep) if k]
    m0vals = m0vals[keep]

elif case == 'pH_6.5_old':
    pass     # file is already cleaned up

print('currently fitting: ', case)


#=============
# MAIN
#=============
if __name__ == '__main__':

