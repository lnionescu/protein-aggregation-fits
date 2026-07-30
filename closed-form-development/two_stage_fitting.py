'''
Problem with full global fitting: the tail which is (~)completely controlled by ko_minus can trade off with the other fitted parameters, i.e. k1 and k2.
On the other hand, the early time data is unaffected by ko_minus, so k1 and k2 can be better constrained from that alone. Once these are fixed, we can fit ko_minus to the whole curve.
Also, similarly to Ana's project on boundary layer kinks, we can in fact read off ko_minus directly off the late-time slope of the (log) data.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import basinhopping, brentq
from dataclass import dataclass

def qss_free_monomer(m0_val, m_star_val, nk_val, n_val):
    '''
    polynomial for initial free monomer concentration, in QSS:
    m_free + n * m_star^(1-nk) * m_free^nk = 0
    '''
    if m0_val <= m_star_val:
        return m0_val
    # QSS polynomial
    prefactor = n_val * m_star_val ** (1 - nk_val)
    def f(m_free):
        return m_free + prefactor * m_free ** nk_val - m0_val
    return brentq(f, 1e-14, m0_val * (1 - 1e-12))

# kinetic parameters
@ dataclass
class KineticParameters:
    k1: float
    k2: float
    kp: float   # fix at 1
    ko_minus: float
    nc: float
    n2: float
    nk: float
    n: float
    m_star: float

def model_analytical(t, m0, p: KineticParameters):
    m_free_qss = qss_free_monomer(m0, p.m_star, p.nk, p.n)
    Omega = m_free_qss / m0

    kappa_val = np.sqrt(2 * p.kp * m0 * p.k2 * m0 ** p.n2)
    eps_val = p.k1 * m0 ** p.nc / (2 * m0 * p.k2 * m0 ** p.n2)
    lam = np.sqrt(Omega ** (p.n2 + 1))
    Gamma = p.ko_minus / kappa_val
    c = Gamma / lam

    tau = kappa_val * t
    arg = np.clip(lam * tau, 0, 500)
    return 1 - (1 + 2 * eps_val * Omega ** (p.nc - p.n2) / c * (np.cosh(arg) - 1)) ** (-c)

def model_analytical_fixed_c(t, m0, p: KineticParameters, c_fixed):
    '''
    use for the early time data, which are unaffected by c, to extract k1 and k2 using a dummy fi    xed c
    '''
    m_free_qss = qss_free_monomer(m0, p.m_star, p.nk, p.n)
    Omega = m_free_qss / m0

    kappa_val = np.sqrt(2 * p.kp * m0 * p.k2 * m0 ** p.n2)
    eps_val = p.k1 * m0 ** p.nc / (2 * m0 * p.k2 * m0 ** p.n2)
    lam = np.sqrt(Omega ** (p.n2 + 1))
    c = c_fixed

    tau = kappa_val * t
    arg = np.clip(lam * tau, 0, 500)
    return 1 - (1 + 2 * eps_val * Omega ** (p.nc - p.n2) / c * (np.cosh(arg) - 1)) ** (-c)

#=================================
# FIT EARLY TIME DATA WITH FIXED C
#=================================
def minimize_early(guess, x_actual, y_actual, m0vals, init_guesses, free_params, fixed_params, c_fixed, M_cutoff):
    fitted = {name: np.exp(guess[i]) * init_guesses[i] for i, name in enumerate(free_params)}
    all_params = {**fixed_params, **fitted}
    # fill ko_minus value in the dataclass with some default value, it doesn't matter, it will get overwritten
    all_params.setdefault('ko_minus', 1.0)
    params = KineticParameters(**all_params)
    residual = 0.0
    for i in range(len(x_actual)):
        x, y, m0 = x_actual[i], y_actual[i], m0vals[i]
        mask = y < M_cutoff
        y_model = model_analytical_fixed_c(x[mask], m0, params, c_fixed)
        residual += np.sum((y[mask] - y_model) ** 2)
    return residual

#==========================================
# FIT ALL DATA WITH FIXED K1, K2 FROM ABOVE
#==========================================
def minimize_all(guess, x_actual, y_actual, m0vals, init_guesses, free_params, fixed_params):
    fitted = {name: np.exp(guess[i]) * init_guesses[i] for i, name in enumerate(free_params)}
    all_params = {**fixed_params, **fitted}
    params = KineticParameters(**all_params)
    residual = 0.0
    for i in range(len(x_actual)):
        y_model = model_analytical(x_actual[i], m0vals[i], params)
        residual += np.sum((y_actual[i] - y_model) ** 2)
    return residual

#======================================================================
# EXTRACT KO_MINUS FROM THE LATE TIME SLOPE TO CHECK AGAINST GLOBAL FIT
# in this regime we have
# 1 - M / m0 ~ exp(-ko_minus * t) as per the asymptotic matching argument
#======================================================================
def extract_ko_minus(t, y, y_low=0.7, y_high=0.99):
    t = np.asarray(t)
    y = np.asarray(y)
    mask = (y > y_low) & (y < y_high)
    log_difference = np.log(1 - y[mask])   # y data is the normalized M data
    slope, intercept = np.polyfit(t[mask], log_difference, 1)
    return -slope

def mean_residual_error(y_actual_list, y_model_list):
    # make this a function since now it will be called a few times
    total = sum(np.sum((y_a - y_m) ** 2) for y_a, y_m in zip(y_actual_list, y_model_list))
    total_points = sum(len(y_a) for y_a in y_actual_list)
    return total / total_points

#=================================
# UGLY BORING DATA LOAD AND CLEANUP
#=================================
from richards import load_data     # my trusted function
data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_proper_norm.tsv'
case = 'pH_6_old'
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

#===================
# MAIN TWO-STAGE FIT
#==================
if __name__ == '__main__':






