'''
Problem with full global fitting: the tail which is (~)completely controlled by ko_minus can trade off with the other fitted parameters, i.e. k1 and k2.
On the other hand, the early time data is unaffected by ko_minus, so k1 and k2 can be better constrained from that alone. Once these are fixed, we can fit ko_minus to the whole curve.
Also, similarly to Ana's project on boundary layer kinks, we can in fact read off ko_minus directly off the late-time slope of the (log) data.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import basinhopping, brentq
from dataclasses import dataclass
import sys

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

sys.path.append('/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits')
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
    # parameters that are fixed throughout both stages of the fit
    fixed_params_common = {'n': 100, 'nk': 100, 'kp': 1, 'm_star': 2.5, 'n2': 0, 'nc': 2}

    #================================================
    # FIT K1, K2 ON EARLY-TIME DATA ONLY, C = 1 FIXED
    #================================================

    cutoff = 0.5
    c_fixed = 1.0

    free_params_early = ['k1', 'k2']
    init_guesses_early = [1, 1]
    fixed_params_early = dict(fixed_params_common)

    res_early = basinhopping(minimize_early, np.zeros(len(init_guesses_early)), niter=100, stepsize=0.3, seed=42, minimizer_kwargs={'method': 'Nelder-Mead', 'tol': 1e-10, 'args': (x_data, y_data, m0vals, init_guesses_early, free_params_early, fixed_params_early, c_fixed, cutoff)})
    fitted_early = {name: init_guesses_early[i] * np.exp(res_early.x[i]) for i, name in enumerate(free_params_early)}
    k1_fit = fitted_early['k1']
    k2_fit = fitted_early['k2']
    print('early time fitted k1: ', k1_fit)
    print('early time fitted k2: ', k2_fit)

    #===============================================================
    # FIT KO_MINUS ON THE FULL DATA, WITH K1 AND K2 FIXED FROM ABOVE
    #===============================================================
    free_params_all = ['ko_minus']
    init_guesses_all = [5.0]
    fixed_params_all = {**fixed_params_common, 'k1': k1_fit, 'k2': k2_fit}

    res_all = basinhopping(minimize_all, np.zeros(len(init_guesses_all)), niter=100, stepsize=0.3, seed=42, minimizer_kwargs = {'method': 'Nelder-Mead', 'tol': 1e-10, 'args': (x_data, y_data, m0vals, init_guesses_all, free_params_all, fixed_params_all)})
    ko_minus_fit = init_guesses_all[0] * np.exp(res_all.x[0])
    print('fitted ko_minus on complete curves: ', ko_minus_fit)

    # calculate total MRE obtained with this method
    params_fit = KineticParameters(**{**fixed_params_common, 'k1': k1_fit, 'k2': k2_fit, 'ko_minus': ko_minus_fit})
    y_model = [model_analytical(x, m0, params_fit) for x, m0 in zip(x_data, m0vals)]
    mre = mean_residual_error(y_data, y_model)
    print('MRE is: ', mre)

    #=========================================
    # CROSS-CHECK KO_MINUS FROM LATE TIME DATA
    #=========================================
    for x, y, m0 in zip(x_data, y_data, m0vals):
        ko_minus_estimate = extract_ko_minus(x, y)
        print('estimated ko_minus from late time data: ', ko_minus_estimate)

    #===============
    # PLOT FINAL FIT
    #===============
    unique_m0 = sorted(set(m0vals))
    palette = sns.color_palette('tab10', n_colors = len(unique_m0))
    color_map = {m0: palette[i] for i, m0 in enumerate(unique_m0)}
    seen_m0 = set()

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
        m0 = m0vals[i]
        color = color_map[m0]

        data_label = f'{m0}' if m0 not in seen_m0 else None
        seen_m0.add(m0)

        y_fit = model_analytical(x, m0vals[i], params_fit)
        ax1.plot(x, y_fit, color=color, linewidth=2)
        ax1.scatter(x, y, s=35, color=color, alpha=0.7, linewidth=0, label=data_label)
    plt.xlabel('Time (h)', fontsize=15)
    plt.ylabel('Normalised fibril mass', fontsize=15)
    plt.legend()
    plt.show()




