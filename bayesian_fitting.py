import sys
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm   # the python package for bayesian inference

# pymc models translate to pytensor graphs, whatever that means
import pytensor
import pytensor.tensor as pt
import scipy.stats

import arviz as az   # visualization for bayesian inference

from scipy.optimize import brentq   # for initial condition explicit solution

from dataclasses import dataclass

sys.path.append('/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/')
from richards import load_data

# load and clean data
# use for old pH 6 data:
data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_proper_norm.tsv'
case = 'pH_6_old'

x_data, y_data, m0vals = load_data(data_path)

# remove m0=0.8 curves; remove the third m0=1.4 replicate (old pH 6 data)
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



def qss_free_monomer(m0_val, m_star_val, nk_val, n_val):
    if m0_val <= m_star_val:
        return m0_val
    prefactor = n_val * m_star_val ** (1 - nk_val)
    def f(m_free):
        return m_free + prefactor * m_free ** nk_val - m0_val
    return brentq(f, 1e-14, m0_val * (1 - 1e-12))

# constants (for pH 6 !!!!)
m_star = 2.5
n = 100
nk = 100
kp = 1
nc = 2
n2 = 0

# ingredients 

# analytical solution, this time using the pytensor package for pymc to be able to use it
def model_analytical(t, m0, kn, k2, ko_minus):
    m_free_qss = qss_free_monomer(m0, m_star, nk, n)
    Omega = m_free_qss / m0
    kappa_val = pt.sqrt(2 * kp * m0 * k2 * m0 ** n2)
    eps_val = k1 * m0 ** nc / (2 * m0 * k2 * m0 **n2)
    lam = pt.sqrt(Omega ** (n2+1))
    Gamma = ko_minus / kappa_val

    c = Gamma / lam
    tau = kappa_val * t
    arg = pt.clip(lam * tau, 0, 500)
    return 1 - (1 + 2 * eps_val * Omega ** (nc-n2) / c * (pt.cosh(arg) -1))**(-c)

# we are interested in the posteriors of all of kn, k2, ko_minus when fitting to this model

# put all data together, each with their own Omega but the rest is shared
Omega_vals = [qss_free_monomer(m0, m_star, nk, n) / m0 for m0 in m0vals]
Omega_vals = np.array(Omega_vals)

t_all = np.concatenate(x_data)
y_all = np.concatenate(y_data)
m0_all = np.concatenate([np.full(len(x), m0) for (x, m0) in zip(x_data, m0vals)])
Omega_all = np.concatenate([np.full(len(x), Omega) for (x, Omega) in zip(x_data, Omega_vals)])


# build pymc model
with pm.Model() as model:
    # priors for unknown model parameters: k1, k2, ko_minus; basically assume they are normal distributions around the initial guess (where the initial guess is the same as in basinhopping)
    # for consistency with basinhopping, use logs of the actual parameters, but i'm not sure if this is justified or necessary
    log_k1 = pm.Normal('log_k1', mu = np.log(1e-3), sigma=5)
    log_k2 = pm.Normal('log_k2', mu=np.log(10), sigma=5)
    log_ko = pm.Normal('log_ko', mu =np.log(3), sigma=5)

    # record intermediate optimization results as deterministic variables
    k1 = pm.Deterministic('k1', pt.exp(log_k1))
    k2 = pm.Deterministic('k2', pt.exp(log_k2))
    ko = pm.Deterministic('ko', pt.exp(log_ko))







