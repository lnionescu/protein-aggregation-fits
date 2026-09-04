import sys
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
import pymc as pm   # the python package for bayesian inference

# pymc models translate to pytensor graphs, whatever that means
import pytensor
import pytensor.tensor as pt
import scipy.stats
from scipy.stats import gaussian_kde

import arviz as az   # visualization for bayesian inference
print(hasattr(az, 'plot_posterior'))

from scipy.optimize import brentq   # for initial condition explicit solution

from dataclasses import dataclass

sys.path.append('/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/')
from richards import load_data

# load and clean data

# use for old pH 6 data:
data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_proper_norm.tsv'
case = 'pH_6_old'

# use for old pH 6.5 data:
#data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv'
#case = 'pH_6.5_old'

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
    if m0_val.all() <= m_star_val:
        return m0_val
    prefactor = n_val * m_star_val ** (1 - nk_val)
    def f(m_free):
        return m_free + prefactor * m_free ** nk_val - m0_val
    return brentq(f, 1e-14, m0_val * (1 - 1e-12))

# constants (for pH 6 !!!!)
m_star = 2.5
#m_star = 4   # for pH 6.5
n = 100
nk = 100
kp = 1
nc = 2
n2 = 0

# ingredients 

# analytical solution, this time using the pytensor package for pymc to be able to use it
def model_analytical(t, m0,Omega, k1, k2, ko_minus):
    #m_free_qss = qss_free_monomer(m0, m_star, nk, n)
    #Omega = m_free_qss / m0
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
    log_k1 = pm.Normal('log_k1', mu = np.log(1), sigma=5)
    log_k2 = pm.Normal('log_k2', mu=np.log(1), sigma=5)
    log_ko_minus = pm.Normal('log_ko_minus', mu =np.log(3), sigma=5)

    # record intermediate optimization results as deterministic variables
    k1 = pm.Deterministic('k1', pt.exp(log_k1))
    k2 = pm.Deterministic('k2', pt.exp(log_k2))
    ko_minus = pm.Deterministic('ko_minus', pt.exp(log_ko_minus))

    # standard deviation of observed likelihood, half normal since it has to be positive
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.1)

    # model prediction
    y_pred = model_analytical(t_all, m0_all, Omega_all, k1, k2, ko_minus)

    # observed likelihood
    pm.Normal('obs', mu=y_pred, sigma=sigma_obs, observed=y_all)

    # sampling parameter space
    # draws is defaulted to 1000; tune: adjust step sizes/scalings during tuning
    samples = pm.sample(draws=1000, tune=1000, chains=4, cores=1, random_seed=42, progressbar=True)
    # TODO maybe specify the method and add a target_accept later?

summary = az.summary(samples, var_names=['k1', 'k2', 'ko_minus', 'sigma_obs'])
print(summary)


# posterior plots
#plots = az.plot_dist(samples, var_names=['k1', 'k2', 'ko_minus'], backend='matplotlib')
#plots.show()

# check correlations between parameters
#fig, ax = plt.subplots()
#log_k1_samples = samples.posterior['log_k1'].values.flatten()
#log_k2_samples = samples.posterior['log_k2'].values.flatten()
#ax.scatter(log_k1_samples, log_k2_samples)
#plt.tight_layout()
#plt.show()

# posterior plots made with matplotlib
from scipy.stats import gaussian_kde

k1_samples = samples.posterior['k1'].values.flatten()
k2_samples = samples.posterior['k2'].values.flatten()
ko_samples = samples.posterior['ko_minus'].values.flatten()

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))

# k1 spans many orders of magnitude
log_k1 = np.log10(k1_samples[k1_samples > 0])
kde = gaussian_kde(log_k1)
xs = np.linspace(log_k1.min(), log_k1.max(), 300)
axes[0].plot(xs, kde(xs))
axes[0].fill_between(xs, kde(xs), alpha=0.2)
axes[0].set_xlabel(r'$\log_{10}(k_1)$')
axes[0].set_title(r'$k_1$')

for ax, vals, title in zip(axes[1:], [k2_samples, ko_samples],
                           [r'$k_2$', r'$k_{o,-}$']):
    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(xs, kde(xs))
    ax.fill_between(xs, kde(xs), alpha=0.2)
    ax.set_xlabel(title)
    ax.set_title(title)

for ax in axes:
    ax.set_yticks([])
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()

# joint posterior of k1 vs k2
fig, ax = plt.subplots(figsize=(3.4, 3))
counts, xedges, yedges = np.histogram2d(np.log10(k1_samples), np.log10(k2_samples), bins=40)
im = ax.imshow(counts.T, origin='lower', aspect='auto', cmap='Blues',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.set_xlabel(r'$\log_{10}(k_1)$')
ax.set_ylabel(r'$\log_{10}(k_2)$')
fig.colorbar(im, ax=ax, label='density')
plt.tight_layout()
plt.show()

# joint posterior of k1 and ko_minus
fig, ax = plt.subplots(figsize=(3.4, 3))
counts, xedges, yedges = np.histogram2d(np.log10(k1_samples), ko_samples, bins=40)
im = ax.imshow(counts.T, origin='lower', aspect='auto', cmap='Blues',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.set_xlabel(r'$\log_{10}(k_1)$')
ax.set_ylabel(r'$k_{o,-}$')
fig.colorbar(im, ax=ax, label='density')
plt.tight_layout()
plt.show()






# checking fit quality for a parameter combination within the error bounds of the Bayesian sampler







