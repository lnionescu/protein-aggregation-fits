import sys
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm   # the python package for bayesian inference

# pymc models translate to pytensor graphs
import pytensor
import pytensor.tensor as pt
import scipy.stats

import arviz as az   # visualization for bayesian inference

from scipy.opimize import brentq   # for initial condition explicit solution

def qss_free_monomer(m0_val, m_star_val, nk_val, n_val):
    if m0_val <= m_star_val:
        return m0_val
    prefactor = n_val * m_star_val ** (1 - nk_val)
    def f(m_free):
        return m_free + prefactor * m_free ** nk_val - m0_val
    return brentq(f, 1e-14, m0_val * (1 - 1e-12))


