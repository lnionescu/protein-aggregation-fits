'''
Extract kn, k2 from early-time portion of the curves; extract ko_minus from plateau directly; simulate exact kinetics and compare with actual curves.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
sys.path.append('/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits')
from richards import load_data

#=====================
# CONSTANTS THROUGHOUT
#=====================
n = 100
nk = 100
nc = 2
n2 = 0
kp = 1



#==================== 
# SETUP: KNOWN KN, K2
#====================

# use for old pH 6 data
k1_fit = 0.0459
k2_fit = 23.6
m_star = 2.5
data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_proper_norm.tsv'
case = 'pH_6_old'

