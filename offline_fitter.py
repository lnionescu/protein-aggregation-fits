#=========================================================================
# recreate amylofit fitting offline, with the most general model available
# all fits are global
# input normalized data
#========================================================================

import numpy as np
from scipy import special
from scipy.optimize import basinhopping
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

#=============================================================================================
#without inhibition:
#    set cd = 0 (concentration of the inhibitor)
#    set KIE = 1e8, KIP = 1e8, KIS = 1e8, KIM = 1e8 or to any random number since they will drop out due to cd = 0 anyway (inhibition constants for possible steps)

#without saturation:
#    set KE, KP, KS = 1e12 or some other large number
# default kinetic parameters remove inhibition and saturation, can modify this in minimize()
#==============================================================================================

@dataclass
class KineticParameters:
    kn: float
    nc: float
    k2: float
    n2: float
    kp: float 
    km: float 
    KE: float = 1e12   # by default unsaturated elongation at all concentartions
    KP: float = 1e12   # by default unsaturated primary nucleation at all concentrations
    KS: float = 1e12   # by default unsaturated secondary nucleation at all concentrations
    cd: float = 0.0    # by default no inhibitor
    KIE: float = 1.0
    KIP: float = 1.0
    KIS: float = 1.0
    KIM: float = 1.0
    P0: float = 0.0
    M0: float = 0.0



def all_inhib_all_sat_base_function(
    t,
    m0,    # initial monomer concentration
    P0,    # initial fibril number concentration
    M0,    # initial fibril mass concentration
    kn,    # rate constant for primary nucleation
    nc,    # exponent for primary nucleation
    k2,    # rate constant for secondary nucleation
    n2,    # exponent for secondary nucleation
    kp,    # rate constant for elongation
    km,    # rate constant for fragmentation
    KE,    # saturation constant for elongation
    KP,    # saturation constant for primary nucleation
    KS,    # saturation constant for secondary nucleation
    cd,    # inhibitor concentration
    KIE,   # inhibition constant of elongation
    KIP,   # inhibition constant of primary nucleation
    KIS,   # inhibition constant for secondary nucleation
    KIM,   # inhibition constant for monomers
):  # the general function that has all processes, with saturation, and all inhibition mechanisms
    
    mtot = m0 + M0
    delta0 = M0 / m0 if m0 !=0 else 0.0
    mu0 = m0 / mtot if mtot !=0 else 0.0
    kp = kp / (1 + cd / KIM)
    kn = kn / (1 + cd / KIM) ** nc
    k2 = k2 / (1 + cd / KIM) ** n2
    KE = KE * (1 + cd / KIM)
    KP = KP * (1 + cd / KIM)
    KS = KS * (1 + cd / KIM)

    # above quantities stay the same if there is no inhibitor, cd=0

    #=========================================================================================
    # ensure arbitrarily large values of saturation constants remove saturation from the model
    #=========================================================================================

    # evaluated at mtot
    ksd = KS / mtot if mtot != 0 else 0.0
    if n2 < 1:
        ksn = 10**8
    else:
        ksn = ksd**n2

    kud = KP / mtot if mtot !=0 else 0.0
    if nc < 1:
        kun = 10**8
    else:
        kun = kud * nc

    # evaluated at m0
    ksdo = KS / m0 if m0 !=0 else 0.0
    if n2 < 1:
        ksno = 10**8
    else:
        ksno = ksdo**n2

    kudo = KP / m0 if m0 != 0 else 0.0
    if nc < 1:
        kuno = 10**8
    else:
        kuno = kudo ** nc

    #=================================
    # core model effective paraameters
    #================================
    kappa = np.sqrt(
        2
        * kp
        * mtot
        * (km + (k2 * mtot**n2) / (1 + 1 / ksn + cd / KIS))
        / (1 + mtot / KE + cd / KIE)
    )
    epsilon = (kn * mtot**nc / (1 + 1 / kun + cd / KIP)) / (
        2 * (km * mtot + k2 * mtot ** (n2 + 1) / (1 + 1 / ksn + cd / KIS))
    )
    kappa0 = np.sqrt(
        2
        * kp
        * m0
        * (km + (k2 * m0**n2) / (1 + 1 / ksno + cd / KIS))
        / (1 + m0 / KE + cd / KIE)
    )
    epsilon0 = (kn * m0**nc / (1 + 1 / kuno + cd / KIP)) / (
        2 * (km * m0 + (k2 * m0 ** (n2 + 1)) / (1 + 1 / ksno + cd / KIS))
    )
    p = 2 * kp * P0 / (kappa * (1 + mtot / KE + cd / KIE))
    p0 = 2 * kp * P0 / (kappa0 * (1 + m0 / KE + cd / KIE))

    #================================================
    # mu -> 1 asymptotic symmetry solution parameters
    #================================================ 
    n2p = (
        (k2 * mtot**n2 / (1 + 1 / ksn + cd / KIS))
        / (km + (k2 * mtot**n2 / (1 + 1 / ksn + cd / KIS)))
        * (n2 * (1 + cd / KIS) / (1 + 1 / ksn + cd / KIS))
        - ((2 * mtot / KE) / (1 + mtot / KE + cd / KIE))
    )
    c1 = 3 / (2 * n2p + 1)

    #======================================= 
    # compositie symmetry solution parameters
    #========================================
    q2a = (
        special.hyp2f1(1, 1 + 1 / n2, 2 + 1 / n2, -mu0**n2 / ksn)
        * mu0 ** (1 + n2)
        / (1 + n2)
    )
    q2b = (
        special.hyp2f1(1, 1 + 2 / n2, 2 + 2 / n2, -mu0**n2 / ksn)
        * mu0 ** (2 + n2)
        / (2 + n2)
    )
    q1 = (
        special.hyp2f1(1, 1 + 1 / nc, 2 + 1 / nc, -mu0**nc / kun)
        * mu0 ** (1 + nc)
        / (1 + nc)
    )
    c0 = np.sqrt(
        p**2
        + 4
        * epsilon
        * ((1 + 1 / kun) / (1 + mtot / KE))
        * (np.log(1 + mu0**nc / kun) * kun / nc + q1 * mtot / KE)
        + 2
        * ((1 + 1 / ksn) / (1 + mtot / KE))
        * (
            np.log(1 + mu0**n2 / ksn) * ksn / n2
            - (1 - mtot / KE) * q2a
            - q2b * mtot / KE
        )
    )

    #===================================================================
    # switching condition based on values of epsilon evaluated at mtot/2
    # i.e. the ratio of primary to secondary process rates at half time
    #==================================================================
    ksdh = (2 * KS) / mtot if mtot != 0 else 0.0
    if n2 < 1:
        ksnh = 10**8
    else:
        ksnh = ksdh**n2
    kudh = (2 * KP) / mtot if mtot !=0 else 0.0
    if nc < 1:
        kunh = 10 **8
    else:
        kunh = kudh ** nc

    epsh = (kn * mtot**nc / (1 + 1 / kunh + cd / KIP)) / (
        km * mtot + k2 * mtot ** (n2 + 1) / (1 + 1 / ksnh + cd / KIS)
    )

    if p > 0.2:
        c = c0
    elif epsh > 0.1:
        c = c0
    else:
        c = c1

    a1 = kappa * t * (kappa * t < 30) + 30 * (kappa * t >= 30)  # to avoid overflow in exp
    mt = m0 * (
        1
        + (p0 / (2 * c))
        * (kappa0 / kappa)
        * (np.exp(a1) - np.exp(-a1))
        + (kappa0**2 / kappa**2)
        * (
            epsilon0 / c
            + delta0 / (2 * c)
            + (p0**2 / (2 * c**2))
            * (1 + (c * m0 / KE) / (1 + m0 / KE + cd / KIE))
        )
        * (np.exp(a1) + np.exp(-a1) - 2)
    ) ** (-c)
    return (mtot - mt) / mtot


#=============================================================================
# define a particular model (can remove inhibition, saturation via parameters)
#=============================================================================
def model(t: np.ndarray, m0: float, params: KineticParameters) -> np.ndarray:
    return all_inhib_all_sat_base_function(
        t,
        m0,
        params.P0,
        params.M0,
        params.kn,
        params.nc,
        params.k2,
        params.n2,
        params.kp,
        params.km,
        params.KE,
        params.KP,
        params.KS,
        params.cd,
        params.KIE,
        params.KIP,
        params.KIS,
        params.KIM,
    )

#==================================
# objective function for minimizing
# guess are parameters in log space
#=================================
def minimize(guess, x_actual, y_actual, m0vals, init_guesses, free_params, fixed_params):
    '''
    guess = values in log space for the free parameters
    free_params = list of parameters to fit
    fixed_params = dictionary of parameters that stay constant and their respective values
    '''
    fitted = {name: np.exp(guess[i]) * init_guesses[i] for i, name in enumerate(free_params)}

    # collect parameters of both types
    all_params = {**fixed_params, **fitted}
    params = KineticParameters(**all_params)
    residual = 0.0
    for i in range(len(x_actual)):
       y_model = model(x_actual[i], m0vals[i], params)
       residual += np.sum((y_actual[i] - y_model) ** 2)

    return residual

def main():
    #===================== 
    # load normalized data
    #=====================
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/ph-7.5-normalized-data/data.tsv' 
    df_with_header = pd.read_csv(data_path, sep='\t', header=None)   # keep header to extract m0
    df = pd.read_csv(data_path, sep = '\t', header=1)
    current_df = df
    x_actual_data = []
    y_actual_data = []

    for i in range(len(current_df.columns) // 2):
        x_current = current_df.iloc[:, i*2].values
        y_current = current_df.iloc[:, i*2+1].values
        x_actual_data.append(x_current[~np.isnan(x_current)])
        y_actual_data.append(y_current[~np.isnan(y_current)])

    m0vals = [float(df_with_header.iloc[0,i*2].split(': ')[-1]) for i in range(len(current_df.columns) // 2)]
    print(m0vals)



    #===========================
    # here we decide what to fit
    #===========================

    # model_type = 'primary-and-secondary-nucleation'
    # model_type = 'saturating-secondary-nucleation'
    # model_type = 'saturating-primary-and-secondary'
    model_type = 'custom'

    if model_type == 'primary-and-secondary-nucleation':
        free_params = ['kn', 'nc', 'k2', 'n2']
        fixed_params = {'kp': 1, 'km': 1e-4}
        initial_guesses = np.array([1.0, 1.0, 1.0, 1.0])
    
    elif model_type == 'saturating-secondary-nucleation':
        free_params = ['k2', 'n2', 'KS']
        fixed_params = {'kp': 1, 'km': 1e-4, 'kn': 1, 'nc': 1}
        initial_guesses = np.array([1e-5, 2.0, 1])

    elif model_type == 'saturating-primary-and-secondary':
        free_params = ['kn', 'nc', 'KP', 'k2', 'n2', 'KS']
        fixed_params = {'kp': 1, 'km': 1e-4}
        initial_guesses = np.array([1, 2, 10, 1, 2, 10])


    elif model_type == 'custom':
        free_params = ['kn', 'k2', 'nc',  'n2', 'KS']
        fixed_params = {'km': 1e-10, 'kp': 1}
        initial_guesses = np.array([1e-2, 10, 1e-2, 2, np.mean(m0vals)])





    seed = 42
    res = basinhopping(minimize, np.zeros(len(initial_guesses), dtype=float), niter=200, stepsize=0.5, seed =seed, minimizer_kwargs={'method': 'Nelder-Mead', 'tol': 1e-10, 'args': (x_actual_data, y_actual_data, m0vals, initial_guesses, free_params, fixed_params)})

    fitted_values = {name: initial_guesses[i] * np.exp(res.x[i]) for i, name in enumerate(free_params)}
    print('fitted values', fitted_values)
    total_data_points = sum(len(y) for y in y_actual_data)
    print('residual', res.fun)
    print('mean residual error', res.fun / total_data_points)


    # Plot data and fitted curves
    # all curves with the same m0 have the same colour
    sns.set_theme(style='whitegrid', context='paper')
    unique_m0 = sorted(set(m0vals))
    palette = sns.color_palette('tab10', n_colors = len(unique_m0))
    color_map = {m0: palette[i] for i, m0 in enumerate(unique_m0)}
    
    # merge fixed parameters and fitted values of free parameters in the data class
    all_params = {**fixed_params, **fitted_values}
    params_fit = KineticParameters(**all_params) 


    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(111)
    seen_m0 = set()    # keep track of labels to only label each m0 once
    for i in range(len(x_actual_data)):
        x = x_actual_data[i]
        y = y_actual_data[i]
        m0 = m0vals[i]
        color = color_map[m0]

        data_label = f'{m0}' if m0 not in seen_m0 else None
        fit_label = f'{m0}' if m0 not in seen_m0 else None
        seen_m0.add(m0)

        y_fit = model(x, m0vals[i], params_fit)
        ax1.plot(x, y_fit, color=color, linewidth=3)
        ax1.scatter(x, y, s=35, color=color, alpha=0.9, linewidth=5, label=data_label)
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()










