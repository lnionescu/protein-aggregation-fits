import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import basinhopping
from scipy import special
from dataclasses import dataclass

@dataclass
class KineticParameters:
    kn: float
    nc: float
    k2: float
    n2: float
    kp: float
    ko_plus: float
    ko_minus: float
    nk: float

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
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv'
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


    # free and fixed params
    free_params = ['kn', 'k2', 'ko_plus', 'ko_minus']
    fixed_params = {'kp': 1, 'n2': 2, 'nc': 2, 'nk': 3}
    # initial guesses kn, k2
    initial_guesses = [1, 100, 0.01, 10]

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

def model(t, m0, p: KineticParameters):
    kappa = np.sqrt(2 * p.kp * m0 * p.k2 * m0**p.n2)
    tau = kappa * t

    Lambda = p.ko_plus * m0**p.nk / kappa
    Gamma = p.ko_minus / kappa
    Omega = Gamma / (Lambda + Gamma)

    lam = Omega**((p.n2 + 1)/2)
    eps = p.kn * m0**(p.nc-1) / (2 * p.k2 * m0**p.n2)
    c = Gamma 

    prefactor = 2 * eps * Omega**(p.nc - p.n2)
    return 1 - (1 + prefactor/c * (np.cosh(lam*tau) - 1))**(-c)



main()


