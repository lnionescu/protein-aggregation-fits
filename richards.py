import numpy as np
from scipy.optimize import basinhopping
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

'''
fitting constraints:
    1. eps, kappa and c are fit per curve for m0 < m*, but shared for all m0 > m*
    2. the regime switch aboe/below m* is currently a sharp transition
'''

@dataclass
class KineticParameters:
    eps0: float    # constant in front of eps sclaing with m0
    alpha: float   # how eps scales with m0
    kappa0: float    # constant in front of kappa sclaing with m0
    beta: float    # how kappa scales with m0
    c0: float     # constant in front of c scaling with m0
    gamma: float    # how c scales with m0
    m_star: float     # either fit or fix to e.g. the median of the m0 values


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

'''
first define the model such that eps, kappa and c are shared across all m0 values, then layer the m0 > m* / m0 < m* logic on top of that
'''
def richards_curve(t, eps, c, kappa):
    bracket = 1 + eps / c * (np.exp(kappa*t) + np.exp(-kappa*t) - 2)
    return 1 - bracket**(-c)    # M(t) / m0, normalised kinetic curve

def model(t, m0, params: KineticParameters):
    m0_eff = np.minimum(m0, params.m_star)
    # add dependence on m_eff to kappa and eps
    eps = params.eps0 * m0_eff**params.alpha
    kappa = params.kappa0 * m0_eff**params.beta
    c = params.c0 * m0_eff**params.gamma
    return richards_curve(t, eps, c, kappa)

def load_data(data_path):
    df = pd.read_csv(data_path, sep='\t', header=1)
    df_with_header = pd.read_csv(data_path, sep='\t', header=None)
 
    x_data, y_data = [], []
    for i in range(len(df.columns) // 2):
        x = df.iloc[:, i * 2].values
        y = df.iloc[:, i * 2 + 1].values
        x_data.append(x[~np.isnan(x)])
        y_data.append(y[~np.isnan(y)])
 
    m0vals = np.array([
        float(df_with_header.iloc[0, i * 2].split(': ')[-1])
        for i in range(len(df.columns) // 2)
    ])
    print('m0 values:', m0vals)
    return x_data, y_data, m0vals




if __name__ == "__main__":
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv' 
    x_data, y_data, m0vals = load_data(data_path)
    
    #========
    # FITTING
    #========
    '''move this to a function later if it gets too long'''
    free_params = ['eps0', 'alpha', 'kappa0', 'beta', 'c0', 'gamma']
    fixed_params = {'m_star': 3}
    # initial guesses eps0, alpha, kappa0, beta, c0, gamma: log space (TODO check that)
    initial_guesses = [1, 0.5, 1, 1, 1, 1]

    seed = 42
    res = basinhopping(minimize, np.zeros(len(initial_guesses), dtype=float), niter=50, stepsize=0.5, seed=seed, minimizer_kwargs={'method': 'Nelder-Mead', 'tol': 1e-10, 'args': (x_data, y_data, m0vals, initial_guesses, free_params, fixed_params)})
    fitted_values = {name: initial_guesses[i] * np.exp(res.x[i]) for i, name in enumerate(free_params)}
    print('fitted values', fitted_values)
    total_data_points = sum(len(y) for y in y_data)
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
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
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

