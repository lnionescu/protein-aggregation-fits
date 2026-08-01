import numpy as np
from scipy.optimize import basinhopping
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from scipy.optimize import brentq

@dataclass
class KineticParameters:
    eps0: float
    alpha: float   # scaling of eps with m0
    kappa0: float
    beta: float    # scaling of kappa with m0
    c_low: float
    m_star: float
    n: float   # not actually used but still
    c_high: float = None   # to handle the single c case, where it gets overwirtten after fitting
'''
if we want c_low = c_high, constrain that locally when the model is called, no need for a separate data class or model definition
'''



def minimize(guess, x_actual, y_actual, m0vals, init_guesses, free_params, fixed_params):
    '''
    guess = values in log space for the free parameters
    free_params = list of parameters to fit
    fixed_params = dictionary of parameters that stay constant and their respective values
    '''
    fitted = {name: np.exp(guess[i]) * init_guesses[i] for i, name in enumerate(free_params)}
    #fitted = {name: np.sinh(guess[i]) * init_guesses[i] for i, name in enumerate(free_params)}
    #fitted = {name: np.sign(guess[i]) * np.exp(np.abs(guess[i])) * init_guesses[i] 
          #for i, name in enumerate(free_params)}
    # allow fitter to go negative on all parameters

    # collect parameters of both types
    all_params = {**fixed_params, **fitted}
    params = KineticParameters(**all_params)
    residual = 0.0
    for i in range(len(x_actual)):
       y_model = model(x_actual[i], m0vals[i], params)
       residual += np.sum((y_actual[i] - y_model) ** 2)

    # for the sigmoid to not be flat, add some bounds to the c values
    c_min, c_max = 0.05, 5.0
    for name in ['c_low', 'c_high']:
        if name in fitted:
            c = fitted[name]
            if c <= c_min or c >= c_max:
                residual += 1e6   # don't let c overcome boundaries ever
            else:
                residual -= 0.01 * (np.log(c-c_min) + np.log(c_max-c)) 
                # also penalize c approaching the boundaries too much but allow it to get close if need be



    return residual

def richards_curve(t, eps, c, kappa):
    bracket = 1 + eps / c * (np.exp(kappa*t) + np.exp(-kappa*t) - 2)
    return 1 - bracket**(-c)    # M(t) / m0, normalised kinetic curve

def model(t, m0, params: KineticParameters):

    m0_eff = np.minimum(m0, params.m_star)
    eps = params.eps0 * m0_eff**params.alpha
    kappa = params.kappa0 * m0_eff**params.beta
    c = params.c_low if m0 < params.m_star else (params.c_high or params.c_low)
    # to handle single c case: if c_high is None, use c_low (c_high gets written as c_low after fitting is complete)

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
    # load data

    # use for old pH 6 data:
    #data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_proper_norm.tsv'
    #case = 'pH_6_old'

    # use for old pH 6.5 data:
    #data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv'
    #case = 'pH_6.5_old'

    # use for new pH 6.5 data:
    #data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6.5_june.tsv'
    #case = 'pH_6.5_new'

    # use for new pH 6 data:
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_june_shorter_plateau.tsv'
    case = 'pH_6_new'

    import seaborn as sns
    x_data, y_data, m0vals = load_data(data_path)

    # remove m0=0.8 curves, remove the third m0=1.4 curve due to high overlap with other m0 curves for old pH 6 data
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

    # for the new pH 6.5 data, remove all m0=1.25 curves, as well as the first m0=1.75 curve, as well as the third m0=2.5 curve
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

        # try removing all m0=2.5 curves
        #mask = m0vals != 2.5
        #x_data = [x for x, m0 in zip(x_data, m0vals) if m0 != 2.5]
        #y_data = [y for y, m0 in zip(y_data, m0vals) if m0 != 2.5]
        #m0vals = m0vals[mask]

        drop_idx = np.where(m0vals == 2.5)[0][2]
        keep = np.arange(len(m0vals)) != drop_idx
        x_data = [x for x, k in zip(x_data, keep) if k]
        y_data = [y for y, k in zip(y_data, keep) if k]
        m0vals = m0vals[keep]

    # for the new pH 6 data, drop: second m0=5 curve; second m0=3.8 curve; second m0=1.75 curve; remove third m0=1.25 curve
    # remove also all m0=1.25 curves
    elif case == 'pH_6_new':
        drop_idx = np.where(m0vals == 5)[0][1]
        keep = np.arange(len(m0vals)) != drop_idx
        x_data = [x for x, k in zip(x_data, keep) if k]
        y_data = [y for y, k in zip(y_data, keep) if k]
        m0vals = m0vals[keep]
        drop_idx = np.where(m0vals == 3.8)[0][1]
        keep = np.arange(len(m0vals)) != drop_idx
        x_data = [x for x, k in zip(x_data, keep) if k]
        y_data = [y for y, k in zip(y_data, keep) if k]
        m0vals = m0vals[keep]
        drop_idx = np.where(m0vals == 1.75)[0][1]
        keep = np.arange(len(m0vals)) != drop_idx
        x_data = [x for x, k in zip(x_data, keep) if k]
        y_data = [y for y, k in zip(y_data, keep) if k]
        m0vals = m0vals[keep]

        mask = m0vals != 1.25
        x_data = [x for x, m0 in zip(x_data, m0vals) if m0 != 1.25]
        y_data = [y for y, m0 in zip(y_data, m0vals) if m0 != 1.25]
        m0vals = m0vals[mask]

       

        print('kept m0vals:', m0vals)

    



    #========
    # FITTING
    #========
   
    # decide if we use a c all across the m0 range or two c's, one above m_star and one below
    SPLIT_C = False

    free_params = ['eps0', 'kappa0', 'alpha', 'beta', 'c_low']
    #initial_guesses = [1e-5, 100, 2, 1, 0.2]
    initial_guesses = [1, 1, 1, 1, 1]
    fixed_params = {'n': 10, 'm_star': 2.5}

    if SPLIT_C:
        free_params.append('c_high')
        initial_guesses.append(0.5)


    seed = 42
    res = basinhopping(minimize, np.zeros(len(initial_guesses), dtype=float), niter=100, stepsize=0.3, seed=seed, minimizer_kwargs={'method': 'Nelder-Mead', 'tol': 1e-10, 'args': (x_data, y_data, m0vals, initial_guesses, free_params, fixed_params)})
    fitted_values = {name: initial_guesses[i] * np.exp(res.x[i]) for i, name in enumerate(free_params)}
    print('fitted values', fitted_values)
    total_data_points = sum(len(y) for y in y_data)
    print('residual', res.fun)
    print('mean residual error', res.fun / total_data_points)

    # after fitting: if SPLIT_C was false, set c_high to the fitted value of c_low
    if not SPLIT_C:
        fitted_values['c_high'] = fitted_values['c_low']


    # Plot data and fitted curves
    # all curves with the same m0 have the same colour
    sns.set_theme(style='whitegrid', context='paper')
    unique_m0 = sorted(set(m0vals))
    palette = sns.color_palette('tab10', n_colors = len(unique_m0))
    color_map = {m0: palette[i] for i, m0 in enumerate(unique_m0)}
    
    # merge fixed parameters and fitted values of free parameters in the data class
    all_params = {**fixed_params, **fitted_values}
    params_fit = KineticParameters(**all_params) 


    # PLOTTING FIT



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
        ax1.plot(x, y_fit, color=color, linewidth=2)
        ax1.scatter(x, y, s=35, color=color, alpha=0.7, linewidth=0, label=data_label)
    plt.xlabel('Time (h)', fontsize=15)
    plt.ylabel('Normalised fibril mass', fontsize=15)
    plt.legend()
    plt.show()





