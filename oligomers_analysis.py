# simulation and fitting of oligomer models
# methods imported from OligomerMethods.py

import numpy as np
import matplotlib.pyplot as plt
dpi = 300
import seaborn as sns
import pandas as pd

print(np.log10(2))
print(np.log10(5))

from OligomerMethods import OffPathwayFastEq, OffPathwayDelayed, OligomerFitter

# m0 vals to simulate
M0VALS = np.array([1.1, 1.4, 1.9, 2.5, 3.4, 4.5, 6.0])

# simulate model without oligomer pre-equilibrium
def plot_delayed_model(m0vals: np.ndarray = M0VALS) -> None:
    model = OffPathwayDelayed(dict(nc=2.0, n2=2.0, kominus=10, m_star=3))
    free_params = dict(kn=5, k2=100, kp=1e-1, n=30)
    # plot for various n and kominus values 
    # fit n and not kominus

    m_star = model.fixed['m_star']
    n = free_params['n']
    kominus = model.fixed['kominus']
    # fibril mass and oligomer concentration side by side
    fig, axes = plt.subplots(1, 3)
    for m0 in m0vals:
        t, M_norm, S, free_m = model.simulate(m0, free_params)
        axes[0].plot(t, M_norm, label=f'{m0}')
        axes[1].plot(t, S, label=f'{m0}')
        axes[2].plot(t, free_m, label=f'{m0}')
    axes[0].set_xlabel('time (h)')
    axes[1].set_xlabel('time (h)')
    axes[0].set_ylabel('normalised fibril mass')
    axes[1].set_ylabel('oligomer concentration')
    axes[2].set_xlabel('time (h)')
    axes[2].set_ylabel('free monomer concentration')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(f'simulation_kominus={kominus}_n={n}.png', dpi=dpi)
    plt.show()
    # max slope of kinetic curve / slope at half time depends only on elongation and 2 nucleation rates, not on 1 nucleation
    # does the half time plot go flat because all steps are almost saturated?
    # small differences in plateau height
 
    # half-time scaling
    half_times = []
    for m0 in m0vals:
        t, M_norm, S, m_free = model.simulate(m0, free_params)
        half_times.append(model.get_half_time(t, M_norm))
    half_times = np.array(half_times)
 
    plt.figure()
    plt.scatter(np.log10(m0vals), np.log10(half_times))
    plt.xlabel('log m0')
    plt.ylabel('log t_half')
    plt.title('Delayed oligomers half time plot')
    plt.savefig(f'halftimes_kominus={kominus}_n={n}.png', dpi=dpi)
    plt.show()

# load real data
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


# plot fitted curves, colour coded by m0 value
def plot_fit(fitter: OligomerFitter, fitted_params: dict,
             x_actual_data: list, y_actual_data: list,
             m0vals_data: np.ndarray, tend_plot: float = 50):
    m0_unique = sorted(set(m0vals_data))
    palette = sns.color_palette('tab10', n_colors=len(m0_unique))
    color_map = {m0: palette[i] for i, m0 in enumerate(m0_unique)}
 
    for m0 in m0_unique:
        color = color_map[m0]
        indices = np.where(m0vals_data == m0)[0]
        for i in indices:
            plt.scatter(x_actual_data[i], y_actual_data[i], color=color)
        sim_result = fitter.model.simulate(m0, fitted_params, tend=tend_plot)
        t_sim, M_sim = sim_result[0], sim_result[1]
        plt.plot(t_sim, M_sim, color=color, label=f'{m0}')
 
    plt.xlim(0, fitter.tend)
    plt.xlabel('time (h)')
    plt.ylabel('normalised fibril mass')
    plt.legend()
    plt.show()
 


def fit_delayed_model(x_data, y_data, m0vals) -> dict:
    # create OligomerFitter from OffPathwayDelayed
    # basinhopping routine is handled inside OligomerMethods.py
    fitter = OligomerFitter(
        model=OffPathwayDelayed(dict(nc=2.0, n2=2.0, kominus=10, m_star=300)),
        x_data=x_data,
        y_data=y_data,
        m0vals=m0vals
    )
    # initial guesses in log space: kn, k2, kp, n
    fitted_params, _ = fitter.fit([0.7, 1, -1, 1])    # LOG SPACE !!!!!!!!!!!!
    plot_fit(fitter, fitted_params, x_data, y_data, m0vals)
    return fitted_params

# TODO try numerical solution fitting for a few initial guesses
# free params: m_star, initial guess around 3, and n, initial guess around 15
# TODO maybe set n outside of log space and force it to be an integer during fitting? or maybe not?

if __name__ == "__main__":
    plot_delayed_model()    # simulate: kinetic curves of M, S, m & half-time plot
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv' 
    x_data, y_data, m0vals = load_data(data_path) 
    fitted_params = fit_delayed_model(x_data, y_data, m0vals)
