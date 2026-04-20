import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from OligomerMethods import OligomerModel, OffPathwayDelayed
from pca import load_data

# fitting in a lower dimensional space will likely not require basinhopping
# maybe this makes numerical fitting tractable
# try with the simple off-pathway oligomer model

call_count = 0

# load data
data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv'
x_data, y_data, m0vals = load_data(data_path)
tend = max(x[-1] for x in x_data)
t_axis = np.linspace(0, tend, 300)
data_matrix = [np.interp(t_axis, x, y) for x, y in zip(x_data, y_data)]
data_matrix = np.array(data_matrix)

pca = PCA()
pca.fit(data_matrix)
# look at the reduced problem in 2D since the first two PCs explain most of the variance in the data
n_pc = 2
data_scores = pca.transform(data_matrix)
weights = np.sqrt(pca.explained_variance_[:n_pc])

# create oligomer model
model = OffPathwayDelayed({'kp': 1, 'm_star': 2, 'n': 3, 'n2': 0})

def objective(log_free_params):
    global call_count; call_count +=1
    kn = 10**log_free_params[0]
    kominus = 10 ** log_free_params[1]
    nc = 10 ** log_free_params[2]
    free_params = {'kn': kn, 'kominus': kominus, 'nc': nc}
    print(f'call {call_count}; kn is {kn:.2e}, kominus is {kominus:.2e}, nc is {nc:.2e}')
    sim_data_matrix = []
    for m0 in m0vals:
        try:
            sim_result = model.simulate(m0, free_params, tend=tend, n_grid=100)
            t_sim = sim_result[0]
            M_sim = sim_result[1]
            if not np.all(np.isfinite(M_sim)) or M_sim[-1] < 0.5:    # simulation explodes or doesn't plateau
                return 1e10    # huge loss, not valid fit
            sim_data_matrix.append(interp1d(t_sim, M_sim, bounds_error=False, fill_value=(0.0, 1.0))(t_axis))
        except Exception as e:
            print(e)
            return 1e10
    sim_data_matrix = np.array(sim_data_matrix)
    '''compare PCA scores of entire data matrix from simulation vs actual data matrix'''
    sim_scores = pca.transform(sim_data_matrix)
    residuals = (data_scores[:, :n_pc] - sim_scores[:, :n_pc]) * weights
    return float(np.dot(residuals.ravel(), residuals.ravel()))

#========
# FITTING
#========
# init guess in LOG SPACE!!! kn, kominus, nc
initial_guess = [1, 1, np.log10(2)]
result = minimize(objective, initial_guess, method='Nelder-Mead')
kn_fit = 10**result.x[0]
kominus_fit = 10**result.x[1]
nc_fit = 10**result.x[2]
print(kn_fit, kominus_fit, nc_fit)

fitted_params = {'kn': kn_fit, 'kominus': kominus_fit, 'nc': nc_fit}
colors    = plt.cm.viridis(np.linspace(0, 1, len(np.unique(m0vals))))
color_map = {m0: colors[i] for i, m0 in enumerate(sorted(np.unique(m0vals)))}

fig, ax = plt.subplots(figsize=(9, 5))
seen = set()
for x, y, m0 in zip(x_data, y_data, m0vals):
    ax.scatter(x, y, color=color_map[m0], s=8, alpha=0.4,
               label=f'm0={m0}' if m0 not in seen else None)
    seen.add(m0)
for m0 in sorted(np.unique(m0vals)):
    t_sim, M_sim, *_ = model.simulate(m0, fitted_params, tend=tend, n_grid=500)
    ax.plot(t_sim, M_sim, color=color_map[m0], lw=2)

ax.set_xlabel('Time')
ax.set_ylabel('Normalised mass concentration')
ax.set_title(f'kn={kn_fit:.2e}   kominus={kominus_fit:.2e}   nc={nc_fit:.2f}')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()





