# numerical simulation of an oligomer model that explains desired features: sharp transition in the equilibrium oligomer concentration when increasing m0; saturates fibril mass concentration (monomer concentration in equilibrium is constant irrespective of m0)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping
import pandas as pd
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod

class OligomerModel(ABC):
    # subclasses will be particular kinetic models, that need
    # i) kinetic parameters
    # ii) mass conservation constraint
    # iii) ODEs

    param_names: list[str] = []

    def __init__(self, fixed_params: dict):
        self.fixed = fixed_params    # global constants during fitting


    @abstractmethod
    def get_free_monomer(self, M: float, m0: float, free_params: dict) -> float:
        pass

    @abstractmethod
    def odes(self, t: float, y: list, m0: float, free_params: dict) -> list:
        pass

    # simulation concrete method: do this on a grid and later interpolate to fit to the data because not all datasets may have the same time points
    def simulate(self, m0: float, free_params: dict, tend: float = 200, n_grid: int = 1000):
        y0 = [0.0, 0.0]
        t_grid = np.linspace(0, tend, n_grid)
        sol = solve_ivp(self.odes, (0,tend), y0, method='Rk45', t_eval=t_grid, rtol=1e-8, atol=1e-10, args=(m0, free_params))
        M_norm = sol.y[0] / m0
        return sol.t, M_norm

    def get_half_time(t, M_norm):
        for i in range(len(M_norm)-1):
            if M_norm[i] <= 0.5 & M_norm[i+1] >= 0.5:
                slope = (0.5 - M_norm[i]) / (M_norm[i+1] - M_norm[i])
                t_half = t[i] + slope * (t[i+1] - t[i])
                return t_half
        return None





def get_free_monomer(M, m0, Keq, n):
    # m from conservation of mass constraint
    # m + n * Keq * m**n + M = m0
    if Keq == 0:
        return max(m0 - M, 0)
    
    def equation(m):
        return m + n * Keq * m**n + M - m0
    m_max = max(m0-M, 0.0)
    if m_max <=0:
        return 0.0
    return brentq(equation, 0, m_max, xtol=1e-12)




def odes(t, y, m0, kn, k2, kp, nc, n2, Keq, n):    
    M, P = y
    m = get_free_monomer(M, m0, Keq, n)

    dP = kn * m**nc + k2 * m**n2 * M
    dM = 2 * kp * m * P

    return [dM, dP]

def simulate(m0, params, tend = 300,  n_grid = 1000):
    kn, k2, kp, nc, n2, Keq, n = params
    y0 =[0.0, 0.0]  # unseeded
    t_grid = np.linspace(0, tend, n_grid)
    sol = solve_ivp(odes, (0,tend), y0, method='RK45', t_eval = t_grid, rtol=1e-8, atol=1e-10, args=(m0, kn, k2, kp, nc, n2, Keq, n))
    M_norm = sol.y[0] / m0
    return sol.t, M_norm


def get_half_time(t, M_norm):
    for i in range(len(M_norm) - 1):
        if M_norm[i] <= 0.5 <= M_norm[i+1]:
            slope = (0.5 - M_norm[i]) / (M_norm[i+1] - M_norm[i])
            half_time = t[i] + slope * (t[i+1] - t[i])
            return half_time

m0vals = np.array([1.1, 1.4, 1.9, 2.5, 3.4, 4.5, 6.0])   # same values we have in the data
params = (0.1,  # kn
          100,  # k2
          1e-5,   # kp
          2.0,   #nc
          2.0,   # n2
          3e-3,   # Keq
          5     # n, oligomer size
        )
half_times = []
for m0 in m0vals:
    t, M_norm = simulate(m0, params)
    half_times.append(get_half_time(t, M_norm))

tend=200
half_times = np.array(half_times)
plt.scatter(np.log10(m0vals), np.log10(half_times))
plt.xlabel('log10(m0)')
plt.ylabel('log10(t_half)')
plt.show()


for m0 in m0vals:
    t, M_norm = simulate(m0, params)
    plt.plot(t, M_norm, label=f'{m0} µM')

plt.xlabel('Time (h)')
plt.ylabel('Normalised fibril mass')
plt.legend(fontsize=7)
plt.show()

# attempt fitting without an analytical solution
data_path = '/Users/nataliaionescu/Desktop/AB42_project/fits/pH_6.5/6.5_without_1.4.tsv'
df = pd.read_csv(data_path, sep='\t', header=1)
df_with_header = pd.read_csv(data_path, sep='\t', header=None)
x_actual_data=[]    # time
y_actual_data=[]    # signal
for i in range((len(df.columns) //2)):
    x = df.iloc[:, i*2].values
    y = df.iloc[:, i*2+1].values
    x_actual_data.append(x[~np.isnan(x)])
    y_actual_data.append(y[~np.isnan(y)])

m0vals_data = [float(df_with_header.iloc[0,i*2].split(': ')[-1]) for i in range(len(df.columns)//2)]
print(m0vals_data)
m0vals_data = np.array(m0vals_data)
tend_data = max(x[-1] for x in x_actual_data) * 2   # simulate long enough
n2 = 2.0
nc = 2.0
n = 5   # keep these fixed

def objective(log_free_params):
    # will search through log of parameter space for basinhopping to be more efficient
    kn, k2, kp, Keq = [10**logp for logp in log_free_params]
    params = (kn, k2, kp, nc, n2, Keq, n)   # args for simulate()
    total_loss = 0.0
    for x_data, y_data, m0 in zip(x_actual_data, y_actual_data, m0vals_data):
        try:
            t_sim, M_sim = simulate(m0, params, tend=tend_data, n_grid=300)
            if not np.all(np.isfinite(M_sim)) or M_sim[-1] < 0.5:    # simulation explodes or doesn't plateau
                return 1e10    # huge loss, not valid fit 
            interp = interp1d(t_sim, M_sim, bounds_error=False, fill_value=(0.0,1.0))
            # interpolate between simulated values to match measured times
            total_loss += np.sum((interp(x_data) - y_data)**2)
        except Exception as e:
            print(e)
            return 1e10
    return total_loss

init_guess = [0.2, 0, 1, -3]   # log space: kn, k2, kp, Keq
fit = basinhopping(objective,init_guess, niter=10)
fitted_params = 10**(fit.x)

m0_unique = sorted(set(m0vals_data))
palette = sns.color_palette('tab10', n_colors = len(m0_unique))
color_map = {m0: palette[i] for i, m0 in enumerate(m0_unique)}
for m0 in m0_unique:
    color = color_map[m0]
    indices = np.where(m0vals_data == m0)[0]
    for i in indices:
        x_data = x_actual_data[i]
        y_data = y_actual_data[i]
        plt.scatter(x_data, y_data,color=color)
    all_params_fit = (fitted_params[0], fitted_params[1], fitted_params[2], nc, n2, fitted_params[3], n)
    t_sim, M_sim = simulate(m0, all_params_fit, tend=50)
    plt.plot(t_sim, M_sim, color=color)
    #plt.savefig('oligomer_fit.png')
plt.xlim(0, tend_data)
plt.show()

print(fitted_params)
print(fit.fun)


# TODO add progres bar to basinhopping procedure
# TODO add classes to easily change afterwards what oligomer model is being simulated/fit to
# TODO maybe define params as a dictionary to also print out their names





