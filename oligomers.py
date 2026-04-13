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
        sol = solve_ivp(self.odes, (0,tend), y0, method='RK45', t_eval=t_grid, rtol=1e-8, atol=1e-10, args=(m0, free_params))
        M_norm = sol.y[0] / m0
        return sol.t, M_norm

    def get_half_time(self, t, M_norm):
        for i in range(len(M_norm)-1):
            if M_norm[i] <= 0.5 <=  M_norm[i+1]:
                slope = (0.5 - M_norm[i]) / (M_norm[i+1] - M_norm[i])
                t_half = t[i] + slope * (t[i+1] - t[i])
                return t_half
        return None

class OffPathwayFastEq(OligomerModel):
    '''
    off-pathway oligomers that equilibrate instantly + elongation, primary nucleation, secondary nucleaion of fibrils
    '''
    param_names = ['kn', 'k2', 'kp', 'm_star']

    def get_free_monomer(self, M, m0, free_params: dict) -> float:
        # m from conservation of mass constraint
        # m + n * Keq * m**n + M = m0
        m_star = free_params['m_star']
        n = self.fixed['n']
        
        def constraint(m):
            return m + n * (m/m_star)**(n-1) * m +M - m0
        m_max = max(m0-M, 0.0)
        if m_max <=0:
            return 0.0
        return brentq(constraint, 0, m_max, xtol=1e-12)
    

    def odes(self, t, y, m0, free_params):
        M, P = y
        kn = free_params['kn']
        k2 = free_params['k2']
        kp = free_params['kp']
        nc = self.fixed['nc']
        n2 = self.fixed['n2']
        m = self.get_free_monomer(M, m0, free_params)

        dP = kn * m**nc + k2 * m**n2 * M
        dM = 2 * kp * m * P
    
        return [dM, dP]


class OffPathwayDelayed(OligomerModel):
    '''
    off_pathway oligomers that equilibrate at some point during the initial plateau + elongation, primary nucleation, secondary nucleation of fibrils
    '''

    param_names = ['kn', 'k2', 'kp', 'm_star', 'kominus']

    def get_free_monomer(self, M, m0, free_params: dict, S) -> float:
        n = self.fixed['n']
        return max(m0-M, 0)

    def odes(self, t, y, m0, free_params):
        M, P, S = y   # add oligomer dynamics as well
        kn = free_params['kn']
        k2 = free_params['k2']
        kp = free_params['kp']
        m_star = free_params['m_star']
        kominus = free_params['kominus']
        nc = self.fixed['nc']
        n2 = self.fixed['n2']
        n = self.fixed['n']
        koplus =  kominus / (n*m_star**(n-1))
        m = self.get_free_monomer(M, m0, free_params, S)

        dS = koplus * m**n - kominus * S
        dP = kn * m**nc + k2 * m**n2 * M
        dM = 2 * kp * m * P

        return [dM, dP, dS]

    def simulate(self, m0, free_params, tend=300, n_grid=1000):
        y0 = [0., 0., 0.]
        t_grid = np.linspace(0, tend, n_grid)
        sol = solve_ivp(self.odes, (0,tend), y0, method='RK45', t_eval=t_grid, rtol=1e-8, atol=1e-10, args=(m0, free_params))
        M_norm = sol.y[0] / m0
        S = sol.y[-1]
        return sol.t, M_norm, S



class OligomerFitter:
    def __init__(self, model: OligomerModel, x_data:list, y_data:list, m0vals: np.ndarray, niter: int = 10):
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.m0vals = m0vals
        self.niter = niter
        self.tend = max(x[-1] for x in x_data) * 1.5

    def basinhopping_callback(self, x, f, accept):
        self.iter_count += 1
        status = 'accepted' if accept else 'rejected'
        print(f'iteration {self.iter_count}: {status}')


    def log_params_to_dict(self, log_free_params: np.ndarray) -> dict:
        return {name: 10**lp for name, lp in zip(self.model.param_names, log_free_params)}

    def objective(self, log_free_params: np.ndarray) -> float:
        # will search through log of parameter space for basinhopping to be more efficient
        free_params = self.log_params_to_dict(log_free_params)
        total_loss = 0.0
        for x_data, y_data, m0 in zip(self.x_data, self.y_data, self.m0vals):
            try:
                sim_result = self.model.simulate(m0, free_params, tend=self.tend, n_grid=300)
                t_sim = sim_result[0]
                M_sim = sim_result[1]
                if not np.all(np.isfinite(M_sim)) or M_sim[-1] < 0.5:    # simulation explodes or doesn't plateau
                    return 1e10    # huge loss, not valid fit 
                interp = interp1d(t_sim, M_sim, bounds_error=False, fill_value=(0.0,1.0))
                # interpolate between simulated values to match measured times
                total_loss += np.sum((interp(x_data) - y_data)**2)
            except Exception as e:
                print(e)
                return 1e10
        return total_loss

   
    def fit(self, init_log_guess: list) -> dict:
        self.iter_count = 0
        print(f'running basinhopping with {self.niter} iterations')
        result = basinhopping(self.objective, init_log_guess, niter=self.niter, callback = self.basinhopping_callback)
        fitted_log = result.x
        fitted = self.log_params_to_dict(fitted_log)
        print(f'loss: {result.fun}')
        print('fitted parameters:')
        for k, v in fitted.items():    # dict
            print(f'{k} is {v}')
        return fitted, result

if __name__ == '__main__':
    #==============================================
    # simulating kinetic curves of different models
    #==============================================
    
    #===================================
    # fast equilibration model simulation
    #===================================
    fixed = dict(nc=2.0, n2=2.0, n=3)
    model = OffPathwayFastEq(fixed)

    free_params = dict(kn=1.0, k2=1.0, kp=1e-3, m_star=3.0)
    m0vals = np.array([1.1, 1.4, 1.9, 2.5, 3.4, 4.5, 6.0])

    half_times = []
    for m0 in m0vals:
        t, M_norm = model.simulate(m0, free_params)
        half_times.append(model.get_half_time(t, M_norm))

    half_times = np.array(half_times)
    plt.figure()
    plt.scatter(np.log10(m0vals), np.log10(half_times))
    plt.xlabel('log m0')
    plt.ylabel('log t_half')

    plt.figure()
    for m0 in m0vals:
        t, M_norm = model.simulate(m0, free_params)
        plt.plot(t, M_norm, label=f'{m0}')
    plt.xlabel('time (h)')
    plt.ylabel('normalised fibril mass')
    plt.title('Fast equilibration oligomers')
    plt.legend()
    plt.show()

    #======================================
    # delayed equilibration model simulation
    #======================================


    delayed_eq_model = OffPathwayDelayed(dict(nc=2.0, n2=2.0, n=10))
    free_params_delayed = dict(kn=0.1, k2=100, kp=1e-5, m_star=3, kominus=0.5)

    # plot fibril mass and oligomer concentration together
    fig, axes = plt.subplots(1,2)
    for m0 in m0vals:
        t, M_norm, S = delayed_eq_model.simulate(m0, free_params_delayed)
        axes[0].plot(t, M_norm, label=f'{m0}')
        axes[1].plot(t, S, label=f'{m0}')
    axes[0].set_xlabel('time (h)')
    axes[1].set_xlabel('time(h)')
    axes[0].set_ylabel('normalised fibril mass')
    axes[1].set_ylabel('oligomer concentration')
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # half time plot for delayed oligomer model
    half_times_delayed = []
    for m0 in m0vals:
        t, M_norm, _ = delayed_eq_model.simulate(m0, free_params_delayed)
        half_times_delayed.append(model.get_half_time(t, M_norm))
    half_times_delayed = np.array(half_times_delayed)
    plt.figure()
    plt.scatter(np.log10(m0vals), np.log10(half_times_delayed))
    plt.xlabel('log m0')
    plt.ylabel('log t_half')
    plt.title('Delayed oligomers half time plot')
    plt.show()

    #===============================================
    # read normalized data
    #===============================================
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

    #==============================
    # fit to delayed oligomer model
    #==============================
    fitter = OligomerFitter(model=OffPathwayDelayed(dict(nc=2.0,n2=2.0,n=10)),
                            x_data = x_actual_data,
                            y_data = y_actual_data,
                            m0vals = m0vals_data,
                            niter=10)
    # initial guesses in log space: kn, k2, kp, m_star, kominus
    fitted_params, fit_result = fitter.fit([-1, 2, -5, 0, 1])    # LOG SPACE!!!!
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
        t_sim, M_sim, _ = fitter.model.simulate(m0, fitted_params, tend=50)
        plt.plot(t_sim, M_sim, color=color, label=f'{m0}')

    plt.xlim(0, fitter.tend)
    plt.legend()
    plt.show()



    #=========================================
    # fit to fast equilibration oligomer model
    #=========================================


    fitter = OligomerFitter(model=OffPathwayFastEq(dict(nc=2.0,n2=2.0,n=5)),
                            x_data = x_actual_data,
                            y_data = y_actual_data,
                            m0vals = m0vals_data,
                            niter=10)

    # initial guess in log space: kn, k2, kp, m_star
    fitted_params, fit_result = fitter.fit([1,1,1,0])


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
        t_sim, M_sim = fitter.model.simulate(m0, fitted_params, tend=50)
        plt.plot(t_sim, M_sim, color=color, label=f'{m0}')

    plt.xlim(0, fitter.tend)
    plt.legend()
    plt.show()



