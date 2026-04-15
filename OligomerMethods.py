import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping
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
    # FREE PARAMS FOR SIMULATION AND MORE IMPORTANTLY FITTING
    param_names = ['kn', 'k2', 'kp']

    def get_free_monomer(self, M, m0, free_params: dict, S) -> float:
        n = free_params['n']
        return max(m0-n*S-M, 0)

    def odes(self, t, y, m0, free_params):
        M, P, S = y   # add oligomer dynamics as well
        kn = free_params['kn']
        k2 = free_params['k2']
        kp = free_params['kp']
        m_star = self.fixed['m_star']
        n = self.fixed['n']
        kominus = self.fixed['kominus']
        nc = self.fixed['nc']
        n2 = self.fixed['n2']
        koplus =  kominus / (n*m_star**(n-1))
        m = self.get_free_monomer(M, m0, free_params, S)

        dS = koplus * m**n - kominus * S
        dP = kn * m**nc + k2 * m**n2 * M
        dM = 2 * kp * m * P

        return [dM, dP, dS]

    def simulate(self, m0, free_params, tend=3, n_grid=1000):
        y0 = [0., 0., 0.]
        n = free_params['n']
        t_grid = np.linspace(0, tend, n_grid)
        sol = solve_ivp(self.odes, (0,tend), y0, method='RK45', t_eval=t_grid, rtol=1e-8, atol=1e-10, args=(m0, free_params))
        M = sol.y[0]
        M_norm = M / m0
        S = sol.y[-1]
        free_m = m0 - M - n * S
        return sol.t, M_norm, S, free_m



class OligomerFitter:
    def __init__(self, model: OligomerModel, x_data:list, y_data:list, m0vals: np.ndarray, niter: int = 7):
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.m0vals = m0vals
        self.niter = niter
        self.tend = max(x[-1] for x in x_data)

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
