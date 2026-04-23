import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA

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
    param_names = ['kn']

    def get_free_monomer(self, M, m0, free_params: dict) -> float:
        # m from conservation of mass constraint
        # m + n * Keq * m**n + M = m0
        m_star = self.fixed['m_star']
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
        k2 = 100 * kn
        kp = self.fixed['kp']
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
    #param_names = ['nc','kn', 'kominus']
    param_names = ['nc', 'kn', 'm_star']   # fitting mstar instead of kominus (only one of them should be fit)

    def get_free_monomer(self, M, m0, free_params: dict, S) -> float:
        n = self.fixed['n']
        S = max(0, S)
        m = max(m0 - n*S - M, 0)
        return m


    def odes(self, t, y, m0, free_params):
        M, P, S = y   # add oligomer dynamics as well
        kn = free_params['kn']
        k2 = 1000 * kn
        kp = self.fixed['kp']
        #m_star = self.fixed['m_star']
        m_star = free_params['m_star']
        n = self.fixed['n']
        #kominus = free_params['kominus']
        kominus = self.fixed['kominus']
        nc = free_params['nc']
        n2 = self.fixed['n2']
        koplus =  kominus / (n*m_star**(n-1))
        m = self.get_free_monomer(M, m0, free_params, S)

        dS = (kominus/n) * (m / m_star)**n - kominus * S
        dP = kn * m**nc + k2 * m**n2 * M
        dM = 2 * kp * m * P

        return [dM, dP, dS]

    def simulate(self, m0, free_params, tend=3, n_grid=1000):
        y0 = [0., 0., 0.]
        n = self.fixed['n']
        t_grid = np.linspace(0, tend, n_grid)
        sol = solve_ivp(self.odes, (0,tend), y0, method='RK45', t_eval=t_grid, rtol=1e-8, atol=1e-10, args=(m0, free_params))
        M = sol.y[0]
        M_norm = M / m0
        S = sol.y[-1]
        free_m = m0 - M - n * S
        return sol.t, M_norm, S, free_m

class OnPathwayCMC(OligomerModel):
    # kn, k2, n2 and m_star are free
    param_names = ['kn', 'k2', 'n2', 'm_star']
    def get_free_monomer(self, M, m0, free_params: dict):
        m_star = free_params['m_star']
        n = self.fixed['n']
        m_max = m0-M
        if m_max <= 0:
            return 0
        def constraint(m):
            return m + n * (m/m_star)**(n-1)*m + M - m0
        return brentq(constraint, 0.0, m_max, xtol = 1e-10)


    def odes(self, t, y, m0, free_params):
        M, P = y
        kn = free_params['kn']
        k2 = free_params['k2']
        n2 = free_params['n2']
        m_star = free_params['m_star']
        kp = self.fixed['kp']
        nc = self.fixed['nc']

        m = self.get_free_monomer(M, m0, free_params)
        # saturating term: effective scaling with m0
        m_eff = (m * m_star) / (m + m_star)
        dP = kn * m_eff**nc + k2 * m_eff**n2 * M
        dM = 2 * kp * P
        return [dM, dP]







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
        print('objective called with', free_params)
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

class PCAFitter:
    def __init__(self, model: OligomerModel, x_data:list, y_data:list, m0vals: np.ndarray, niter: int = 7):
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.m0vals = m0vals
        self.niter = niter
        self.tend = max(x[-1] for x in x_data)
        self.t_axis = np.linspace(0, self.tend, 300)
        self.n_pc = 2
        self.pca = None

    def basinhopping_callback(self, x, f, accept):
        self.iter_count += 1
        status = 'accepted' if accept else 'rejected'
        print(f'iteration {self.iter_count}: {status}')

    def data_pca_scores(self):
        t_axis = self.t_axis
        n_pc = self.n_pc
        data_matrix = [np.interp(t_axis, x, y) for x, y in zip(self.x_data, self.y_data)]
        data_matrix = np.array(data_matrix)
        self.data_mean = data_matrix.mean(axis=0)
        centered = data_matrix - self.data_mean

        #data_matrix = np.array(data_matrix)
        #data_matrix = np.array(centered)
        pca = PCA()
        #pca.fit(data_matrix)
        pca.fit(centered)
        self.pca = pca
        self.data_scores = pca.transform(data_matrix)
        self.weights = np.sqrt(pca.explained_variance_[:n_pc])

    def log_params_to_dict(self, log_free_params: np.ndarray) -> dict:
        return {name: 10**lp for name, lp in zip(self.model.param_names, log_free_params)}

    def objective(self, log_free_params: np.ndarray) -> float:
        # basinhopping makes kominus assume huge values, which makes the ODE system very stiff, which then takes forever to simulate and numerical fitting becomes intractable since each initial guess takes several hours
        # a temporary fix: force parameters to be within some reasonable bounds and return a huge error if they are not, so basinhopping can hopefully move on faster
        # nc, kn, kominus bounds in log space
        log_param_bounds = {'nc': (0.0, 1.5), 'kn':(-4, 4), 'm_star': (-3,3)}
        for i, name in enumerate(self.model.param_names):
            low, high = log_param_bounds[name]
            if not (low <= log_free_params[i] <= high):
                return 1e10




        t_axis = self.t_axis
        n_pc = self.n_pc
        if self.pca is None:
            self.data_pca_scores()

        free_params = self.log_params_to_dict(log_free_params)
       
        print('objective called with', free_params)

        sim_data_matrix = []
        for m0 in self.m0vals:
            try:
                sim_result = self.model.simulate(m0, free_params, tend = self.tend, n_grid=200)
                t_sim = sim_result[0]
                M_sim = sim_result[1]
                if not np.all(np.isfinite(M_sim)) or M_sim[-1] < 0.5:    # simulation explodes or doesn't plateau
                    return 1e10    # huge loss, not valid fit
                sim_data_matrix.append(interp1d(t_sim, M_sim, bounds_error=False, fill_value=(0.0, 1.0))(t_axis))
            except Exception as e:
                print(e)
                return 1e10
        sim_data_matrix = np.array(sim_data_matrix)
        self.sim_mean = sim_data_matrix.mean(axis=0)
        sim_centered = sim_data_matrix - self.sim_mean
        sim_scores = self.pca.transform(sim_centered)
        #sim_scores = self.pca.transform(sim_data_matrix)
        # compare PCA scores of entire data matrix from simulation vs actual data matrix; the difference must be weighted by the relative importance of the PCs in describing the data
        sim_reconstructed = self.pca.inverse_transform(sim_scores)
        data_reconstructed = self.pca.inverse_transform(self.data_scores)
        residuals = sim_reconstructed - data_reconstructed
        #residuals = (self.data_scores[:, :n_pc] - sim_scores[:, :n_pc]) * self.weights
        return float(np.dot(residuals.ravel(), residuals.ravel()))


    def fit(self, init_log_guess: list) -> dict:
        self.iter_count = 0
        print(f'running basinhopping with {self.niter} iterations')
        minimizer_kwargs = {'method': 'Nelder-Mead'}
        result = basinhopping(self.objective, init_log_guess, niter=self.niter, callback = self.basinhopping_callback, minimizer_kwargs=minimizer_kwargs)
        fitted_log = result.x
        fitted = self.log_params_to_dict(fitted_log)
        print(f'loss: {result.fun}')
        print('fitted parameters:')
        for k, v in fitted.items():    # dict
            print(f'{k} is {v}')
        return fitted, result

