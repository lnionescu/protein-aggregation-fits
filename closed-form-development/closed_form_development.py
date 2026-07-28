import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping, brentq
from dataclasses import dataclass

# dimensional params
m0 = 5
n = 100  # oligomer size
kp = 1   # this will always be 1
k1 = 0.1
k2 = 1
nc = 2
n2 = 0
nk = 100   # exponent in oligomer formation from monomers
#ko_plus = 10   # oligomer formation
ko_minus = 5  # oligomer dissociation
alphaminus = ko_minus
m_star = 3   # set this to compute alphaplus directly from m_star, nk and alphaminus
ko_plus = ko_minus * m_star**(1-nk)
#alphaplus = lambda m:  ko_plus


# rates
def alpha1(m):
    return k1 * m**nc

def alpha2(m):
    return k2 * m**n2

def alphae(m):
    return 2 * kp * m

def alphaplus(m):
    return ko_plus * m**nk


def eps(m0):
    return alpha1(m0) / (2 * m0 * alpha2(m0))

def kappa(m0):
    return np.sqrt(alphae(m0) * alpha2(m0))


def qss_free_monomer(m0_val, m_star_val, nk_val, n_val):
    """Solve QSS polynomial for initial free monomer concentration.

    Exact QSS condition (dS/dt = 0, mass conservation):
        m_free + n * m_star^(1-nk) * m_free^nk = m0

    Returns m_free_qss (dimensional).  For nk -> inf this converges to m_star
    (the large-nk / CMC approximation used previously).
    """
    if m0_val <= m_star_val:
        return m0_val   # below CMC: all protein as free monomer, no oligomers
    prefactor = n_val * m_star_val ** (1 - nk_val)   # = n * ko_plus / ko_minus
    def f(m_free):
        return m_free + prefactor * m_free ** nk_val - m0_val
    # root is guaranteed in (0, m0); solution < m_star for large nk
    return brentq(f, 1e-14, m0_val * (1 - 1e-12))


print('kappa is', kappa(m0))


# simulate the system in its dimensional form
def rhs_dim(t, y):
    P, M, S = y
    m = m0 - n*S - M
    m = max(m, 0.0)   # don't let this drift negative from numerics
    dP = alpha1(m) + alpha2(m) * M
    dM = alphae(m) * P
    dS = alphaplus(m) - ko_minus * S

    return [dP, dM, dS]


tend = 7
n_grid = 500
t_grid = np.linspace(0, tend, n_grid)

# initial condition for numerical simulation
Gamma = ko_minus / kappa(m0)
Lambda = n * alphaplus(m0) / (m0 * kappa(m0))
my_Omega = Gamma / (Lambda + Gamma)
Omega = m_star / m0
#Omega = Lambda / (Lambda + Gamma)

print('my Omega is', my_Omega)
print('proper Omega is', Omega)
print('Gamma is', Gamma)

# exact QSS polynomial root — consistent with model_analytical
m_free_qss = qss_free_monomer(m0, m_star, nk, n)
S0 = (m0 - m_free_qss) / n
# large-nk CMC approximation (previous):  S0 = (m0 - m_star) / n

y0 = [0, 0, S0]   # change ICs
sol = solve_ivp(rhs_dim, (0, tend), y0, method='Radau', t_eval=t_grid)

# plot normalized fibril mass and oligomer concentration
plt.figure()
plt.plot(sol.t, sol.y[1]/m0, color='red', label='M/m0')
plt.plot(sol.t, n*sol.y[-1], color='blue', label='S')
plt.legend()
plt.show()

# plot monomer concentration and oligomer concentration, dimensional form, against dimensional time
plt.figure()
plt.plot(sol.t, sol.y[-1], color='blue', label='S')
plt.plot(sol.t, m0-sol.y[1] - sol.y[2], color='orange', label='m')
plt.xlabel('t')
plt.legend()
plt.show()

# nondimensionalize numerical output
t_num = sol.t
P_num = sol.y[0]
M_num = sol.y[1]
S_num = sol.y[2]

m_num = np.ones(len(t_num)) * m0 - n*S_num - M_num

# nondimensional time
tau = kappa(m0) * t_num

# nondimensional fibril number concentration
Pi = alphae(m0) / (kappa(m0) * m0) * P_num

# nondimensional monomer
mu = m_num / m0

# nondimensional oligomer
sigma = n*S_num / m0

# chi = mu + sigma
chi = mu + sigma

# early time and global analytical solutions
lam = np.sqrt(Omega**(n2+1))
print('lam is', lam)
M_early = 2 * eps(m0) * Omega**(nc-n2) * (np.cosh(lam*tau) - 1)
plt.figure()
valid = M_num/m0 < 0.2   # early times
plt.plot(tau[valid], M_num[valid]/m0, 'r--', label='numerical')
plt.plot(tau[valid], M_early[valid], 'b--', label='analytical')
plt.legend()
plt.xlabel('tau')
plt.ylabel('M(tau)/m0')
plt.title("Early time normalized fibril mass")
plt.show()

c = Gamma / lam
#if Gamma < 1:
#    c = Gamma / lam
#else:
#    c = 3 / (2*n2 + 1)   # saturating secondary nucleation

M_global = 1 - (1 + 2*eps(m0)*Omega**(nc-n2)/c * (np.cosh(lam*tau) - 1))**(-c)

plt.figure()
plt.plot(tau, M_num/m0, 'r-', label='numerical')
plt.plot(tau, M_global, 'b--', label='global analytical')
plt.xlabel('tau')
plt.ylabel('M(tau)/m0')
plt.legend()
plt.show()

print('Karlovitz number is', ko_minus / kappa(m0))


# ─── FITTING ────────────────────────────────────────────────────────────────

@dataclass
class KineticParameters:
    k1: float        # primary nucleation prefactor
    k2: float        # secondary nucleation prefactor
    kp: float        # elongation rate, fix at 1
    ko_minus: float  # oligomer dissociation rate
    nc: float        # primary nucleation order
    n2: float        # secondary nucleation order
    nk: float        # oligomer formation order
    n: float         # oligomer size
    m_star: float    # CMC


def model_analytical(t, m0, p: KineticParameters):
    ko_plus = p.ko_minus * p.m_star ** (1 - p.nk)
    m_free_qss = qss_free_monomer(m0, p.m_star, p.nk, p.n)
    Omega = m_free_qss / m0   # exact QSS, not the nk->inf approximation m_star/m0

    kappa_val = np.sqrt(2 * p.kp * m0 * p.k2 * m0 ** p.n2)
    eps_val = p.k1 * m0 ** p.nc / (2 * m0 * p.k2 * m0 ** p.n2)
    lam = np.sqrt(Omega ** (p.n2 + 1))
    Gamma = p.ko_minus / kappa_val

    #c = Gamma / lam if Gamma < 1 else 3 / (2 * p.n2 + 1)
    c = Gamma / lam

    tau = kappa_val * t
    arg = np.clip(lam * tau, 0, 500)    # avoid overflow in analytical solution
    return 1 - (1 + 2 * eps_val * Omega ** (p.nc - p.n2) / c * (np.cosh(arg) - 1)) ** (-c)


def minimize_analytical(guess, x_actual, y_actual, m0vals, init_guesses, free_params, fixed_params):
    fitted = {name: np.exp(guess[i]) * init_guesses[i] for i, name in enumerate(free_params)}
    all_params = {**fixed_params, **fitted}
    params = KineticParameters(**all_params)
    residual = 0.0
    for i in range(len(x_actual)):
        y_model = model_analytical(x_actual[i], m0vals[i], params)
        residual += np.sum((y_actual[i] - y_model)**2)
    return residual


if __name__ == '__main__':
    # load data

    # use for old pH 6 data:
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6_proper_norm.tsv'
    case = 'pH_6_old'

    # use for old pH 6.5 data:
    #data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv'
    #case = 'pH_6.5_old'

    # use for new pH 6.5 data:
    #data_path = '/Users/nataliaionescu/Desktop/AB42_project/normalized-data/pH_6.5_june.tsv'
    #case = 'pH_6.5_new'

    # use for new pH 6 data:
    #data_path = TODO
    #case = 'pH_6_new'

    import sys
    import seaborn as sns
    sys.path.append('/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits')
    from richards import load_data
    x_data, y_data, m0vals = load_data(data_path)

    # remove m0=0.8 curves; remove the third m0=1.4 replicate (old pH 6 data)
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

    # for new pH 6.5 data: remove m0=1.25, first m0=1.75, third m0=2.5
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

        # try removing all m0=2.5 curves:
        #mask = m0vals != 2.5
        #x_data = [x for x, m0 in zip(x_data, m0vals) if m0 != 2.5]
        #y_data = [y for y, m0 in zip(y_data, m0vals) if m0 != 2.5]
        #m0vals = m0vals[mask]

        drop_idx = np.where(m0vals == 2.5)[0][2]
        keep = np.arange(len(m0vals)) != drop_idx
        x_data = [x for x, k in zip(x_data, keep) if k]
        y_data = [y for y, k in zip(y_data, keep) if k]
        m0vals = m0vals[keep]

    # free and fixed params
    free_params = ['k1', 'k2', 'ko_minus']
    initial_guesses = [1, 1, 1, 2]
    fixed_params = {'n': 100, 'nk': 100, 'kp': 1, 'm_star': 3, 'n2': 0, 'nc': 2}

    seed = 42
    res = basinhopping(
        minimize_analytical,
        np.zeros(len(initial_guesses), dtype=float),
        niter=100,
        stepsize=0.3,
        seed=seed,
        minimizer_kwargs={
            'method': 'Nelder-Mead',
            'tol': 1e-10,
            'args': (x_data, y_data, m0vals, initial_guesses, free_params, fixed_params),
        },
    )
    fitted_values = {name: initial_guesses[i] * np.exp(res.x[i]) for i, name in enumerate(free_params)}
    print('fitted values', fitted_values)
    total_data_points = sum(len(y) for y in y_data)
    print('residual', res.fun)
    print('mean residual error', res.fun / total_data_points)

    # plot preparation
    all_params = {**fixed_params, **fitted_values}
    params_fit = KineticParameters(**all_params)

    unique_m0 = sorted(set(m0vals))
    palette = sns.color_palette('tab10', n_colors=len(unique_m0))
    color_map = {m0: palette[i] for i, m0 in enumerate(unique_m0)}
    seen_m0 = set()

    # plotting fit
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
        m0 = m0vals[i]
        color = color_map[m0]

        data_label = f'{m0}' if m0 not in seen_m0 else None
        seen_m0.add(m0)

        y_fit = model_analytical(x, m0vals[i], params_fit)
        ax1.plot(x, y_fit, color=color, linewidth=2)
        ax1.scatter(x, y, s=35, color=color, alpha=0.7, linewidth=0, label=data_label)
    plt.xlabel('Time (h)', fontsize=15)
    plt.ylabel('Normalised fibril mass', fontsize=15)
    plt.legend()
    plt.show()
