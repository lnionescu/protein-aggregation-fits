import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import basinhopping

def load_data(path):
    df = pd.read_csv(path, sep='\t', header=1)
    hdr = pd.read_csv(path, sep='\t', header=None)
    x_data, y_data = [], []
    for i in range(len(df.columns) // 2):
        x = df.iloc[:, i*2].values
        y = df.iloc[:, i*2+1].values
        x_data.append(x[~np.isnan(x)])
        y_data.append(y[~np.isnan(y)])
    m0 = np.array([float(hdr.iloc[0, i*2].split(': ')[-1]) for i in range(len(df.columns)//2)])
    return x_data, y_data, m0

def m_eff(m0, m_star, n):
    return m0 / (1 + (m0/m_star)**(n - 1))
    # the phenomenological exponents n will generally be different for m_eff_prim and m_eff_sec related processes
    # note that this has a ton of parameters so maybe not Occam's razor friendly

def model(t, m0, p):
    """
    decoupling eps and kappa to be able to get both a long lag phase for lower m0 vals and sharp growth for all curves
    primary nucleation sees m_eff_prim
    secondary nucleation and elongation see m_eff_sec... may be differently affected since these processes happen on surfaces vs in solution
    c is currently a free parameter, but could maybe be physically justified TBA 
     """
    kn = 10**p[0]
    k2 = 10**p[1]
    nc, n2 = p[2], p[3]
    c = p[4]
    n_prim, m_star_prim = p[5], 10**p[6]
    n_sec, m_star_sec = p[7], 10**p[8]
    
    m_eff_prim = m_eff(m0, m_star_prim, n_prim)
    m_eff_sec = m_eff(m0, m_star_sec, n_sec)
    
    eps = kn * m_eff_prim**(nc-1) / (2 * k2 * m_eff_prim**n2)
    kappa = np.sqrt(2 * kn * k2 * m_eff_sec**(n2 + 1))
   
    # cap the exponent c so evaluated brackets don't blow up or vanish
    if c < 0.05: c = 0.05
    if c > 5: c = 5
    
    kt = kappa * t
    exp_term = np.exp(kt) + np.exp(-kt) - 2
    bracket = 1 + (eps/c) * exp_term
    bracket = np.maximum(bracket, 1e-15)
    return np.clip(1 - bracket**(-c), 0, 1)

if __name__ == "__main__":
    path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv'
    x_data, y_data, m0 = load_data(path)
    total_pts = sum(len(y) for y in y_data)
    # p = [log10(kp*kn), log10(kp*k2), nc, n2, c,
    #      n_prim, log10(m*_prim), n_sec, log10(m*_sec)]
    
    # initial guess: rate parameters in log space, scaling exponents in real space
    # hot start
    x0 = [np.log10(0.178), np.log10(1e7), 1.01, 0.07, 0.5622, 30, np.log10(1.72), 6.9, np.log10(5.24)]
    #x0 = [-0.7, 2.5, 1.0, 0.07, 0.56, 25, np.log10(1.7), 7, np.log10(5.3)]
    #bounds = [(-3, 1), (1, 5), (0.8, 3), (0.03, 0.5), (0.2, 1.0),
              #(8, 30), (np.log10(0.5), np.log10(3.5)),
              #(3, 15), (np.log10(2), np.log10(10))]
    bounds = None


    best = {'mse': np.inf, 'p': None}
    def obj(p):
        err = 0
        for x, y, m in zip(x_data, y_data, m0):
            y_fit = model(x, m, p)
            err += np.sum((y - y_fit)**2)
        mse = err / total_pts
        if mse < best['mse']:
            best['mse'] = mse
            best['p'] = p.copy()
            print(f"  MSE = {mse:.6f}, m*_prim = {10**p[6]:.2f}, m*_sec = {10**p[8]:.2f}")
        return err
    
    res = basinhopping(obj, x0, niter=500, stepsize=0.1, seed=42,
                       minimizer_kwargs={'method':'Nelder-Mead', 'bounds':bounds,
                                        'options':{'maxiter':10000, 'ftol':1e-15}})
    
    p = best['p']
    mse = best['mse']
    
    print(f"\nMSE = {mse:.6f}")
    print(f"kp*kn = {10**p[0]:.4e}, kp*k2 = {10**p[1]:.4e}")
    print(f"nc = {p[2]:.2f}, n2 = {p[3]:.2f}, c = {p[4]:.4f}")
    print(f"Primary partition:   n = {p[5]:.1f}, m* = {10**p[6]:.2f} μM")
    print(f"Secondary partition: n = {p[7]:.1f}, m* = {10**p[8]:.2f} μM")
    
    for m in [0.8, 1.4, 1.9, 2.5, 3.4, 4.5, 6.0]:
        mp = m_eff(m, 10**p[6], p[5])
        ms = m_eff(m, 10**p[8], p[7])
        ep = 10**p[0] * mp**(p[2]-1) / (2 * 10**p[1] * mp**p[3])
        ka = np.sqrt(2 * 10**p[0] * 10**p[1] * ms**(p[3]+1))
        print(f"{m:6.1f}  {mp:8.4f}  {ms:8.4f}  {ep:10.2e}  {ka:8.2f}")
    
    sns.set_theme(style='whitegrid', context='paper')
    unique_m0 = sorted(set(m0))
    colors = sns.color_palette('tab10', len(unique_m0))
    cmap = {m: colors[i] for i, m in enumerate(unique_m0)}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    t = np.linspace(0, 3.2, 250)
    seen = set()
    
    for x, y, m in zip(x_data, y_data, m0):
        c_color = cmap[m]
        lbl = f'{m:.1f} μM' if m not in seen else None
        seen.add(m)
        ax.scatter(x, y, s=20, color=c_color, alpha=0.5, linewidth=0, zorder=5)
        ax.plot(t, model(t, m, p), color=c_color, linewidth=2.2, label=lbl)
    
    ax.set_xlabel('Time (h)', fontsize=13)
    ax.set_ylabel('M(t)/m0', fontsize=13)
    ax.set_title(f'Oligomers affecting steps differently  |  MSE = {mse:.5f}  |  '
                f'm*_prim = {10**p[6]:.1f}, m*_sec = {10**p[8]:.1f}',
                fontsize=12)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.05, 3.25)
    ax.legend(fontsize=10, loc='lower right', ncol=2)
    sns.despine()
    
    plt.tight_layout()
    plt.show()
