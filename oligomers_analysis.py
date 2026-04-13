# simulation and fitting of oligomer models
# methods imported from OligomerMethods.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from OligomerMethods import OffPathwayFastEq, OffPathwayDelayed, OligomerFitter

# m0 vals to simulate
M0VALS = np.array([1.1, 1.4, 1.9, 2.5, 3.4, 4.5, 6.0])

# simulate model without oligomer pre-equilibrium
def plot_delayed_model(m0vals: np.ndarray = M0VALS) -> None:
    model = OffPathwayDelayed(dict(nc=2.0, n2=2.0, n=10))
    free_params = dict(kn=0.1, k2=100, kp=1e-5, m_star=3, kominus=0.5)
 
    # fibril mass and oligomer concentration side by side
    fig, axes = plt.subplots(1, 2)
    for m0 in m0vals:
        t, M_norm, S = model.simulate(m0, free_params)
        axes[0].plot(t, M_norm, label=f'{m0}')
        axes[1].plot(t, S, label=f'{m0}')
    axes[0].set_xlabel('time (h)')
    axes[1].set_xlabel('time (h)')
    axes[0].set_ylabel('normalised fibril mass')
    axes[1].set_ylabel('oligomer concentration')
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()
 
    # half-time scaling
    half_times = []
    for m0 in m0vals:
        t, M_norm, _ = model.simulate(m0, free_params)
        half_times.append(model.get_half_time(t, M_norm))
    half_times = np.array(half_times)
 
    plt.figure()
    plt.scatter(np.log10(m0vals), np.log10(half_times))
    plt.xlabel('log m0')
    plt.ylabel('log t_half')
    plt.title('Delayed oligomers half time plot')
    plt.show()





if __name__ == "__main__":
    plot_delayed_model()

