import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# extract early time slope from a single aggregation curve
def get_early_slope(time, signal, threshold):
    # threshold = which signal values (0-1) to consider at the beginning to compute the slope
    # select those points
    mask = signal < threshold
    if mask.sum() < 5:
        raise ValueError('increase threshold, not enough points to compute the slope')

    early_time = time[mask]
    early_signal = signal[mask]

    slope, intercept, rvalue, pvalue, se = stats.linregress(early_time, early_signal)

    return slope



# read normalized seeded data
# obtained in amylofit 
data_path = '/Users/nataliaionescu/Desktop/AB42_project/seeded_analysis/pH_6.5_no_0.8/data.tsv' 
df = pd.read_csv(data_path, sep = '\t', header=1)
current_df = df
x_actual_data = []
y_actual_data = []

for i in range(len(current_df.columns) // 2):
    x_current = current_df.iloc[:, i*2].values
    y_current = current_df.iloc[:, i*2+1].values
    x_actual_data.append(x_current[~np.isnan(x_current)])
    y_actual_data.append(y_current[~np.isnan(y_current)])

m0vals = [6, 6, 6, 4.5, 4.5, 4.5, 3.4, 3.4, 3.4, 2.5, 2.5, 2.5, 1.9, 1.9, 1.9, 1.4, 1.4, 1.4, 1.1, 1.1, 1.1 ]

# visualize curves, labeled by m0
sns.set_theme(style='whitegrid', context='paper')
unique_m0 = sorted(set(m0vals))
palette = sns.color_palette('tab10', n_colors = len(unique_m0))
color_map = {m0: palette[i] for i, m0 in enumerate(unique_m0)}

fig = plt.figure(figsize=(10,7))
seen_m0 = set()
for i in range(len(x_actual_data)):
    x = x_actual_data[i]
    y = y_actual_data[i]
    m0 = m0vals[i]
    color = color_map[m0]

    data_label = f'{m0}' if m0 not in seen_m0 else None
    seen_m0.add(m0)
    plt.scatter(x, y, color=color, label=data_label)
plt.legend()
plt.show()

# get early time slopes for each m0 curve
# and add linear fits to the plot
slopes = []

for i in range(len(x_actual_data)):
    time = x_actual_data[i]
    signal = y_actual_data[i]
    slope = get_early_slope(time, signal, threshold=0.3)
    slopes.append(slope)


# average slopes for each m0 before plotting
slopes = np.array(slopes)
m0vals_arr = np.array(m0vals)
mean_slopes = []
for m0 in unique_m0:
    mask = m0vals_arr == m0
    print('masked m0', m0vals_arr[mask])
    print('masked slopes', slopes[mask])
    mean_slopes.append(slopes[mask].mean()*m0) 

print('mean slopes', mean_slopes)

# plot avg slope for each m0 vs m0
fig, ax = plt.subplots(figsize=(10,7))
ax.scatter(unique_m0, mean_slopes)
ax.set_xlabel('m0')
ax.set_ylabel('mean early slope')
plt.show()

# fit KE
def saturated_elongation(m0, A, KE):
    return A * m0 / (1 + m0/KE)

m0_arr = np.array(unique_m0)
slope_arr = np.array(mean_slopes)

# initial guesses
A0 = slope_arr[-1] / m0_arr[-1]
KE0 = m0_arr[-1]

popt, pcov = curve_fit(saturated_elongation, m0_arr, slope_arr, p0=[A0, KE0], bounds=([0,0], [np.inf, np.inf]), maxfev=10000)
A_fit, KE_fit = popt
A_err, KE_err = np.sqrt(np.diag(pcov))
print(f'KE = {KE_fit:.4f} +- {KE_err:.4f}')




# if this were linear, it would mean elongation is not saturated
# however, it looks very weird, can KE be fit from here?

# adding oligomers: constant monomer conc at equilibrium -> saturation-like effects, no rates depend on m0 anymore
# multi-saturation: very sharp curve in the first half
# oligomers: shape of second half of the curve given by oligomer dissociation, exponential decay-like process
# kinetic model from oligomer papers, but they are not able to have a sharp transition in the equilibrium oligomer conc when increasing m0
# come up with simple rate eqs that do that with oligomer kinetics and CMC
# test by numerical simulation to test that they give wanted behaviour: oligomers above a certain m0 have a high conc in eq with const monomer conc: saturates fibril mass concentraion; simulate for several m0 at the same time
# also exponential decay curve
# fit model to the data or find analytical solution and do analytical fitting

# constrain unseeded data: plot unseeded half times to constrain n2, KS, KE (see 2020 paper)
# 1st half -0.5 slope, set n2=0 and KS arbitrarily large, 2nd nucl fully saturated at every conc
# can pretend oligomers are off-pathway and decouple models of fibril formation and oligomer creation



